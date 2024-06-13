from operator import attrgetter

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.lib import hub
from ryu.base.app_manager import lookup_service_brick

import setting


class NetworkMonitor(app_manager.RyuApp):
    """
    监控网络流量状态
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]  # openflow协议版本

    def __init__(self, *_args, **_kwargs):
        super(NetworkMonitor, self).__init__(*_args, **_kwargs)
        self.name = 'monitor'
        self.datapaths_table = {}  # 交换机通道信息{dpid: datapath}
        self.dpid_port_fueatures_table = {}  # 交换机端口信息{dpid:{port_no: (config, state, curr_speed, max_speed)}}
        self.port_stats_table = {}  # 端口状态{(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec, tat.duration_nsec, stat.tx_packets, stat.rx_packets)}
        self.flow_stats_table = {}  # 流表状态{dpid:{(in_port, ipv4_dsts, out_port): (packet_count, byte_count, duration_sec, duration_nsec)}}
        self.port_speed_table = {}  # 端口流量速度{(dpid, port_no): [speed, .....]}
        self.flow_speed_table = {}  # 发包的流速{dpid: {(in_port, ipv4_dsts, out_port): speed}}

        self.links_free_bandwidth = {}  # 链路的剩余带宽 {dpid: {port_no: free_bw}}
        self.port_flow_dpid_stats = {'port': {}, 'flow': {}}  # 交换机端口流量速度
        self.port_curr_speed = {}  # 端口当前流量速度{dpid: {port_no: curr_bw}}
        self.port_loss = {}  # 端口的丢包率
        self.port_pkt_err = {} # 端口错包率
        self.port_pkt_drop = {}  # 端口弃包个数

        self.network_structure = lookup_service_brick("discovery")  # 创建一个NetworkStructure的实例，不同模块之间的数据通信

        # self.monitor_thread = hub.spawn(self.scheduler)  # 启动协程
        self.monitor_thread = hub.spawn(self.monitor)
        # self.save_thread = hub.spawn(self.save_bw_loss_graph)

    def scheduler(self):
        """
        调度协程
        """
        self._request_stats()
        hub.sleep(setting.MONITOR_PERIOD)
        self.create_bandwidth_graph()
        self.create_loss_graph()
        if setting.PRINT_SHOW:
            self.print_parameters()
        
    
    def monitor(self):
        hub.sleep(setting.DISCOVERY_PERIOD)
        while True:
            hub.sleep(setting.MONITOR_PERIOD)
            # print("monitor\n----->self.datapaths_table", self.datapaths_table)
            self._request_stats()
            self.create_bandwidth_graph()
            self.update_graph_loss()
            # if setting.PRINT_SHOW:
            #     self.print_parameters()

    
    def save_bw_loss_graph(self):
        """保存带宽、丢包率"""
        while True:
            self.create_bandwidth_graph()
            self.create_loss_graph()
            hub.sleep(setting.MONITOR_PERIOD)

    def print_parameters(self):
        self.logger.info("monitor---->===================================%s=================================",
                         self.name)
        print("monitor\n----->self.datapaths_table", self.datapaths_table)
        print("monitor\n----->self.dpid_port_fueatures_table", self.dpid_port_fueatures_table)
        print("monitor\n----->self.port_stats_table", self.port_stats_table)
        print("monitor\n----->self.flow_stats_table", self.flow_stats_table)
        print("monitor\n----->self.port_speed_table", self.port_speed_table)
        print("monitor\n----->self.flow_speed_table", self.flow_speed_table)
        print("monitor\n----->self.port_free_bandwidth", self.port_free_bandwidth)
        self.logger.info("===================monitor=======================")

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """
        存放所有的datapath实例
        获取交换机连接的信息（改变），并处理该事件
        """
        datapath = ev.datapath  # 获取交换机
        # dpid =  self.network_structure.apid_dict.get(datapath.id)
        if ev.state == MAIN_DISPATCHER:  # 交换机上线
            if datapath.id not in self.datapaths_table:
                print("MMMM--->  register datapath: %016x" % datapath.id)

                dpid =  self.network_structure.apid_dict.get(datapath.id)
                self.datapaths_table[dpid] = datapath  # 记录未记录的交换机datapath

                # 一些初始化
                self.dpid_port_fueatures_table.setdefault(dpid, {})
                self.flow_stats_table.setdefault(dpid, {})

        elif ev.state == DEAD_DISPATCHER:  # 交换机下线
            if datapath.id in self.datapaths_table:
                print("MMMM--->  unreigster datapath: %016x" % datapath.id)
                del self.datapaths_table[datapath.id]  # 清除在记录中的下线的交换机

    def _request_stats(self):
        """
        主动发送request, 请求状态信息
        """
        # print("MMMM--->  send request --->   ---> send request ---> ")

        for datapath in self.datapaths_table.values():
            # self.dpid_port_fueatures_table.setdefault(datapath.id, {})
            # print("MMMM--->  send stats request: %016x", datapath.id)
            ofproto = datapath.ofproto  # 提取OpenFlow的版本信息
            parser = datapath.ofproto_parser  # 获取OpenFlow解析器

            # 1. 端口描述请求
            req = parser.OFPPortDescStatsRequest(datapath, 0)
            datapath.send_msg(req)

            # 2. 端口统计请求
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)  # 实例化端口请求报文发送类
            datapath.send_msg(req)

            # 3. 单个流统计请求
            req = parser.OFPFlowStatsRequest(datapath)  # 实例化流表项请求报文发送类
            datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """
        存储端口描述信息, 见OFPPot类, 配置、状态、当前速度
        """
        # print("MMMM--->  EventOFPPortDescStatsReply")
        msg = ev.msg  # 监听到事件的消息
        
        datapath = msg.datapath  # 数据平面的通道
        dpid = self.network_structure.apid_dict.get(datapath.id)

        ofproto = msg.datapath.ofproto  # OpenFlow的版本

        # 给openflow的配置信息取别名
        config_dict = {ofproto.OFPPC_PORT_DOWN: 'Port Down',
                       ofproto.OFPPC_NO_RECV: 'No Recv',
                       ofproto.OFPPC_NO_FWD: 'No Forward',
                       ofproto.OFPPC_NO_PACKET_IN: 'No Pakcet-In'}
        # config_dict--------> {1: 'Port Down', 4: 'No Recv', 32: 'No Forward', 64: 'No Pakcet-In'}
        # print("config_dict-------->",config_dict)

        # 给链路状态信息取别名
        state_dict = {ofproto.OFPPS_LINK_DOWN: "Link Down",
                      ofproto.OFPPS_BLOCKED: "Blocked",
                      ofproto.OFPPS_LIVE: "Live"}
        # state_dict--------> {1: 'Link Down', 2: 'Blocked', 4: 'Live'}
        # print("state_dict-------->", state_dict)

        for ofport in ev.msg.body:  # 获取事件的消息体
            if ofport.port_no != ofproto_v1_3.OFPP_LOCAL:  # 0xfffffffe  4294967294
                if ofport.config in config_dict:
                    config = config_dict[ofport.config]
                else:
                    config = 'Up'

                if ofport.state in state_dict:
                    state = state_dict[ofport.state]
                else:
                    state = 'Up'

                # 存储配置，状态，curr_speed,max_speed=0
                port_features = (config, state, ofport.curr_speed, ofport.max_speed)
                # print("MMMM--->  ofport.curr_speed", ofport.curr_speed)
                self.dpid_port_fueatures_table[dpid][ofport.port_no] = port_features

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_table_reply_handler(self, ev):
        """ 
        存储端口统计信息, 见OFPPortStats, 发送bytes、接收bytes、生效时间duration_sec等
         Replay message content:
            (stat.port_no,
             stat.rx_packets, stat.tx_packets,
             stat.rx_bytes, stat.tx_bytes,
             stat.rx_dropped, stat.tx_dropped,
             stat.rx_errors, stat.tx_errors,
             stat.rx_frame_err, stat.rx_over_err,
             stat.rx_crc_err, stat.collisions,
             stat.duration_sec, stat.duration_nsec)
        """
        # print("MMMM--->  EventOFPPortStatsReply")
        body = ev.msg.body
        dpid = self.network_structure.apid_dict.get(ev.msg.datapath.id)
        self.port_flow_dpid_stats['port'][dpid] = body
        # print("self.port_flow_dpid_stats",self.port_flow_dpid_stats)

        for stat in sorted(body, key=attrgetter("port_no")):
            port_no = stat.port_no
            if port_no != ofproto_v1_3.OFPP_LOCAL:
                key = (dpid, port_no)
                value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,
                         stat.duration_sec, stat.duration_nsec, stat.tx_packets, stat.rx_packets)
                self._save_stats(self.port_stats_table, key, value, 5)  # 保存信息，最多保存前5次

                pre_bytes = 0
                delta_time = setting.MONITOR_PERIOD  # 每10s检测一次网络流量状态
                stats = self.port_stats_table[key]  # 获得已经存了的统计信息
                # print("stats-------->", stats)

                if len(stats) > 1:  # 有两次以上的信息
                    pre_bytes = stats[-2][0] + stats[-2][1]
                    delta_time = self._calculate_delta_time(stats[-1][3], stats[-1][4],
                                                            stats[-2][3], stats[-2][4])  # 倒数第一个统计信息，倒数第二个统计信息

                speed = self._calculate_speed(stats[-1][0] + stats[-1][1],
                                                  pre_bytes, delta_time)

                self._save_stats(self.port_speed_table, key, speed, 5)
                self._calculate_port_speed(dpid, port_no, speed)    #师兄的更新带宽函数
                # self._calculate_links_free_bandwidth(dpid, port_no, speed)

        self.calculate_loss_of_link()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """
        存储流表统计信息
        """
        msg = ev.msg  # 获取消息
        body = msg.body  # 获取时间的消息体
        datapath = msg.datapath  # 获取数据平面的通道
        dpid = self.network_structure.apid_dict.get(datapath.id)  # 获取AP

        self.port_flow_dpid_stats['flow'][dpid] = body
        # print("MMMM--->  body", body)

        for stat in sorted([flowstats for flowstats in body if flowstats.priority == 1],
                           key=lambda flowstats: (flowstats.match.get('in_port'),
                                                  flowstats.match.get('ipv4_dst'))):
            # print("MMMM--->  stat.match", stat.match)
            # print("MMMM--->  stat", stat)
            key = (stat.match['in_port'], stat.match['ipv4_dst'],
                   stat.instructions[0].actions[0].port)
            value = (stat.packet_count, stat.byte_count, stat.duration_sec, stat.duration_nsec)
            self._save_stats(self.flow_stats_table[dpid], key, value, 5)

            pre_bytes = 0
            delta_time = setting.MONITOR_PERIOD
            # delta_time = setting.SCHEDULE_PERIOD
            value = self.flow_stats_table[dpid][key]
            if len(value) > 1:
                pre_bytes = value[-2][1]
                # print("MMMM--->  _flow_stats_reply_handler delta_time: now", value[-1][2], value[-1][3], "pre", value[-2][2], value[-2][3])
                delta_time = self._calculate_delta_time(value[-1][2], value[-1][3],
                                                        value[-2][2], value[-2][3])
            speed = self._calculate_speed(self.flow_stats_table[dpid][key][-1][1], pre_bytes, delta_time)
            self.flow_speed_table.setdefault(dpid, {})
            self._save_stats(self.flow_speed_table[dpid], key, speed, 5)

        self.calculate_loss_of_link()

    @staticmethod
    def _save_stats(_dict, key, value, keep):
        """
        存储端口或流表的统计数据,只存最新的5条
        """
        if key not in _dict:
            _dict[key] = []
        _dict[key].append(value)

        if len(_dict[key]) > keep:
            _dict[key].pop(0)  # 弹出最早的数据

    def _calculate_delta_time(self, now_sec, now_nsec, pre_sec, pre_nsec):
        """
        计算统计时间，即两个消息时间差
        """
        return self._calculate_seconds(now_sec, now_nsec) - self._calculate_seconds(pre_sec, pre_nsec)

    @staticmethod
    def _calculate_seconds(sec, nsec):
        """
        计算sec + nsec 的和， 单位为seconds
        """
        return sec + nsec / 10 ** 9

    @staticmethod
    def _calculate_speed(now_bytes, pre_bytes, delta_time):
        """
        计算链路的流量速度（带宽）
        (now_bytes - pre_bytes) / delta_time
        """
        if delta_time:
            return (now_bytes - pre_bytes) / delta_time
        else:
            return 0

    def _calculate_port_speed(self, dpid, port_no, speed):
        """
        计算端口的流量速度
        """
        curr_bw = speed * 8 / 10 ** 6  # MBit/s
        # print(f"monitorMMMM---> _calculate_port_speed: {curr_bw} MBits/s", )
        self.port_curr_speed.setdefault(dpid, {})
        self.port_curr_speed[dpid][port_no] = curr_bw

    def _calculate_links_free_bandwidth(self, dpid, port_no, speed):
        """
        计算当前剩余带宽
        """
        port_features = self.dpid_port_fueatures_table.get(dpid).get(port_no)
        print("""port_features--------->""", port_features)
        if port_features:
            capacity = port_features[2]  # 链路设置的带宽
            print("capacity--------->", capacity)
            free_bw = max(capacity / 10 * 3 - speed * 8 / 10 ** 9, 0)  # 当前链路的剩余带宽 bit/s
            print("free_bw--------->", free_bw/10**6)

            self.links_free_bandwidth.setdefault(dpid, {})
            self.links_free_bandwidth[dpid][port_no] = free_bw / 10 ** 6  # MBit/s
        else:
            self.logger.info("monitor---->Fail in getting port features")

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        """ 
        处理端口状态： ADD, DELETE, MODIFIED
        """
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto

        if msg.reason == ofp.OFPPR_ADD:
            reason = 'ADD'
        elif msg.reason == ofp.OFPPR_DELETE:
            reason = 'DELETE'
        elif msg.reason == ofp.OFPPR_MODIFY:
            reason = 'MODIFY'
        else:
            reason = 'unknown'

        print('MMMM---> _port_status_handler OFPPortStatus received: reason=%s desc=%s' % (reason, msg.desc))

    def create_bandwidth_graph(self):
        """
        通过获得的网络拓扑,更新其bw
        """
        # print("MMMM--->  create bandwidth graph")
        link_port_table = self.network_structure.link_port_table
        for link in link_port_table:
            src_dpid, dst_dpid = link
            src_port, dst_port = link_port_table[link]

            if src_dpid in self.port_curr_speed.keys() and dst_dpid in self.port_curr_speed.keys():
                src_port_bw = self.port_curr_speed[src_dpid][src_port]  # 取两个交换机连接的源端口剩余带宽
                # print("monitor------> src_port_bw", src_port_bw)
                dst_port_bw = self.port_curr_speed[dst_dpid][dst_port]  # 取两个交换机连接的目的端口剩余带宽
                # print("monitor------->dst_port_bw", dst_port_bw)
                src_dst_bandwidth = min(src_port_bw, dst_port_bw)  # 取两者之间的最小值作为当前链路的剩余带宽
                # print("!!!!!!!monitor------> src_dst_bandwitdh", src_dst_bandwitdh)
                self.network_structure.graph[src_dpid][dst_dpid]['used_bw'] = min(src_port_bw, dst_port_bw)

                # 对图的edge设置bw值
                capacity = self.network_structure.m_graph[src_dpid][dst_dpid]['free_bw']
                # print("capacity--------->", capacity)
                self.network_structure.graph[src_dpid][dst_dpid]['free_bw'] = max(capacity - src_dst_bandwidth, 0)

                # 对图的edge设置距离
                distance = self.network_structure.m_graph[src_dpid][dst_dpid]['distance']
                self.network_structure.graph[src_dpid][dst_dpid]['distance'] = distance


            else:
                # self.logger.info("MMMM--->  create_bandwidth_graph: [{}] [{}] not in port_free_bandwidth "
                #                  .format(src_dpid, dst_dpid))
                self.network_structure.graph[src_dpid][dst_dpid]['free_bw'] = 0

        # print("monitor---> ", self.network_structure.graph.edges(data=True))
        # return self.network_structure.graph
    

    # calculate loss tx - rx / tx
    def calculate_loss_of_link(self):
        """
            发端口 和 收端口 ,端口loss
        """
        for link, port in self.network_structure.link_port_table.items():
            src_dpid, dst_dpid = link
            src_port, dst_port = port
            if (src_dpid, src_port) in self.port_stats_table.keys() and \
                    (dst_dpid, dst_port) in self.port_stats_table.keys():
                # {(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec,
                # stat.duration_nsec, stat.tx_packets, stat.rx_packets)}
                # 1. 顺向  2022/3/11 packets modify--> bytes
                #计算loss，丢包率
                tx = self.port_stats_table[(src_dpid, src_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(dst_dpid, dst_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link, loss_ratio, 5)
                # print(f"MMMM--->[{link}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)

                # 计算错包率
                rx_err = self.port_stats_table[(src_dpid, src_port)][-1][2] # rx_err_bytes 
                rx = self.port_stats_table[(dst_dpid, dst_port)][-1][1]  # rx_bytes
                pkt_err = (rx_err/rx) * 100
                self._save_stats(self.port_pkt_err, link, pkt_err, 5)

                # 计算弃包个数
                tx_packets = self.port_stats_table[(src_dpid, src_port)][-1][-2] # tx_packets
                rx_packets = self.port_stats_table[(dst_dpid, dst_port)][-1][-1]  # rx_packets
                pkt_drop = abs(tx_packets - rx_packets)
                self._save_stats(self.port_pkt_drop, link, pkt_drop, 5)



                # 2. 逆项
                tx = self.port_stats_table[(dst_dpid, dst_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(src_dpid, src_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link[::-1], loss_ratio, 5)

                # print(f"MMMM--->[{link[::-1]}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)

                # 计算错包率
                rx_err = self.port_stats_table[(dst_dpid, dst_port)][-1][2] # rx_err_bytes 
                rx = self.port_stats_table[(src_dpid, src_port)][-1][1]  # rx_bytes
                pkt_err = (rx_err/rx) * 100
                self._save_stats(self.port_pkt_err, link, pkt_err, 5)

                # 计算弃包个数
                tx_packets = self.port_stats_table[(dst_dpid, dst_port)][-1][-2] # tx_packets
                rx_packets = self.port_stats_table[(src_dpid, src_port)][-1][-1]  # rx_packets
                pkt_drop = abs(tx_packets - rx_packets)
                self._save_stats(self.port_pkt_drop, link, pkt_drop, 5)


            else:
                # self.logger.info("MMMM--->  calculate_loss_of_link error", )
                pass

    # update graph loss
    def update_graph_loss(self):
        """从1 往2 和 从2 往1,取最大作为链路loss """
        for link in self.network_structure.link_port_table:
            src_dpid = link[0]
            dst_dpid = link[1]
            if link in self.port_loss.keys() and link[::-1] in self.port_loss.keys():
                src_loss = self.port_loss[link][-1]  # 1-->2  -1取最新的那个
                dst_loss = self.port_loss[link[::-1]][-1]  # 2-->1
                link_loss = max(src_loss, dst_loss)  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['loss'] = link_loss

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['loss'] = 100
            
            if link in self.port_pkt_err.keys() and link[::-1] in self.port_pkt_err.keys():
                src_pkt_err = self.port_pkt_err[link][-1]  # 1-->2  -1取最新的那个
                dst_pkt_err = self.port_pkt_err[link[::-1]][-1]  # 2-->1
                link_pkt_err = max(src_pkt_err , dst_pkt_err )  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_err'] = link_pkt_err

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_err'] = -1
            
            if link in self.port_pkt_drop.keys() and link[::-1] in self.port_pkt_drop.keys():
                src_pkt_drop = self.port_pkt_drop[link][-1]  # 1-->2  -1取最新的那个
                dst_pkt_drop = self.port_pkt_drop[link[::-1]][-1]  # 2-->1
                link_pkt_drop = max(src_pkt_drop, dst_pkt_drop)  # 百分比 max loss between port1 and port2
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_drop'] = link_pkt_drop

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.network_structure.graph[src_dpid][dst_dpid]['pkt_drop'] = -1

    def create_loss_graph(self):
        """
            在graph中更新边的loss值
        """
        # self.calculate_loss_of_link()
        self.update_graph_loss()
