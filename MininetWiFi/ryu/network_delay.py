import time

from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

from ryu.topology.switches import Switches, LLDPPacket

import setting
import network_structure
import network_monitor


class NetworkDelayDetector(app_manager.RyuApp):
    """
    测量链路时延
                        ┌------Ryu------┐
                        |               |
        src echo latency|               |dst echo latency
                        |               |
                    SwitchA------------SwitchB
                         --->fwd_delay--->
                         <---reply_delay<---

    1、发送时延:发送时延是主机或者路由器发送数据帧所需要的时延; 发送时延=数据帧长度/发送的速率
    2、传播时延:传播时延是电磁波在信道中传播一定的距离花费的时间; 传播时延=传输媒介长度/电磁波在信道上的传播速率
    3、处理时延:主机或者路由器处理数据花费的时间; 
    4、排队时延:数据在进入路由器后排队等待的时间;
    """
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]  # openflow协议版本
    _CONTEXTS = {'switches': Switches}

    def __init__(self, *_args, **_kwargs):
        super(NetworkDelayDetector, self).__init__(*_args, **_kwargs)
        self.name = 'detector'

        self.network_structure = lookup_service_brick('discovery')
        self.network_monitor = lookup_service_brick('monitor')
        self.switch_module = lookup_service_brick('switches')
        # self.switch_module = _kwargs['switches']

        self.echo_delay_table = {}  # 控制器到交换机的往返时延，echo报文测量{dpid: ryu_ofps_delay}
        self.lldp_delay_table = {}  # 控制器--> 交换机1 --> 交换机2 --> 控制器; LLDP(Link Layer Discovery Protocol，链路发现协议){src_dpid: {dst_dpid: delay}}
        self.echo_interval = 0.05  # echo包发送的间隔时间

        self._detector_thread = hub.spawn(self._detector)  # 内部模块测试协程
        # self.datapaths_table = self.network_monitor.datapaths_table
        # self._delay_thread = hub.spawn(self.scheduler)

    def _detector(self):
        """
        内部调用协程
        """
        hub.sleep(setting.DISCOVERY_PERIOD)
        hub.sleep(setting.DISCOVERY_PERIOD)
        while True:
            hub.sleep(setting.DELAY_PERIOD)
            self._send_echo_request()
            self.create_delay_graph()
            if setting.PRINT_SHOW:
                self.show_delay_stats()

    def scheduler(self):
        """
        外部调用协程
        """
        self._send_echo_request()
        self.create_delay_graph()
        if setting.PRINT_SHOW:
            self.show_delay_stats()

        # hub.sleep(setting.DElAY_PERIOD)

    # 1、 利用echo报文得到：控制器<-->交换机的往返时延
    def _send_echo_request(self):
        """
        控制器发送echo请求到交换机, 记录当前发送时间
        """
        datapaths_table = self.network_monitor.datapaths_table.values()
        if datapaths_table is not None:
            for datapath in list(datapaths_table):
                parser = datapath.ofproto_parser  # 获取OpenFlow解析器
                data = time.time()  # 记录当前时间
                echo_req = parser.OFPEchoRequest(datapath, b"%.12f" % data)  # 将发送出的时间戳作为数据发送出去
                datapath.send_msg(echo_req)
                hub.sleep(self.echo_interval)  # 防止发送的太快，控制器接收不到echo回复

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        """
        控制器接收到echo报文回复,并处理
        """
        now_timestamp = time.time()  # 记录收到echo回复的时间，与发送echo记录的时间相减，即得到控制器<-->交换机的往返时延
        data = ev.msg.data  # 解析出消息体中的数据; 即发送过来的发送时间戳
        ryu_ofps_delay = now_timestamp - eval(data)  # 现在的时间减去发送的时间
        dpid = self.network_structure.apid_dict.get(ev.msg.datapath.id)
        self.echo_delay_table[dpid] = ryu_ofps_delay

    # 2、利用LLDP报文得到：控制器--> 交换机1 --> 交换机2 --> 控制器的时延
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
        解析LLDP包;(这个处理程序可以接收所有可以接收的数据包, swicthes.py l:769)
        """
        # print("detector---> PacketIn")
        try:
            recv_timestamp = time.time()
            msg = ev.msg
            dpid = self.network_structure.apid_dict.get(msg.datapath.id)
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)

            # print("DDDD--->  register datapath: %016x" % msg.datapath.id)
            # print("---> self.switch_module.ports", self.switch_module.ports)

            for port in self.switch_module.ports.keys():
                if src_dpid == port.dpid and src_port_no == port.port_no:
                    send_timestamp = self.switch_module.ports[port].timestamp
                    if send_timestamp:
                        delay = recv_timestamp - send_timestamp
                    else:
                        delay = 0
                    
                    src_dpid = self.network_structure.apid_dict.get(src_dpid)
                    
                    self.lldp_delay_table.setdefault(src_dpid, {})
                    self.lldp_delay_table[src_dpid][dpid] = delay  # 存起来

        except LLDPPacket.LLDPUnknownFormat as e:
            return

    def create_delay_graph(self):
        """
        遍历所有的边,并计算出所有链路的往返时延
        """
        # print('---> create delay graph')
        for src, dst in self.network_structure.graph.edges:
            delay = self.calculate_delay(src, dst)
            self.network_structure.graph[src][dst]['delay'] = delay  # s

    def calculate_delay(self, src, dst):
        """
                        ┌------Ryu------┐
                        |               |
        src echo latency|               |dst echo latency
                        |               |
                    SwitchA------------SwitchB
                         --->fwd_delay--->
                         <---reply_delay<---
        """
        fwd_delay = self.lldp_delay_table[src][dst]  # 控制器-->交换机B(src)-->交换机A(dst)-->控制器
        reply_delay = self.lldp_delay_table[dst][src]  # 控制器-->交换机A(dst)-->交换机B(src)-->控制器
        ryu_ofps_src_delay = self.echo_delay_table[src]  # 控制器<-->交换机B(src)
        ryu_ofps_dst_delay = self.echo_delay_table[dst]  # 控制器<-->交换机A(dst)

        # 计算：交换机A<-->交换机B 链路的往返时延（RTT）
        delay = (fwd_delay + reply_delay - ryu_ofps_src_delay - ryu_ofps_dst_delay) / 2
        return max(delay*1000, 0)

    def show_delay_stats(self):
        """
        打印每条链路的时延
        """
        self.logger.info("==============================DDDD delay=================================")
        self.logger.info("src    dst :    delay")
        for src in self.lldp_delay_table.keys():
            for dst in self.lldp_delay_table[src].keys():
                delay = self.lldp_delay_table[src][dst]
                self.logger.info("%s <---> %s : %s", src, dst, delay*1000)
