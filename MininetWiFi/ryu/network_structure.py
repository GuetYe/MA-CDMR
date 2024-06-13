import copy
import time  # 导入时间模块
import xml.etree.ElementTree as ET  # 解析xml树形结构

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.lib import hub
from ryu.lib import igmplib, mac
from ryu.lib.dpid import str_to_dpid
from ryu.lib.packet import packet, arp, ethernet, ipv4, igmp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link, get_host

import networkx as nx

import setting



class NetworkStructure(app_manager.RyuApp):
    """
    发现网络拓扑，保存网络结构
    """
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]  # 定义openflow协议版本

    # _CONTEXTS = {'igmplib': igmplib.IgmpLib}

    def __init__(self, *_args, **_kwargs):
        super(NetworkStructure, self).__init__(*_args, **_kwargs)
        self.start_time = time.time()  # 记录开始执行时间
        self.name = 'discovery'

        self.topology_api_app = self
        self.ap_distance = {} # ap's distance 必须放在解析topo前
        self.link_info_xml = setting.LINKS_INFO  # xml file path of links info
        self.m_graph = self.parse_topo_links_info()  # 解析mininet构建的topo链路信息

        self.graph = nx.Graph()  # 存放根据发现拓扑得到的链路和端口信息新建的拓扑图
        self.pre_graph = nx.Graph()
        self.access_table = {}  # 存放主机信息（交换机， 端口号）: {源ip, 源mac}信息=》(dpid, in_port): (src_ip, src_mac)}
        self.switch_all_ports_table = {}  # {dpid: {port_no, ...}}交换机所有的端口
        self.all_switches_dpid = self.switch_all_ports_table.keys()  # dict_key[dpid]所有交换机
        self.switch_port_table = {}  # {dpid: {port, ...} 交换机中被使用的端口
        self.link_port_table = {}  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}链路连接的端口信息
        self.not_use_ports = {}  # {dpid: {port, ...}}  交换机之间没有用来连接的port
        self.shortest_path_table = {}  # {(src.dpid, dst.dpid): [path]}源交换机到目的交换机的最短路径
        self.arp_table = {}  # {(dpid, eth_src, arp_dst_ip): in_port}路由转发表
        self.arp_src_dst_ip_table = {}
        self.apid_dict = {}  # ap id dict
        self.apid_list = []  # ap id list

        self.initiation_delay = setting.INIT_TIME  # 初始化时延时间
        self.first_flag = True
        self.cal_path_flag = False

        # self._structure_thread = hub.spawn(self._discover_network_structure)
        self._structure_thread = hub.spawn(self.scheduler)
        self._short_path_thread = hub.spawn(self.cal_shortest_path_thread)

    def _discover_network_structure(self):
        while True:
            self.get_topology(None)
            hub.sleep(setting.DISCOVERY_PERIOD)
            if self.graph.edges:
                if self.pre_graph.edges != self.graph.edges or first_flag:
                    self.print_parameters()
                    first_flag = False
        
    def scheduler(self):
        """
        调度协程
        """
        i = 0
        while True:
            if i == 3:
                self.get_topology(None)
                i = 0
            hub.sleep(setting.DISCOVERY_PERIOD)
            if setting.PRINT_SHOW:
                self.print_parameters()
            i += 1


    def print_parameters(self):
        self.logger.info("discovery--->========================== %s ==========================", self.name)
        self.logger.info("discovery---> graph: %s", self.graph.edges)
        # self.logger.info("discovery---> access_table: %s", self.access_table)
        # self.logger.info("discovery---> switch_all_ports_table: %s", self.switch_all_ports_table)
        # self.logger.info("discovery---> switch_port_table: %s", self.switch_port_table)
        # self.logger.info("discovery---> link_port_table: %s", self.link_port_table)
        # self.logger.info("discovery---> not_use_ports: %s", self.not_use_ports)
        # self.logger.info("discovery---> shortest_path_table: %s", self.shortest_path_table)
        self.logger.info("discovery--->==============================")


    def cal_shortest_path_thread(self):
        # self.cal_path_flag = False
        # while True:
        #     if self.cal_path_flag:
        self.calculate_all_nodes_shortest_paths(weight=setting.WEIGHT)
                # print("*****discovery_topo---> self.shortest_path_table:\n", self.shortest_path_table)
            # hub.sleep(setting.DISCOVERY_PERIOD)

    # Flow mod and Table miss
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # ------------------------无线dpid映射处理1-----------------------------------#
        self.apid_list.append(datapath.id)
        self.apid_list.sort()
        # print("!!!!--->", self.apid_list)
        # self.apid_dict = {}
        for t in range(1, len(self.apid_list) + 1):
            self.apid_dict[self.apid_list[t-1]] = t  # 将id存进字典中

        self.logger.info("discovery_topo---> switch: %s connected", self.apid_dict.get(datapath.id))

        # install table miss flow entry
        match = parser.OFPMatch()  # match all
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        inst = [datapath.ofproto_parser.OFPInstructionActions(datapath.ofproto.OFPIT_APPLY_ACTIONS,
                                                               actions)]
        mod = datapath.ofproto_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                                 match=match, instructions=inst)
        datapath.send_msg(mod)

    # 解析Packet In
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # print("discovery_topo---> discovery PacketIn")
        msg = ev.msg
        datapath = msg.datapath

        # 输入端口号
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)

        if isinstance(arp_pkt, arp.arp):
            # print("SSSS---> _packet_in_handler: arp packet")
            arp_src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            self.storage_access_info(self.apid_dict.get(datapath.id), in_port, arp_src_ip, src_mac)
            # print("discovery_topo--->  access_table:\n    ", self.access_table)

    # 将packet-in解析的arp的网络通路信息存储
    def storage_access_info(self, dpid, in_port, src_ip, src_mac):
        # print(f"SSSS--->storage_access_info, self.access_table: {self.access_table}")
        if in_port in self.not_use_ports[dpid]:
            # print("discovery_topo--->", dpid, in_port, src_ip, src_mac)
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (src_ip, src_mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                return

    # 利用topology库获取拓扑信息
    events = [event.EventSwitchEnter, event.EventSwitchLeave,
              event.EventPortAdd, event.EventPortDelete, event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]

    @set_ev_cls(events)
    def get_topology(self, ev):
        # present_time = time.time()
        # if present_time - self.start_time < self.initiation_delay:  # set to 30s
        #     print(f'SSSS--->get_topology: need to WAIT{self.initiation_delay - (present_time - self.start_time):.2f}s')
        #     return
        # elif self.first_flag:
        #     self.first_flag = False
        #     print("SSSS--->get_topology: complete WAIT")

        # self.logger.info("[Topology Discovery Ok]")

        # 事件发生时，获得switch列表
        switch_list = get_switch(self.topology_api_app, None)
        # 将switch添加到self.switch_all_ports_table
        # ------------------------无线dpid映射处理2-----------------------------------#
        for switch in switch_list:
            dpid = switch.dp.id
            self.switch_all_ports_table.setdefault(self.apid_dict.get(dpid), set())
            self.switch_port_table.setdefault(self.apid_dict.get(dpid), set())
            self.not_use_ports.setdefault(self.apid_dict.get(dpid), set())
            # print("discovery_topo---> ",switch, switch.ports)
            for p in switch.ports:
                self.switch_all_ports_table[self.apid_dict.get(dpid)].add(p.port_no)

        self.all_switches_dpid = self.switch_all_ports_table.keys()
        # time.sleep(0.5)

        # 获得link
        link_list = get_link(self.topology_api_app, None)
        self.link_port_table = {}
        # print("discovery---> ",len(link_list))

        # ------------------------无线dpid映射处理3-----------------------------------#
        # 将link添加到self.link_table
        for link in link_list:
            src = link.src
            dst = link.dst
            self.link_port_table[(self.apid_dict.get(src.dpid), self.apid_dict.get(dst.dpid))] = (src.port_no, dst.port_no)

            if self.apid_dict.get(src.dpid) in self.all_switches_dpid:
                self.switch_port_table[self.apid_dict.get(src.dpid)].add(src.port_no)
            if self.apid_dict.get(dst.dpid) in self.all_switches_dpid:
                self.switch_port_table[self.apid_dict.get(dst.dpid)].add(dst.port_no)

        # 统计没用的端口
        for sw_dpid in self.switch_all_ports_table.keys():
            all_ports = self.switch_all_ports_table[sw_dpid]
            linked_port = self.switch_port_table[sw_dpid]
            # print("discovery_topo---> all_ports, linked_port", all_ports, linked_port)
            self.not_use_ports[sw_dpid] = all_ports - linked_port

        # 建立拓扑 bw和delay未设定
        self.build_topology_between_switches()
        # self.cal_path_flag = True

    def build_topology_between_switches(self, free_bw=0, delay=0, loss=0, used_bw=0, pkt_err=0, pkt_drop=0, distance=0):
        """
        根据src_dpid和dst_dpid建立拓扑,bw和delay信息未设定
        """
        # networkxs使用已有Links的src_dpid和dst_dpid信息建立拓扑
        _graph = nx.Graph()

        # slef.graph.clear()
        for (src_dpid, dst_dpid) in self.link_port_table.keys():
            # 建立switch之间的连接，端口可以通过查link_port_table获得
            _graph.add_edge(src_dpid, dst_dpid, free_bw=free_bw, delay=delay, loss=loss, used_bw=used_bw, pkt_err=pkt_err, pkt_drop=pkt_drop, distance=distance)
        if _graph.edges == self.graph.edges:
            return
        else:
            self.graph = _graph

    def calculate_weight(self, node1, node2, weight_dict):
        """
        计算路径时,weight可以调用函数,该函数根据因子计算 bw*factor - delay*(1-factor) 后的weight
        """
        # weight可以调用的函数
        assert 'bw' in weight_dict and 'delay' in weight_dict, "edge weight should have bw and delay"
        try:
            weight = weight_dict['bw'] * setting.FACTOR - weight_dict['delay'] * (1 - setting.FACTOR)
            return weight
        except TypeError:
            print("discovery_topo ERROR---> weight_dict['bw']: ", weight_dict['bw'])
            print("discovery_topo ERROR---> weight_dict['delay']: ", weight_dict['delay'])
            return None

    def calculate_shortest_paths(self, src_dpid, dst_dpid, weight=None):
        """
        计算src到dst的最短路径,存在self.shortest_path_table中
        """
        graph = self.graph.copy()
        # print(graph.edges)
        # print("SSSS--->get_shortest_paths ==calculate shortest path %s to %s" % (src_dpid, dst_dpid))
        self.shortest_path_table[(src_dpid, dst_dpid)] = nx.shortest_path(self.graph,
                                                                          source=src_dpid,
                                                                          target=dst_dpid,
                                                                          weight=weight,
                                                                          method=setting.METHOD)
        # print("SSSS--->get_shortest_paths ==[PATH] %s <---> %s: %s" % (
        #     src_dpid, dst_dpid, self.shortest_path_table[(src_dpid, dst_dpid)]))

    def calculate_all_nodes_shortest_paths(self, weight=None):
        """
        根据已构建的图,调用get_shortest.paths()函数计算所有nodes间的最短路径, weight为权值, 可以用calculate_weight()函数
        """
        self.shortest_path_table = {}  # 先清空，再计算
        for src in self.graph.nodes():
            for dst in self.graph.nodes:
                if src != dst:
                    self.calculate_shortest_paths(src, dst, weight=weight)
                else:
                    continue

    def get_host_ip_location(self, host_ip):
        """
        通过host_ip查询self.access_table: {(dpid, in_port): (src_ip, src_mac)}
        获得(dpid, in_port)连接的交换机信息
        """
        # if host_ip == "0.0.0.0" or host_ip == "255.255.255.255":
        #     return None

        for key in self.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if self.access_table[key][0] == host_ip:
                # print("discovery_topo--->zzzz---> key", key)
                return key
            # print("SSSS--->get_host_ip_location: %s location is not found" % host_ip)
        return None

    # def get_ip_by_dpid(self, dpid):
    #     """
    #     通过dpid查询{(dpid, in_port): (src_ip, src_mac)}
    #     获得 ip  src_ip主机ip
    #     """
    #     for key, value in self.access_table.items():
    #         if key[0] == dpid:
    #             return value[0]
    #     print("SSSS--->get_ip_by_dpid: %s ip is not found" % dpid)
    #     return None

    def parse_topo_links_info(self):
        m_graph = nx.Graph()
        parser = ET.parse(self.link_info_xml)
        root = parser.getroot()

        links_info_element = root.find("links_info")

        def _str_tuple2int_list(s: str):
            s = s.strip()
            assert s.startswith('(') and s.endswith(')'), '应该为str的元组, 如“(1, 2)”'
            s_ = s[1: -1].split(', ')
            return [int(i) for i in s_]

        node1, node2, port1, port2, free_bw, delay, loss, used_bw, pkt_err, pkt_drop, distance= None, None, None, None, None, None, None, None, None, None, None
        for e in root.iter():
            if e.tag == 'links':
                node1, node2 = _str_tuple2int_list(e.text)
            elif e.tag == 'ports':
                port1, port2 = _str_tuple2int_list(e.text)
            elif e.tag == 'free_bw':
                free_bw = float(e.text)
            elif e.tag == 'delay':
                delay = float(e.text[:-2])
            elif e.tag == 'loss':
                loss = float(e.text)
            elif e.tag == 'used_bw':
                used_bw = float(e.text)
            elif e.tag == 'pkt_err':
                pkt_err = float(e.text)
            elif e.tag == 'pkt_drop':
                pkt_drop = float(e.text)
            elif e.tag == 'distance':
                distance = float(e.text)
            else:
                print(e.tag)
                continue
            # self.ap_distance[(node1, node2)] = distance  #正向
            # self.ap_distance[(node2, node1)] = distance  #反向
            # self.graph[node1][node2]['distance'] = distance  # 添加距离
            m_graph.add_edge(node1, node2, port1=port1, port2=port2, free_bw=free_bw, delay=delay, loss=loss, used_bw=used_bw, pkt_err=pkt_err, pkt_drop=pkt_drop, distance=distance)

        for edge in m_graph.edges(data=True):
            print("networkstructure: ", edge)
        return m_graph
