import random
import xml.etree.ElementTree as ET
import networkx
import sys
import json
import os
import re
import time
import threading

from pathlib import Path
from mn_wifi.topo import Topo
from mn_wifi.net import Mininet_wifi
from mininet.node import RemoteController
from mn_wifi.link import wmediumd
from mn_wifi.cli import CLI
from mininet.log import setLogLevel, info
from mn_wifi.wmediumdConnector import interference
from mininet.util import dumpNodeConnections
from json import dumps
from requests import put
from mininet.util import quietRun

random.seed(2021)


def generate_port(node_idx1, node_idx2):
    """
    生成端口
    """
    if (node_idx2 > 9) and (node_idx1 > 9):
        port = str(node_idx1) + "0" + str(node_idx2)
    else:
        port = str(node_idx1) + "00" + str(node_idx2)

    return int(port)


def generate_ap_port(graph):
    """
    生成交换机端口
    """
    switch_port_dict = {}
    for node in graph.nodes:
        switch_port_dict.setdefault(node, list(range(graph.degree[node])))
    return switch_port_dict


def parse_xml_topology(topology_path):
    """
    从topology.xml中解析出topology
    return: topology graph, networkx.Graph()  拓扑图
            nodes_num,  int   节点数
            edges_num, int    链路数
    """
    tree = ET.parse(topology_path)
    root = tree.getroot()
    topo_element = root.find("topology")
    graph = networkx.Graph()
    ap_location = []
    sta_location = []
    for child in topo_element.iter():
        # 解析节点
        if child.tag == 'node':
            node_id = int(child.get('id'))
            graph.add_node(node_id)
            #解析AP和STA的坐标
            locationap = str(child.find('locationap').get('coordinate'))
            locationsta = str(child.find('locationsta').get('coordinate'))

            ap_location.append(locationap) #添加进列表
            sta_location.append(locationsta)

        # 解析链路
        elif child.tag == 'link':
            from_node = int(child.find('from').get('node'))
            to_node = int(child.find('to').get('node'))
            graph.add_edge(from_node, to_node)

    nodes_num = len(graph.nodes)
    edges_num = len(graph.edges)

    print('nodes: ', nodes_num, '\n', graph.nodes, '\n',
          'edges: ', edges_num, '\n', graph.edges)
    return graph, nodes_num, edges_num, ap_location, sta_location


def create_topo_links_info_xml(path, links_info):
    """
        <links_info>
            <links> (switch1, switch2)
                <ports>(1, 1)</ports>
                <bw>100</bw>
                <delay>5ms</delay>
                <loss>1</loss>
                <distance>50</distance>
            </links>
        </links_info>
    :param path: 保存路径
    :param links_info: 链路信息字典 {link: {ports, bw, delay, loss, distance}}
    :return: None
    """
    # 根节点
    root = ET.Element('links_info')

    for link, info in links_info.items():
        # 一级子节点 links
        child = ET.SubElement(root, 'links')
        child.text = str(link)

        # 二级子节点 （ports, bw, delay, loss）
        sub_child1 = ET.SubElement(child, 'ports')
        sub_child1.text = str((info['port1'], info['port2']))

        sub_child2 = ET.SubElement(child, 'bw')
        sub_child2.text = str(info['bw'])

        sub_child2 = ET.SubElement(child, 'delay')
        sub_child2.text = str(info['delay'])

        sub_child2 = ET.SubElement(child, 'loss')
        sub_child2.text = str(info['loss'])

        sub_child2 = ET.SubElement(child, 'distance')
        sub_child2.text = str(info['distance'])

    tree = ET.ElementTree(root)
    indent(root)  #调整xml格式
    Path(path).parent.mkdir(exist_ok=True)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print('saved links info xml.')


def indent(elem, level=0): 
    """
    xml调整格式函数
    """
    i = "\n" + level*" " 
    if len(elem): 
     if not elem.text or not elem.text.strip(): 
      elem.text = i + " " 
     if not elem.tail or not elem.tail.strip(): 
      elem.tail = i 
     for elem in elem: 
      indent(elem, level+1) 
     if not elem.tail or not elem.tail.strip(): 
      elem.tail = i 
    else: 
     if level and (not elem.tail or not elem.tail.strip()): 
      elem.tail = i 


def get_mininet_wifi_device(net, idx: list, device='sta'):
    """
        获得idx中mininet-wifi的实例, 如 sta1, sta2 ...;  ap1, ap2 ...
    :param net: mininet-wifi网络实例
    :param idx: 设备标号集合, list
    :param device: 设备名称 'sta', 'ap'
    :return d: dict{idx: 设备mininet-wifi实例}

    """
    d = {}
    for i in idx:
        d.setdefault(i, net.get(f'{device}{i}'))
    return d


def run_corresponding_sh_script(device: dict, label_path):
    """
        对应的device运行对应的shell脚本
    :param devices: { idx: device}
    :param label_path: './24nodes/TM-{}/{}/{}_'
    """
    p = label_path + '{}.sh'
    for i, d in device.items():
        if i < 9:
            i = f'0{i}'
        else:
            i = f'{i}'
        p = p.format(i)
        _cmd = f'bash{p}'
        d.cmd(_cmd)  # 发送mininet-wifi指令
    print(f'--->complete run {label_path}')


def run_ip_add_default(hosts: dict):
    """
    运行 ip route add defaule via 192.168.0.0.x 命令
    """
    _cmd = 'ip route add default via 192.168.0.'
    for i, h in hosts.items():
        print(_cmd + str(i))
        h.cmd(_cmd + str(i))
    print("---> run ip add default complete")


def _test_cmd(devices: dict, my_cmd):
    for i, d in devices.items():
        d.cmd(my_cmd)
        print(f'exec {my_cmd}zzz{i}')
        # print(f'return{r}')


def run_iperf(path, host):
    _cmd = 'bash' + path + '&'
    host.cmd(_cmd)


def all_host_run_iperf(stations: dict, path, finish_file):
    """
        path = r'./iperfTM/'
    """
    idxs = len(os.listdir(path))
    path = path + '/TM-'
    for idx in range(idxs):
        script_path = str(path) + str(idx)
        for i, h in stations.items():
            servers_path = script_path + '/Servers/server_{}.txt'.format(str(i))
            with open(servers_path, "r", encoding="utf-8") as ServerFile:
                server_cmds = ServerFile.readlines()
                for server_cmd in server_cmds:
                    time.sleep(0.1)
                    print(server_cmd.strip())
                    h.cmd(server_cmd.strip())

        for i, h in stations.items():
            clients_path = script_path + "/Clients/client_{}.txt".format(str(i))
            with open(clients_path, "r", encoding="utf-8") as ClientFile:
                client_cmds = ClientFile.readlines()
                for client_cmd in client_cmds:
                    time.sleep(0.7)
                    print(client_cmd.strip() + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    h.cmd(client_cmd.strip())

        time.sleep(10)  # 防止发包太快

    print("===========done============")
    time.sleep(10)
    write_iperf_time(finish_file)


def write_pingall_time(finish_file):
    with open(finish_file, 'w+') as f:
        _content = {
            "ping_all_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "start_save_flag": True,
            "finish_flag": True
        }
        json.dump(_content, f)

def write_iperf_time(finish_file):
    with open(finish_file, "r+") as f:
        _read = json.load(f)
        _content = {
            "iperf_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "finish_flag": True
        }
        _read.update(_content)

    with open(finish_file, "w+") as f:
        json.dump(_read, f)


def remove_finish_file(finish_file):
    try:
        os.remove(finish_file)
    except FileNotFoundError:
        pass


def net_sta1_ping_others(net):
    stations = net.stations
    for sta in stations[1:]:
        net.ping((stations[0], sta))


def cal_ap_distance(ap1_location, ap2_location):
    ap1_x, ap1_y, ap1_z = ap1_location.split(',')
    ap2_x, ap2_y, ap2_z = ap2_location.split(',')

    x = abs(int(ap1_x) - int(ap2_x))
    y = abs(int(ap1_y) - int(ap2_y))

    distance = pow(x**2+y**2, 0.5)

    return distance

class Nodes14Topo():
    def __init__(self, graph, ap_location, sta_location):
        super(Nodes14Topo, self).__init__()
        "Create a network."
        self.net = Mininet_wifi(controller=RemoteController)

        self.graph = graph
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges

        self.random_bw = 40  # Gbps -> M * 10
        self.bw4 = 50  # host -- switch

        self.delay = 1  # ms
        self.loss = 1  # %

        self.host_port = 9
        self.snooper_port = 10
        self.ap_position = ap_location  #['100,100,0', '50,50,0', '150,50,0']
        self.sta_position = sta_location  #['100,110,0', '40,50,0', '160,50,0']

    def topology(self, args):

        info("*** Creating nodes\n")
        #添加AP
        APs = {}
        for ap in self.node_idx:
            APs.setdefault(ap, self.net.addAccessPoint(f'ap{ap}', 
                           ssid=f"ap{ap}-ssid", mode="g", channel="1",
                           position=self.ap_position[ap-1],
                           mac="{}0:00:00:00:00:00".format(ap)))
            print('添加AP:', ap)

        # print(APs[1])

        #添加sta
        STAs = {}
        for sta in self.node_idx:
            STAs.setdefault(sta, self.net.addStation(f'sta{sta}', mac=f'00:00:00:00:00:0{sta+1}', 
                                 ip=f'192.168.0.{sta}/24', position=self.sta_position[sta-1]))
            print(self.sta_position[sta-1])
            print('添加STA:', sta) 

        #添加控制器  
        c0 = self.net.addController('c0', controller=RemoteController)

        info("*** Configuring Propagation Model\n")
        self.net.setPropagationModel(model="logDistance", exp=4.5)

        info("*** Configuring nodes\n")
        self.net.configureNodes()

        info("*** Adding Links\n")
        ap_port_dict = generate_ap_port(self.graph)
        links_info = {}
        # 添加链路
        for l in self.edges_pairs:
            port1 = ap_port_dict[l[0]].pop(0) + 1
            port2 = ap_port_dict[l[1]].pop(0) + 1

            _bw = random.randint(5, self.random_bw)
            _d = str(random.randint(1, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)
            _distance = round(cal_ap_distance(self.ap_position[l[0]-1], self.ap_position[l[1]-1]), 2)

            self.net.addLink(APs[l[0]], APs[l[1]], cls=wmediumd,
                         bw=_bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": _bw, "delay": _d, "loss": _l, "used_bw": 0, "pkt_err": 0, "pkt_drop": 0, "distance": _distance})

        create_topo_links_info_xml(links_info_xml_path, links_info)
        
        #AP<--->STA
        for i in self.node_idx:
            self.net.addLink(APs[i], STAs[i], bw=self.bw4)
        
        #画图
        # if '-p' not in args:
        #     self.net.plotGraph(max_x=300, max_y=300)

        info("*** Starting network\n")
        self.net.build()
        c0.start()
        
        for i in self.node_idx:
            APs[i].start([c0])
        
        print("=====================Wait ryu init ===========================")
        time.sleep(20)  # 做一个延时等待ryu的连接


        # # -------------------------流量检测----------------------------#
        # for i in self.node_idx:
        #     APs[i].cmd('iw dev %s-mp2 interface add %s-mon0 type monitor' %
        #                (APs[i].name, APs[i].name))
        #     APs[i].cmd('ifconfig %s-mon0 up' % APs[i].name)

        # ifname = 'enp2s0'  # have to be changed to your own iface!
        # collector = os.environ.get('COLLECTOR', '127.0.0.1')
        # sampling = os.environ.get('SAMPLING', '10')
        # polling = os.environ.get('POLLING', '10')
        # sflow = 'ovs-vsctl -- --id=@sflow create sflow agent=%s target=%s ' \
        #         'sampling=%s polling=%s --' % (ifname, collector, sampling, polling)

        # for ap in self.net.aps:
        #     sflow += ' -- set bridge %s sflow=@sflow' % ap
        #     info(' '.join([ap.name for ap in self.net.aps]))
        #     quietRun(sflow)

        # agent = '127.0.0.1'
        # topo = {'nodes': {}, 'links': {}}
        # for ap in self.net.aps:
        #     topo['nodes'][ap.name] = {'agent': agent, 'ports': {}}

        # path = '/sys/devices/virtual/mac80211_hwsim/'
        # for child in os.listdir(path):
        #     dir_ = '/sys/devices/virtual/mac80211_hwsim/' + '%s' % child + '/net/'
        #     for child_ in os.listdir(dir_):
        #         node = child_[:3]
        #         if node in topo['nodes']:
        #             ifindex = open(dir_ + child_ + '/ifindex').read().split('\n', 1)[0]
        #             topo['nodes'][node]['ports'][child_] = {'ifindex': ifindex}

        # path = '/sys/devices/virtual/net/'
        # for child in os.listdir(path):
        #     parts = re.match('(^.+)-(.+)', child)
        #     if parts is None: continue
        #     if parts.group(1) in topo['nodes']:
        #         ifindex = open(path + child + '/ifindex').read().split('\n', 1)[0]
        #         topo['nodes'][parts.group(1)]['ports'][child] = {'ifindex': ifindex}
        # for l in self.edges_pairs:
        #     linkName = '%s-%s' % (APs[l[0]].name, APs[l[1]].name)
        #     topo['links'][linkName] = {'node1': APs[l[0]].name, 'port1': 'APs[{}]-mp2'.format(l[0]),
        #                                'node2': APs[l[1]].name, 'port2': 'APs[{}]-mp2'.format(l[1])}

        #     topo['links'][linkName] = {'node1': APs[l[0]].name, 'port1': APs[l[0]].wintfs[0].name,
        #                                'node2': APs[l[1]].name, 'port2': APs[l[1]].wintfs[0].name}

        #     put('http://127.0.0.1:8008/topology/json', data=dumps(topo))  # 流量发送处理


        # # ------------------------------------发流处理------------------------------#
        # info("***send flow\n")
        # print("---------------get stations device list-----------------------")
        # stations = get_mininet_wifi_device(self.net, self.graph.nodes, device='sta')
        # print(stations)
        # print("================Dumping host connections======================")
        # dumpNodeConnections(self.net.stations)
        

        # # ----------------------------------------添加网关--------------------------------#
        # run_ip_add_default(stations)
        # # pingall 测试整个网络的通信情况
        # self.net.pingAll()
        # # 做一个单机测试sta1
        # # time.sleep(10)
        # # net_sta1_ping_others(self.net)
        # # time.sleep(10)
        # write_pingall_time(finish_file)

        # # ---------------------------------------iperf测试--------------------------------#
        # # stations[1].cmd('iperf -s -u -p 1002 -1 &')
        # # stations[2].cmd('iperf -c 192.168.0.1 - u -p 1002 -b 20000k -t 30  &')
        # # ---------------------------------------iperf正式发流-----------------------------#
        # print("====RUn iperf scripts")
        # # time.sleep(20)  # 做一个延时等待ryu的连接
        # t = threading.Thread(target=all_host_run_iperf, args=(stations, iperf_path, finish_file))  # 建立一个子线程进行发流
        # print("*****************Thread iperf start*************************")
        # t.start()  # 启动线程

        info("*** Running CLI\n")
        CLI(self.net)

        info("*** Stopping network\n")
        self.net.stop()


if __name__ == "__main__":
    xml_topology_path = r'topologies/topology_1.xml'
    links_info_xml_path = r'save_links_info/links_info.xml'
    iperf_path = "./iperfTM"
    iperf_interval = 0
    finish_file = './finish_time.json'

    graph, nodes_num, edges_num, ap_location, sta_location = parse_xml_topology(xml_topology_path)
    mytopo = Nodes14Topo(graph, ap_location, sta_location)

    setLogLevel('info')
    mytopo.topology(sys.argv)
