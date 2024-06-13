# -*- coding: utf-8 -*-
"""
@File     : detele_weight_file.py
@Date     : 2022-11-11
@Author   : Terry_Li  --我很喜欢枫叶，可惜枫叶红时，总多离别。
IDE       : VS Code
@Mail     : terry.ljq.dev@foxmail.com
"""
import os
import time
import json
import networkx as nx
import xml.etree.ElementTree as ET
from pathlib import Path


def generate_port(node_idx1, node_idx2):
    """
    创建端口号
    """
    if (node_idx2 > 9) and (node_idx1 > 9):
        port = str(node_idx1) + "0" + str(node_idx2)

    else:
        port = str(node_idx1) + "00" + str(node_idx2)  # test

    return int(port)


def generate_ap_port(graph):
    """创建ap的端口号"""
    ap_port_dict = {}
    for node in graph.nodes:
        ap_port_dict.setdefault(node, list(range(graph.degree[node])))
    return ap_port_dict


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
    graph = nx.Graph()
    ap_location = []
    sta_location = []
    for child in topo_element.iter():
        # 解析节点
        if child.tag == 'node':
            node_id = int(child.get('id'))
            graph.add_node(node_id)
            # 解析AP和STA的坐标
            locationap = str(child.find('locationap').get('coordinate'))
            locationsta = str(child.find('locationsta').get('coordinate'))

            ap_location.append(locationap)  # 添加进列表
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
                <free_bw>100</free_bw>
                <delay>5ms</delay>
                <loss>1</loss>
                <used_bw>0</used_bw>
                <pkt_err>0</pkt_err>
                <pkt_drop>0</pkt_drop>
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

        sub_child2 = ET.SubElement(child, 'free_bw')
        sub_child2.text = str(info['free_bw'])

        sub_child2 = ET.SubElement(child, 'delay')
        sub_child2.text = str(info['delay'])

        sub_child2 = ET.SubElement(child, 'loss')
        sub_child2.text = str(info['loss'])

        sub_child2 = ET.SubElement(child, 'used_bw')
        sub_child2.text = str(info['used_bw'])

        sub_child2 = ET.SubElement(child, 'pkt_err')
        sub_child2.text = str(info['pkt_err'])

        sub_child2 = ET.SubElement(child, 'pkt_drop')
        sub_child2.text = str(info['pkt_drop'])

        sub_child2 = ET.SubElement(child, 'distance')
        sub_child2.text = str(info['distance'])

    tree = ET.ElementTree(root)
    indent(root)  # 调整xml格式
    Path(path).parent.mkdir(exist_ok=True)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print('saved links info xml.')


def indent(elem, level=0):
    """解决xml文件缩进的问题"""
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
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
    ：param devices: { idx: device}
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

    distance = pow(x ** 2 + y ** 2, 0.5)

    return distance
