# @Project : Experiment-code 
# @File    : dataset.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/17 17:35 
# @Describe:
# @Update  :
import os
import networkx as nx
import xml.etree.ElementTree as ET  # 解析xml树形结构
from pathlib import Path
from parser_config import parse_config


def parse_topo_links_info(links_info_path):
    """
    解析topo生成的xml，为设置env的state和action
    :return:
    """
    m_graph = nx.Graph()
    parser = ET.parse(Path(links_info_path))
    root = parser.getroot()

    def _str_tuple2int_list(s: str):
        s = s.strip()
        assert s.startswith('(') and s.endswith(')'), '应该为str的元组, 如“(1, 2)”'
        s_ = s[1: -1].split(', ')
        return [int(i) for i in s_]

    node1, node2, port1, port2, free_bw, delay, loss, used_bw, pkt_err, pkt_drop, distance = None, None, None, None, None, None, None, None, None, None, None
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
            # print(e.tag)
            continue

        m_graph.add_edge(node1, node2, port1=port1, port2=port2, free_bw=free_bw, delay=delay, loss=loss,
                         used_bw=used_bw, pkt_err=pkt_err, pkt_drop=pkt_drop, distance=distance)

    # for edge in m_graph.edges(data=True):
    #     print("networkstructure: ", edge)

    return m_graph


def get_node_neighbors(links_info_path):
    """
    找出所有节点相对应的邻居节点
    :return:
    """
    m_graph = parse_topo_links_info(links_info_path)
    nodes = sorted(m_graph.nodes())
    neighbors = {}  # 所有节点的邻居节点
    for node in nodes:
        neighbors[node] = list(graph.neighbors(node))
    return neighbors


def file_path_yield(file_dir, s=10, n=1000, step=1):
    """
    按序号读取保存数据的文件
    :param file_dir: 数据目录
    :param s: 开始文件index
    :param n: 结束文件index
    :param step: 读取步长
    :return:
    """
    _dir = os.listdir(Path(file_dir))
    assert n <= len(_dir), "n should small than len(a)"  # 判断是否超出数据文件总数
    file_names = sorted(_dir, key=lambda x: int(x.split('-')[0]))  # 按文件序号排序
    for name in file_names[s:n:step]:
        yield os.path.join(file_dir, name)


def read_pickle(pickle_path):
    """
    读取pickle并转化为graph
    :param pickle_path:
    :return:
    """
    pkl_graph = nx.read_gpickle(pickle_path)
    # print(pkl_graph.edges.data())
    # nx.draw(pkl_graph, with_labels=True)
    # plt.show()
    return pkl_graph


if __name__ == '__main__':
    config = parse_config(Path('config/train_config.yml'))
    graph = parse_topo_links_info(config['links_info_path'])
    neighbors = get_node_neighbors(config['links_info_path'])
    for index, pkl_path in enumerate(file_path_yield(config['files_dir_path'], s=10, n=11, step=2)):
        read_pickle(pkl_path)
