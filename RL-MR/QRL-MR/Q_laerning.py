#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2023/11/26 21:48
@File:Q_laerning.py
@Desc:这种方法完全不考虑其他智能体的影响,这种情况下，每个智能体仅仅是根据自己的需要达到目标，可能导致结果仅仅是局部最优解。
"""
import copy
import numpy as np
import os
import pickle
from my_main import Data_process,generating_tree,graph_display,tree_evaluate
import networkx as nx
import matplotlib.pyplot as plt

class MulticastEnv:
    def __init__(self,graph,src,dst,beta):
        """
        环境初始化
        :param graph:当前的地图
        :param src: 源节点
        :param dst: 目的节点
        """
        # 注意这里的graph中有剩余带宽，时延，丢包率等消息，归一化后的结果
        self.graph = graph
        self.src = src
        self.dst = dst   # 目的节点集列表
        self.state = copy.deepcopy(src)  # 当前状态
        self.beta = beta

    def step(self,action):  # 外部调用这个函数来改变当前位置
        # 动作表示下一个状态
        next_state = action + 1
        done = False
        try:
            bw = self.graph[self.state][next_state]['bw']
            delay = self.graph[self.state][next_state]['delay']
            loss = self.graph[self.state][next_state]['loss']
        except:
            cost = float('inf')
            next_state = self.state
            return next_state,cost,done
        cost = self.beta[0]*(1-bw) + self.beta[1]*delay + self.beta[2] * loss
        if next_state == self.dst:
            cost = -10
            done = True
        self.state = next_state
        return next_state,cost,done

    def reset(self):  # 回归初始状态，源节点
        self.state = copy.deepcopy(self.src)
        return self.state


# 绘制结果图
def env_display(graph,src,stds,weight='bw'):
    pos = nx.spring_layout(graph)
    edge_weights = nx.get_edge_attributes(graph,weight)
    nx.draw(graph,pos,
            with_labels=True,
            node_size=700,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            font_weight='bold'
            )
    # 显示权重
    nx.draw_networkx_edge_labels(graph,pos,edge_labels={(u,v):f'{weight:.2f}' for (u,v),weight in edge_weights.items()})
    # 高亮显示源点(红色)
    nx.draw_networkx_nodes(graph,pos,nodelist=src,node_color='red',node_size=700)
    nx.draw_networkx_nodes(graph,pos,nodelist=stds,node_color='yellow',node_size=700)
    plt.show()


class QLearning:
    """ Q-laerning算法 """
    def __init__(self,graph,epsilon,alpha,gamma):
        self.graph = graph
        num_node = len(self.graph.nodes())
        self.Q_table = np.zeros([num_node,num_node])   # 初始化Q(s,a)表格
        self.n_action = num_node   # 动作个数，以节点为动作
        self.epsilon = epsilon   # epsilon-贪婪策略中的参数
        self.alpha = alpha   # 学习率
        self.gamma = gamma    # 折扣因子
        self.best_path_dict = {}   # 最好的路径字典
        self.min_cost_dict = {}  # 最小的消耗字典

    def take_action(self,state,exploration=True):  # 选取下一步的操作
        neighbors_of_state = list(self.graph.neighbors(state))
        neighbors_of_index = [i-1 for i in neighbors_of_state]
        if np.random.random() < self.epsilon and exploration:
            action = np.random.choice(neighbors_of_index)
        else:
            action = neighbors_of_index[np.argmin(self.Q_table[state-1][neighbors_of_index])]
        if type(action) is not int and type(action) is not np.int32:
            print(action)
        return action


    def best_action(self,state):   # 用于打印策略
        Q_min = np.min(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state,i] == Q_min:
                a[i] = 1
        return a

    def update(self,s0,a0,r,s1):
        td_error = r+self.gamma*self.Q_table[s1-1].min()-self.Q_table[s0-1,a0]
        self.Q_table[s0-1,a0] += self.alpha * td_error


    def best_path(self,env,src,dst):   # 保存相应的路径
        done = False
        state = env.reset()
        self.best_path_dict[(src,dst)] = [src]
        self.min_cost_dict[(src,dst)] = 0
        step = 0
        while not done:
            action = self.take_action(state,exploration=False)
            next_state,cost,done = env.step(action)
            self.min_cost_dict[(src,dst)] += cost
            self.best_path_dict[(src,dst)].append(next_state)
            state = next_state
            step += 1
            if step == 100:
                raise RuntimeError("算法没有收敛")
        return self.best_path_dict[(src,dst)]

# 绘制训练结果
def train_plot(x,y):
    plt.plot(x,y)
    plt.xlabel('episode')
    plt.ylabel('cost')
    plt.show()


# 用于外界调用生成结果
def run(file_path,src,dsts):
    # 内部参数设置
    epsilon = 0.1
    alpha = 0.01
    gamma = 0.95
    beta = [1,1,1]
    negative_attr = [1,0,0]
    num_episodes = 1000

    # graph = read_pkl(file_path)  # 读取topo数据
    with open(file_path, 'rb') as pkl_file:
        graph = pickle.load(pkl_file)
    data_process = Data_process(graph)
    new_graph = data_process.normalization()
    # env_display(new_graph,src,dsts)
    paths = {}
    for src,dst in zip([src[0] for _ in dsts],dsts):
        # print(src,dst)
        return_list = []
        env = MulticastEnv(new_graph, src, dst, beta)
        agent = QLearning(new_graph, epsilon, alpha, gamma)
        for episode in range(num_episodes):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state,reward,done = env.step(action)
                # print(action+1)
                episode_return += reward  # 这里的回报的计算不进行折扣因子衰减
                agent.update(state,action,reward,next_state)
                state = next_state
            return_list.append(episode_return)
            # print(episode,episode_return)
        paths[(src,dst)] = agent.best_path(env,src,dst)
        # train_plot(list(range(num_episodes)),return_list)
    tree = generating_tree(paths)
    # graph_display(new_graph,tree,[src],dsts)
    result = tree_evaluate(graph, tree, paths)  # 评价树的相应的指标,返回（瓶颈路径的平均值，链路平均时延，链路平均丢包率，链路长度）
    # print('瓶颈路径的平均值，链路平均时延，链路平均丢包率，链路长度\n', result)
    return result


if __name__ == '__main__':
    directory_path = '../dataset/topo and data/node21_data'
    src = [3]
    dsts = [16,12,21]
    epsilon = 0.1
    alpha = 0.01
    gamma = 0.95
    beta = [1, 1, 1]
    negative_attr = [1, 0, 0]
    file_list = os.listdir(directory_path)
    illegal_document = {'plot_info.pkl', '0-2023-11-20-15-41-12.pkl',
                        '0-2023-11-20-15-41-22.pkl', '0-2023-11-20-15-41-32.pkl'}
    file_set = set(file_list)
    sample_size = 120
    # sample_file = random.sample(file_list-illegal_document,)
    test_file_index = 60
    num_episodes = 1000  # 智能体在环境中运行的序列数量

    # 循环处理文件
    for index, file_name in enumerate(file_list):
        if index == test_file_index:  # 用于选择文件,后面再进行优化
            # 组合完整文件路径
            file_path = os.path.join(directory_path, file_name)
            # graph = read_pkl(file_path)  # 读取topo数据
            with open(file_path, 'rb') as pkl_file:
                graph = pickle.load(pkl_file)
            data_process = Data_process(graph)
            new_graph = data_process.normalization()
            env_display(new_graph,src,dsts)
            paths = {}
            for src,dst in zip([src[0] for _ in dsts],dsts):
                print(src,dst)
                return_list = []
                env = MulticastEnv(new_graph, src, dst, beta)
                agent = QLearning(new_graph, epsilon, alpha, gamma)
                for episode in range(num_episodes):
                    episode_return = 0
                    state = env.reset()
                    done = False
                    while not done:
                        action = agent.take_action(state)
                        next_state,reward,done = env.step(action)
                        # print(action+1)
                        episode_return += reward  # 这里的回报的计算不进行折扣因子衰减
                        agent.update(state,action,reward,next_state)
                        state = next_state
                    return_list.append(episode_return)
                    # print(episode,episode_return)
                paths[(src,dst)] = agent.best_path(env,src,dst)
                train_plot(list(range(num_episodes)),return_list)
            tree = generating_tree(paths)
            graph_display(new_graph,tree,[src],dsts)
            result = tree_evaluate(graph, tree, paths)  # 评价树的相应的指标,返回（瓶颈路径的平均值，链路平均时延，链路平均丢包率，链路长度）
            print('瓶颈路径的平均值，链路平均时延，链路平均丢包率，链路长度\n', result)












