# @Project : Experiment-code 
# @File    : model.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/4 15:36 
# @Describe:
# @Update  :
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from parser_config import parse_config
from pathlib import Path


class Actor(nn.Module):
    def __init__(self, config_path, domain_node):
        super(Actor, self).__init__()
        self.config = parse_config(config_path)
        self.domain_node = domain_node
        self.conv1 = nn.Conv2d(in_channels=self.config['conv1']['in_channels'],
                               out_channels=self.config['conv1']['out_channels'],
                               kernel_size=self.config['conv1']['kernel_size'])
        self.fc1 = nn.Linear(self.config['fc1']['in_channels'] * self.domain_node * self.domain_node,
                             self.config['fc1']['out_channels'])
        # self.fc1 = nn.Linear(self.config['fc1']['in_channels'] * self.domain_node,
        #                      self.config['fc1']['out_channels'])
        self.fc2 = nn.Linear(self.config['fc2']['in_channels'],
                             self.config['fc2']['out_channels'])
        self.action_head = nn.Linear(self.config['fc3']['in_channels'],
                                     self.config['fc3']['out_channels'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.config['fc1']['in_channels'] * self.domain_node * self.domain_node)
        # x = x.view(-1, self.config['fc1']['in_channels'] * self.domain_node)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, config_path, domain_node):
        super(Critic, self).__init__()
        self.config = parse_config(config_path)
        self.domain_node = domain_node
        self.conv1 = nn.Conv2d(in_channels=self.config['conv1']['in_channels'],
                               out_channels=self.config['conv1']['out_channels'],
                               kernel_size=self.config['conv1']['kernel_size'])
        self.fc1 = nn.Linear(self.config['fc1']['in_channels'] * self.domain_node * self.domain_node,
                             self.config['fc1']['out_channels'])
        # self.fc1 = nn.Linear(self.config['fc1']['in_channels'] * self.domain_node,
        #                      self.config['fc1']['out_channels'])
        self.fc2 = nn.Linear(self.config['fc2']['in_channels'],
                             self.config['fc2']['out_channels'])
        self.state_value = nn.Linear(self.config['fc3']['in_channels'], 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.config['fc1']['in_channels'] * self.domain_node * self.domain_node)
        # x = x.view(-1, self.config['fc1']['in_channels'] * self.domain_node)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class Actor2(nn.Module):
    def __init__(self, config_path, domain_node):
        super(Actor2, self).__init__()
        self.config = parse_config(config_path)
        self.domain_node = domain_node
        self.conv1_1 = nn.Conv2d(5, 32, kernel_size=(5, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(5, 1))

        self.conv2_1 = nn.Conv2d(5, 32, kernel_size=(1, 5))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 5))

        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 256)

        self.action_head = nn.Linear(256, 10)

    def forward(self, x):
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))

        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))

        # x = x1_2.view(x.shape[0], -1)
        x1_3 = x1_2.view(x.shape[0], -1)
        x2_3 = x2_2.view(x.shape[0], -1)
        x = torch.cat([x1_3, x2_3], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic2(nn.Module):
    def __init__(self, config_path, domain_node):
        super(Critic2, self).__init__()
        self.config = parse_config(config_path)
        self.domain_node = domain_node
        self.conv1_1 = nn.Conv2d(5, 32, kernel_size=(5, 1))
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(5, 1))

        self.conv2_1 = nn.Conv2d(5, 32, kernel_size=(1, 5))
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(1, 5))

        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 256)

        self.state_value = nn.Linear(256, 1)

    def forward(self, x):
        x1_1 = F.leaky_relu(self.conv1_1(x))
        x1_2 = F.leaky_relu(self.conv1_2(x1_1))

        x2_1 = F.leaky_relu(self.conv2_1(x))
        x2_2 = F.leaky_relu(self.conv2_2(x2_1))

        # x = x1_2.view(x.shape[0], -1)
        x1_3 = x1_2.view(x.shape[0], -1)
        x2_3 = x2_2.view(x.shape[0], -1)
        x = torch.cat([x1_3, x2_3], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        value = self.state_value(x)
        return value


if __name__ == '__main__':
    free_bw_matrix = np.zeros((10, 10), dtype=numpy.float32)
    delay_matrix = np.ones((10, 10), dtype=numpy.float32)
    loss_matrix = np.zeros((10, 10), dtype=numpy.float32)
    used_bw_matrix = np.ones((10, 10), dtype=numpy.float32) * 3
    pkt_err_matrix = np.zeros((10, 10), dtype=numpy.float32)
    pkt_drop_matrix = np.ones((10, 10), dtype=numpy.float32) * 2
    distance_matrix = np.ones((10, 10), dtype=numpy.float32) * 1
    tree_matrix = np.zeros((10, 10), dtype=numpy.float32)

    print(type(free_bw_matrix))

    tree_matrix[1][1] = 1  # 源节点置1, 注意列表下标从0开始
    tree_matrix[9][9] = -1  # 目的节点置-1

    # 将数据堆叠，得到8通道的14*14的张量：8*14*14
    state_tensor = torch.stack([torch.from_numpy(free_bw_matrix),
                                torch.from_numpy(delay_matrix),
                                torch.from_numpy(loss_matrix),
                                torch.from_numpy(used_bw_matrix),
                                torch.from_numpy(pkt_err_matrix),
                                torch.from_numpy(pkt_drop_matrix),
                                torch.from_numpy(distance_matrix),
                                torch.from_numpy(tree_matrix)], dim=0)

    state_tensor = torch.unsqueeze(state_tensor, dim=0)
    print(state_tensor.shape)
    print(state_tensor.size(1))

    # config = parse_config(Path('config/train_config.yml'))
    # print('A%02i' % 1)
    name = 'A%02i' % 1
    # print(config[name])

    # actor = Actor(config_path=Path(config['model_config_path']), domain_node=config[name]['node_num'])
    # actions = actor.forward(state_tensor)
    # print(actions)
    #
    # critic = Critic(config_path=Path(config['model_config_path']), domain_node=config[name]['node_num'])
    # value = critic.forward(state_tensor)
    # print(value)

    # domain = []
    # domain.append(free_bw_matrix[0:5, 0:5])
    # domain.append(delay_matrix[0:5, 0:5])
    # domain.append(loss_matrix[0:5, 0:5])
    # domain.append(used_bw_matrix[0:5, 0:5])
    # domain.append(pkt_err_matrix[0:5, 0:5])
    # domain.append(pkt_drop_matrix[0:5, 0:5])
    # domain.append(distance_matrix[0:5, 0:5])
    # domain.append(tree_matrix[0:5, 0:5])
    #
    # state_tensor = torch.stack([torch.from_numpy(domain[0]),
    #                             torch.from_numpy(domain[1]),
    #                             torch.from_numpy(domain[2]),
    #                             torch.from_numpy(domain[3]),
    #                             torch.from_numpy(domain[4]),
    #                             torch.from_numpy(domain[5]),
    #                             torch.from_numpy(domain[6]),
    #                             torch.from_numpy(domain[7])], dim=0)

    # test node_config
    node_config = parse_config(Path('config/node_config.yml'))
    # print(node_config[str(1)])
    nodes = node_config[name]
    print(list(nodes))
    print(nodes.index(11))
