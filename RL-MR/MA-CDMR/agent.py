# @Project : Experiment-code 
# @File    : agent.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/17 14:44 
# @Describe:
# @Update  :
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from pathlib import Path
from collections import namedtuple
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model import Actor2, Critic2
from mutil_env import Environment
from dataset import file_path_yield, read_pickle

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Agent(torch.multiprocessing.Process):
    def __init__(self, src_dst, paths_dict, reward_dict, name, config, mode):
        super(Agent, self).__init__()
        self.mode = mode
        self.set_seed(int(name))
        self.name = 'A%02i' % name

        self.config = config
        self.actor_net = Actor2(Path(config['model_config_path']), self.config[self.name]['node_num'])
        self.critic_net = Critic2(Path(config['model_config_path']), self.config[self.name]['node_num'])

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.clip_param = config['clip_param']
        self.max_grad_norm = config['max_grad_norm']
        self.update_time = config['update_time']
        self.buffer_capacity = config['buffer_capacity']
        self.batch_size = config['batch_size']

        self.paths_dict = paths_dict
        self.reward_dict = reward_dict
        self.src_dst = src_dst

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), config['actor_lr'])
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), config['critic_lr'])
        self.env = Environment(self.config, self.name)
        self.env.set_seed(int(name))

    def choose_action(self, state):
        if self.mode == "train":
            action_prob = self.actor_net(state)
        elif self.mode == "test":
            with torch.no_grad():
                action_prob = self.actor_net(state)
        else:
            raise ValueError("mode error")
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), f'param/actor_net-{self.name}.pt')
        torch.save(self.critic_net.state_dict(), f'param/critic_net-{self.name}.pt')

    def load_weight(self, path):
        """
        加载模型
        :return: None
        """
        if self.name == "A00":
            actor = path + "/actor_net-A00.pt"
            critic = path + "/critic_net-A00.pt"
        elif self.name == "A01":
            actor = path + "/actor_net-A01.pt"
            critic = path + "/critic_net-A01.pt"
        elif self.name == "A02":
            actor = path + "/actor_net-A02.pt"
            critic = path + "/critic_net-A02.pt"
        else:
            raise FileNotFoundError

        if os.path.isfile(actor):
            self.actor_net.load_state_dict(torch.load(actor))
        if os.path.isfile(actor):
            self.critic_net.load_state_dict(torch.load(critic))

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state.numpy() for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.config['gamma'] * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                # print("!!!!", V.shape)
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                self.training_step += 1

        del self.buffer[:]  # clear experience

    def train(self):
        """
        for episode <-- 1 to n do
            foreach pkl_graph of pkl_dir do
                foreach (src, dst) of src_dst_list do
                    while state_ is not dst do
        :return:
        """
        total_step = 0
        episode_reward = []
        for episode in range(self.config['max_episode']):
            ep_reward = 0
            for index, pkl_path in enumerate(file_path_yield(self.config['files_dir_path'], s=self.config['pkl_start'],
                                                             n=self.config['pkl_end'], step=self.config['pkl_step'])):
                pkl_graph = read_pickle(pkl_path)
                for _src, _dst in self.src_dst:
                    self.env.update_pkl_graph(pkl_graph)  # 更新环境里的pkl_graph去生成新的TM
                    state = self.env.reset(_src, _dst)  # 重置环境

                    total_reward = 0  # 一个(src, dst)在一张pkl_graph找到路径的总奖励
                    # 开始训练
                    while True:
                        # 1.选择动作 ===>action_space的index, state输入神经网络前先增加一个批量大小维度：(1, 8, 10, 10)
                        action, action_prob = self.choose_action(torch.unsqueeze(state, dim=0))
                        # 2.根据选择的动作与环境交互，获得state_, reward, done
                        state_, reward, done, flag, path_info = self.env.step(action)
                        total_reward += reward
                        # 3.保存数据
                        trans = Transition(state, action, action_prob, reward, state_)
                        self.store_transition(trans)

                        # 4.根据保存的数据更新网络参数
                        if done:
                            if len(self.buffer) >= self.batch_size:
                                # if config.FREEZE_EPISODE > 50:
                                self.update(episode)
                            # self.writer.add_scalar('liveTime/livestep', episode, global_step=episode)
                            ep_reward += total_reward
                            break

                        # 更新状态
                        state = state_
                        total_step += 1
                    print(self.name + "-" + str(episode))
                    print(path_info)

                    self.paths_dict[(_src, _dst)] = path_info["path"]

            episode_reward.append(ep_reward)

        self.reward_dict[self.name] = episode_reward
        self.save_param()  # 保存模型

    def test(self):
        """
        foreach pkl_graph of pkl_dir do
            foreach (src, dst) of src_dst_list do
                while state_ is not dst do
        :return:
        """
        self.load_weight("param")
        for index, pkl_path in enumerate(file_path_yield(self.config['files_dir_path'], s=self.config['pkl_start'],
                                                         n=self.config['pkl_end'], step=self.config['pkl_step'])):
            pkl_graph = read_pickle(pkl_path)
            for _src, _dst in self.src_dst:
                self.env.update_pkl_graph(pkl_graph)  # 更新环境里的pkl_graph去生成新的TM
                state = self.env.reset(_src, _dst)  # 重置环境

                total_reward = 0  # 一个(src, dst)在一张pkl_graph找到路径的总奖励
                # 开始训练
                while True:
                    # 1.选择动作 ===>action_space的index, state输入神经网络前先增加一个批量大小维度：(1, 8, 10, 10)
                    action, action_prob = self.choose_action(torch.unsqueeze(state, dim=0))
                    # 2.根据选择的动作与环境交互，获得state_, reward, done
                    state_, reward, done, flag, path_info = self.env.step(action)
                    total_reward += reward

                    # 3.保存结果
                    if done:

                        break

                    # 更新状态
                    state = state_
                print(path_info)

                self.paths_dict[(_src, _dst)] = path_info["path"]


    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "test":
            self.test()
        else:
            raise ValueError("mode error")

