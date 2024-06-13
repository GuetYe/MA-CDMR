# @Project : Experiment-code 
# @File    : train_main.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/17 14:22 
# @Describe:
# @Update  :

import random
import torch.multiprocessing as mp
from agent import Agent
from parser_config import parse_config
from pathlib import Path
from draw_tools import draw_episode_reward_all
from utils import cal_avg_reward

random.seed(2024)


def main():
    # 智能体共用的全局字典
    reward_dict = mp.Manager().dict()
    paths_dict = mp.Manager().dict()

    # 智能体共用的全局配置
    config = parse_config(Path('config/train_config.yml'))

    # 1. 得到域间组播树和得到域间转发的边间节点
    src_dst_all = [[(3, 9), (3, 8)], [(11, 13), (11, 18)], [(22, 26), (22, 28)]]

    # 2. 每个域的agent负责各自的任务
    agents = []
    mode = "train"
    for num in range(config['agent_num']):
        src_dst = src_dst_all[num]
        agent = Agent(src_dst, paths_dict, reward_dict, num, config, mode)
        agent.start()
        agents.append(agent)

    for agent in agents:
        agent.join()

    # 3. 每个域智能体获得的奖励,以及平均奖励
    reward_list = list(reward_dict.values())
    draw_episode_reward_all(reward_list[0], "c1_reward_")
    draw_episode_reward_all(reward_list[1], "c2_reward_")
    draw_episode_reward_all(reward_list[2], "c3_reward_")

    # 整体平均奖励
    ep_avg_reward = cal_avg_reward(reward_dict)
    draw_episode_reward_all(ep_avg_reward, "avg_reward_")

    # 4. 得到域间组播树和各域的域内组播树，进行组合
    print(paths_dict)


if __name__ == '__main__':
    main()

