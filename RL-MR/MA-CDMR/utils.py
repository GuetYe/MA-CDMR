# @Project : Experiment-code 
# @File    : utils.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/4 15:41 
# @Describe:
# @Update  :
import pickle
import numpy as np
from parser_config import parse_config
from pathlib import Path

config = parse_config(Path('config/train_config.yml'))


def save_experimental_data(data, filepath, name: str):
    """
    将实验数据存储成pickle的文件
    data:将要存储的数据
    filepath:存储数据的路径
    name:文件的名字
    """
    filename = "{}/{}.pkl".format(filepath, name)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()
    return filename


def load_experimental_data(filepath):
    """
    获取实验的数据
    filepath:存储数据的路径
    """
    f = open(filepath, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def cal_avg_reward(reward_dict):
    """
    计算每个episode的平均奖励值
    :param reward_dict:
    :return:
    """
    ep_avg_reward = np.zeros(config['max_episode'], np.float32)
    values = list(reward_dict.values())
    for i in range(len(values)):
        ep_avg_reward = np.add(ep_avg_reward, values[i])

    ep_avg_reward = np.divide(ep_avg_reward, len(values))
    return ep_avg_reward


if __name__ == '__main__':
    pass
