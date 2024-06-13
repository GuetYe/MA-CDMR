# @Project : Experiment-code 
# @File    : parser_config.py
# @IDE     : PyCharm 
# @Author  : hhw
# @Date    : 2024/1/4 11:04 
# @Describe:
# @Update  :
import re
import os
import yaml
from pathlib import Path
from easydict import EasyDict as edict

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def parse_config(config_path: str or Path):
    """parse yaml

    Args:
        config_path (Path): yaml config path, type of pathlib.Path

    Returns:
        EasyDict: a dict of config
    """
    if isinstance(config_path, str):
        config_path = Path(config_path)

    assert Path.exists(config_path), f"{config_path} is not exists"

    with open(config_path, 'r') as f:
        config = edict(yaml.load(f, Loader=loader))

    assert isinstance(config, dict), "config is not a dict type"

    return config


def modify_config(config_path: str or Path, keyword, value):
    if isinstance(config_path, str):
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=loader)

    config[keyword] = value

    with open(config_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    # 获取当前文件所在的目录
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # print(cur_dir)
    modify_config('config/train_config.yml', 'lr', 1e-4)
