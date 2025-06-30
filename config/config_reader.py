#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import yaml
import os


class Config:
    def __init__(self):
        # 读取配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_config_file = os.path.join(current_dir, './db_config.yml')
        # 初始化配置解析器
        self.config = yaml.safe_load(open(db_config_file, encoding='utf-8'))

    def get(self, section, option):
        # 获取配置项的值
        return self.config[section][option]


# 创建配置类的单例实例
# config_instance = Config()
#
#
# def get_config(section, option):
#     # 通过单例模式访问配置信息
#     return config_instance.get(section, option)
