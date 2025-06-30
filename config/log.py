#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import logging.config

# 加载日志配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
logging_config_file = os.path.join(current_dir, 'logging.conf')
logging.config.fileConfig(logging_config_file)

# 创建一个通用的日志记录器，可以在项目中任何地方使用
logger = logging.getLogger(__name__)