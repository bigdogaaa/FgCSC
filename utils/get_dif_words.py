import copy
import json

import os
import numpy as np
import hashlib
import spacy
from config.config_reader import Config


config = Config()

# 从配置文件中获取数据库连接信息
db_config = {
    'enable_user_dic': config.get('word_segment', 'enable_user_dict'),
}
if db_config['enable_user_dic']:
    nlp = spacy.load('zh_core_web_sm')
    print('Loading user dic for word segmentation.')
    nlp.tokenizer.initialize(pkuseg_model="mixed") # pkuseg_user_dict=os.path.dirname(__file__)+'/../static/'+'jieba_user_dict.txt'
    words = open(os.path.dirname(__file__)+'/../static/'+'user_dict.txt', 'r', encoding='utf-8').readlines()
    nlp.tokenizer.pkuseg_update_user_dict(words)



def sen2md5(sentence):
    return hashlib.md5(sentence.encode(encoding='utf-8')).hexdigest()
