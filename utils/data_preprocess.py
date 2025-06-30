import os
from char_similar import std_cal_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
# Example usage
user_dict_words = open(current_dir+'/../static/user_dict.txt', 'r', encoding='utf-8').read().strip().split('\n')


def get_wrong_ids(ori_text, cor_text):
    wrong_ids = []
    for index, a, b in zip(range(len(ori_text)), ori_text, cor_text):
        if a != b:
            wrong_ids.append(index)
    # print(wrong_ids)
    return wrong_ids


def convert_mask2wrong_ids(mask):
    wrong_ids = []
    for i in range(len(mask)):
        if mask[i] == '1':
            wrong_ids.append(i)
    return wrong_ids

def find_char_differences(sen1, sen2):
    """
    输入两个句子，输出两个句子差异部分的字符及其位置（字符级别）。

    参数:
        sen1 (str): 第一个句子
        sen2 (str): 第二个句子

    返回:
        tuple: (sen1_diff, sen2_diff)，分别表示 sen1 和 sen2 中的差异部分，每个元素是元组 (位置, 字符)
    """
    diff = difflib.ndiff(sen1, sen2)
    sen1_diff = []
    sen2_diff = []
    i = j = 0  # 初始化两个句子的当前位置索引

    for line in diff:
        if line.startswith('? '):
            continue  # 忽略差异建议行
        code = line[:2]
        char = line[2:]

        if code == '- ':
            # sen1 中删除的字符，记录当前位置并递增 sen1 的索引
            sen1_diff.append((i, char))
            i += 1
        elif code == '+ ':
            # sen2 中添加的字符，记录当前位置并递增 sen2 的索引
            sen2_diff.append((j, char))
            j += 1
        elif code == '  ':
            # 两个句子共有的字符，同时递增索引
            i += 1
            j += 1
    sims = []
    for (x_ind, x), (y_ind, y) in zip(sen1_diff, sen2_diff):
        sims.append(std_cal_sim(x, y))

    return sen1_diff, sen2_diff, sims


def find_list_differences_ordered(l1, l2):
    """
    输入两个字符串列表，输出两个列表差异部分的元素（保留顺序）。

    参数:
        l1 (list): 第一个字符串列表
        l2 (list): 第二个字符串列表

    返回:
        tuple: (l1_diff, l2_diff)，分别表示 l1 和 l2 中的差异部分
    """
    # 找出 l1 中独有的元素（保留顺序）
    l1_diff = [item for item in l1 if item not in l2]

    # 找出 l2 中独有的元素（保留顺序）
    l2_diff = [item for item in l2 if item not in l1]

    return l1_diff, l2_diff


import difflib

def find_char_differences_pretty(sen1, sen2):
    """
    输入两个句子，返回修改操作列表，格式为["a->b", "c->d"]

    参数:
        sen1 (str): 原始句子
        sen2 (str): 修改后的句子

    返回:
        list: 包含所有修改操作的列表，格式为["原字符->新字符"]
    """
    diff = list(difflib.ndiff(sen1, sen2))
    changes = []
    i = 0

    while i < len(diff):
        d = diff[i]
        # 处理替换操作（- 后接 +）
        if d.startswith('- ') and i + 1 < len(diff) and diff[i + 1].startswith('+ '):
            changes.append(f"{d[2:]}->{diff[i + 1][2:]}")
            i += 2
        # 处理删除操作（单独的 -）
        elif d.startswith('- '):
            changes.append(f"{d[2:]}->")
            i += 1
        # 处理新增操作（单独的 +）
        elif d.startswith('+ '):
            changes.append(f"->{d[2:]}")
            i += 1
        # 无修改内容跳过
        else:
            i += 1

    return changes

