import copy
import json
from utils.data_preprocess import get_wrong_ids
from char_similar.char_similarity_std import sim_pinyin, sim_w2v, sim_order, sim_number, sim_stroke, sim_struct, sim_component, sim_fourangle, sim_frequency, cal_sim_by_pinyin, cal_sim_by_shape, sim_pinyin
import pandas as pd
from random import random

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def merge_dict_and_rename(dict1, dict2):

    merged_dict = {}
    # 先把第一个字典的所有键值对加入到合并字典中
    merged_dict.update(dict1)

    # 遍历第二个字典，处理冲突的键
    for key, value in dict2.items():
        # 如果当前键不存在于合并字典中，直接添加
        if key not in merged_dict:
            merged_dict[key] = value
        else:
            # 生成新的键名（自动重命名）
            new_key = f"{key}_1"
            suffix = 1
            while new_key in merged_dict:
                suffix += 1
                new_key = f"{key}_{suffix}"
            # 将重命名后的键值对加入合并字典
            merged_dict[new_key] = value

    return merged_dict


def convert_candidates2features(_j):
    """
    :param _j: json格式的预测
    :return: # 为多个候选，按双打规则，生成了n(n-1)/2个对比特征项，以文本a\t文本b的md5作为特征
    """
    from utils.get_dif_words import sen2md5
    pres = _j['pres']
    ori = _j['ori']
    result = []
    pres_sample = []
    # 为每个预测生成特征
    scores = _j['scores']
    for index, p in enumerate(pres):
        sample = {
            'text': p,
        }
        # 从LLM结果中抽取出语义评分和流畅度评分
        try:
            score = scores[str(index)]
        except:
            print(line)
            raise Exception("解析失败")
        score = list(filter(lambda x: is_float(x[1]), score.items()))
        for (k, v) in score:
            if k == 'sentence':
                continue
            if 'f' in k or 'F' in k:
                sample['fluency'] = v
            else:
                sample['semantic'] = v
        print(line)
        assert "fluency" in sample
        # 计算每个候选和原始句子的字符级差异，作为特征；如果存在多个差异字符，取平均
        wrong_ids = get_wrong_ids(ori, p)
        if len(wrong_ids) > 0:
            phono_sims = []
            glyph_sims = []
            pinyin_sims = []
            component_sims = []
            stroke_sims = []

            for wrong_id in wrong_ids:
                c_pre = p[wrong_id]
                c_ori = ori[wrong_id]
                phono_sims.append(cal_sim_by_pinyin(c_pre, c_ori))
                glyph_sims.append(cal_sim_by_shape(c_pre, c_ori))
                pinyin_sims.append(sim_pinyin(c_pre, c_ori))
                component_sims.append(sim_component(c_pre, c_ori))
                stroke_sims.append(sim_stroke(c_pre, c_ori))

            phono_sim = sum(phono_sims) / len(phono_sims)
            glyph_sim = sum(glyph_sims) / len(glyph_sims)
            pinyin_sim = sum(pinyin_sims) / len(pinyin_sims)
            component_sim = sum(component_sims) / len(component_sims)
            stroke_sim = sum(stroke_sims) / len(stroke_sims)

        else:
            phono_sim = 1.0
            glyph_sim = 1.0
            pinyin_sim = 1.0
            component_sim = 1.0
            stroke_sim = 1.0

        sample['phono_sim'] = phono_sim
        sample['glyph_sim'] = glyph_sim
        sample['pinyin_sim'] = pinyin_sim
        sample['component_sim'] = component_sim
        sample['stroke_sim'] = stroke_sim
        pres_sample.append(sample)

    # 基于候选答案生成特征，每个特征用md5进行唯一标识
    for i in range(len(pres_sample)):
        for j in range(i + 1, len(pres_sample)):
            if random() > 0.5:
                m_dic = merge_dict_and_rename(pres_sample[i], pres_sample[j])
                m_dic['md5'] = sen2md5(f'{pres_sample[i]['text']}\t{pres_sample[j]['text']}')
                result.append(m_dic)
    # 为多个候选，按双打规则，生成了n(n-1)/2个对比特征项，以文本a\t文本b的md5作为特征
    return result


samples = []
for line in open('pre_top7_scores_6.txt', encoding='utf-8'):
    j = json.loads(line)
    pres = j['pres']
    tgt = j['tgt']
    ori = j['ori']

    is_llm = j['is_llm']
    if not is_llm:
        continue
    pres_sample = []
    # 为每个预测生成特征
    scores = j['scores']
    for index, p in enumerate(pres):
        sample = {
            'text': p,
        }
        # 从LLM结果中抽取出语义评分和流畅度评分
        try:
            score = scores[str(index)]
        except:
            print(line)
            raise Exception("解析失败")
        score = list(filter(lambda x: is_float(x[1]), score.items()))
        for (k, v) in score:
            if 'f' in k or 'F' in k:
                sample['fluency'] = v
            else:
                sample['semantic'] = v
        print(line)
        assert "fluency" in sample
        # 计算每个候选和原始句子的字符级差异，作为特征；如果存在多个差异字符，取平均
        wrong_ids = get_wrong_ids(ori, p)
        if len(wrong_ids) > 0:
            phono_sims = []
            glyph_sims = []
            pinyin_sims = []
            component_sims = []
            stroke_sims = []

            for wrong_id in wrong_ids:
                c_pre = p[wrong_id]
                c_ori = ori[wrong_id]
                phono_sims.append(cal_sim_by_pinyin(c_pre, c_ori))
                glyph_sims.append(cal_sim_by_shape(c_pre, c_ori))
                pinyin_sims.append(sim_pinyin(c_pre, c_ori))
                component_sims.append(sim_component(c_pre, c_ori))
                stroke_sims.append(sim_stroke(c_pre, c_ori))

            phono_sim = sum(phono_sims) / len(phono_sims)
            glyph_sim = sum(glyph_sims) / len(glyph_sims)
            pinyin_sim = sum(pinyin_sims) / len(pinyin_sims)
            component_sim = sum(component_sims) / len(component_sims)
            stroke_sim = sum(stroke_sims) / len(stroke_sims)

        else:
            phono_sim = 1.0
            glyph_sim = 1.0
            pinyin_sim = 1.0
            component_sim = 1.0
            stroke_sim = 1.0

        sample['phono_sim'] = phono_sim
        sample['glyph_sim'] = glyph_sim
        sample['pinyin_sim'] = pinyin_sim
        sample['component_sim'] = component_sim
        sample['stroke_sim'] = stroke_sim
        if p == tgt:
            sample['label'] = 1
        else:
            sample['label'] = 0
        samples.append(sample)

df = pd.DataFrame.from_records(samples)
print(df)
df.to_csv('./scores_non2.csv', encoding='utf-8', index=False)


