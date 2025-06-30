import json

import pandas as pd
from utils.llm_request import ds7b_api
from tqdm import tqdm
from llm_utils.llm_result_utils import extract_json_from_text


llm_failure_count = 0
total_count = 0

def replace_unk(t):
    return t.replace('[UNK]', '‽')

prompt_template = """
这几个句子是对原始句子的矫正，他们的差异在什么地方？对它们与原始句子的语义相似性和语言流畅度分别打分，分数在0-1之间。

原始句子：%s
候选句子：
%s

参考样例如下：

原始句子：我哎吃苹果
候选列表：
0. 我哎吃苹果
1. 我挨吃苹果
2. 我爱吃苹果
3. 我艾吃苹果

输出：
{
"0": {
'semantic': '分数',
'fluency': '分数'
},
"1": {
'semantic': '分数',
'fluency': '分数'
},
"2": {
'semantic': '分数',
'fluency': '分数'
},
"3": {
'semantic': '分数',
'fluency': '分数'
},
}
""".strip()


if __name__ == '__main__':
    oris = []
    tgts = []
    pres = []
    fn = '/data/datasets/sighan/train_pre_top3.tsv'
    df = pd.read_csv(fn, encoding='utf-8', names=['ori', 'tgt', 'pres'])
    with open('top3_scores.txt', 'w', encoding='utf-8') as f:
        for _, item in tqdm(df.iterrows(), total=len(df)):
            pres_list = eval(item['pres'])
            is_llm = 1
            for i in range(len(pres_list)):
                pres_list[i] = replace_unk(pres_list[i])
            ori = replace_unk(item['ori'])
            tgt = replace_unk(item['tgt'])
            # CSC模型未发现错误，直接采用
            if len(pres_list) == 1:
                oris.append(ori)
                tgts.append(tgt)
                pres.append(pres_list[0])
                is_llm = 0
                f.write(json.dumps({'ori': ori, 'tgt': tgt, 'pre': pres_list[0], 'is_llm': is_llm, 'pres': pres_list},
                                   ensure_ascii=False) + '\n')
                continue
            total_count += 1
            # ------------------------ 定制化prompt的内容 ------------------------
            candidates = '\n'.join(['%d. %s' % (i, pres_list[i]) for i in range(len(pres_list))])
            send_promt = prompt_template % (ori, candidates)
            ds_result = ds7b_api(send_promt)
            print(ds_result)
            try:
                score_dic = extract_json_from_text(ds_result)
                print('------------------')
                f.write(json.dumps({'ori': ori, 'tgt': tgt, 'is_llm': is_llm, 'pres': pres_list, 'scores': score_dic, 'lm_result': ds_result},
                                   ensure_ascii=False) + '\n')
            except:
                # -------------------------------------------------------------------
                # LLM结果格式有问题，采用候选的第一个
                llm_failure_count += 1
                is_llm = 0
                f.write(json.dumps({'ori': ori, 'tgt': tgt, 'pre': pres_list[0], 'is_llm': is_llm, 'pres': pres_list,
                                    'lm_result': ''},
                                   ensure_ascii=False) + '\n')
                continue

    print(llm_failure_count/total_count)


"""
LLM失败率

"""
