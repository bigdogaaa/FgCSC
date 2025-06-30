import json
from judge.evaluate import calculate_metric
from utils.data_preprocess import find_char_differences
from pprint import pprint
from utils.llm_request import ds7b_api
from llm_utils.llm_result_utils import extract_json_from_text
from tqdm import tqdm
"""
用于对未成功请求llm的样本进行再次请求，以获得语义相似性和语言流畅度分数
"""


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

    count = 0
    with open('top3_scores_3.txt', 'w', encoding='utf-8') as f:
        for line in tqdm(open('./top3_scores_2.txt', encoding='utf-8')):
            j = json.loads(line)
            pres_list = j['pres']
            ori = j['ori']
            tgt = j['tgt']
            is_llm = j['is_llm']

            if 'lm_result' in j:
                if j['lm_result'] == '':
                    count += 1
                    # 再次生成score
                    candidates = '\n'.join(['%d. %s' % (i, pres_list[i]) for i in range(len(pres_list))])
                    send_promt = prompt_template % (ori, candidates)
                    ds_result = ds7b_api(send_promt)
                    print(ds_result)
                    try:
                        score_dic = extract_json_from_text(ds_result)
                        print('------------------')
                        f.write(json.dumps(
                            {'ori': ori, 'tgt': tgt, 'is_llm': is_llm, 'pres': pres_list, 'scores': score_dic,
                             'lm_result': ds_result},
                            ensure_ascii=False) + '\n')
                        continue
                    except:
                        # -------------------------------------------------------------------
                        # LLM结果格式有问题，采用候选的第一个
                        llm_failure_count += 1
                        is_llm = 0
                        f.write(line)
                        continue

            # 直接添加回去
            f.write(line)


