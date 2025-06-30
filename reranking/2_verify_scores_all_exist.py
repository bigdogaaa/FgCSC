import json

failed_num = 0
total = 0
with open('top3_scores_2.txt', 'w', encoding='utf-8') as f:
    for line in open('top3_scores.txt', encoding='utf-8'):
        j = json.loads(line)
        pres_list = j['pres']
        ori = j['ori']
        tgt = j['tgt']
        is_llm = j['is_llm']

        if not is_llm:
            if 'lm_result' not in j:
                f.write(line)
            else:
                # 上轮生成分数解析失败，再次生成
                if j['lm_result'] == '':
                    failed_num+=1
                    print(line)
                    f.write(line)
                else:
                    f.write(line)
        else:
            total+=1
            if 'scores' in j:
                try:
                    assert '2' in j['scores']
                    check_num = 0
                    scores = j['scores']['6']
                    for k, v in scores.items():
                        if k == 'sentence':
                            continue
                        if 'f' in k:
                            if type(float(v)) == float:
                                check_num+=1
                        elif 's' in k:
                            if type(float(v)) == float:
                                check_num+=1
                    assert check_num == 2
                    f.write(line)
                except:
                    print(line)
                    failed_num+=1
                    # 上轮生成分数没有包含所有候选，再次生成
                    j['is_llm'] = 0
                    j['lm_result'] = ''
                    f.write(json.dumps(j, ensure_ascii=False)+'\n')
            else:
                failed_num += 1
                j['is_llm'] = 0
                j['lm_result'] = ''
                f.write(json.dumps(j, ensure_ascii=False) + '\n')

print(failed_num/total)