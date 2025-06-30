from collections import defaultdict


def predict_print_fmt(oris, tgts, pres, write_file=None):
    # 写入文件
    if write_file:
        f =  open(write_file, 'w', encoding='utf-8')
    # 统计整个集合上的差异词对
    global_diff_pairs = defaultdict(int)  # 全局差异词对统计
    total_diff = 0  # 总差异字符数

    for i in range(len(oris)):
        ori = oris[i]
        tgt = tgts[i]
        pre = pres[i]

        # 处理当前三元组
        marked_ori = []
        marked_pre = []
        marked_tgt = []
        diff_indices = []
        diff_pairs = defaultdict(int)  # 当前三元组的差异词对统计

        max_len = max(len(ori), len(pre), len(tgt))

        for j in range(max_len):
            c_ori = ori[j] if j < len(ori) else ' '
            c_pre = pre[j] if j < len(pre) else ' '
            c_tgt = tgt[j] if j < len(tgt) else ' '

            if c_pre != c_tgt:
                diff_indices.append(j)
                pair = (c_pre, c_tgt)  # a->b对
                diff_pairs[pair] += 1
                total_diff += 1

                marked_ori.append(f'[{c_ori}]')
                marked_pre.append(f'[{c_pre}]')
                marked_tgt.append(f'[{c_tgt}]')
            else:
                marked_ori.append(c_ori)
                marked_pre.append(c_pre)
                marked_tgt.append(c_tgt)

        # 合并标记后的字符串
        marked_ori_str = ''.join(marked_ori)
        marked_pre_str = ''.join(marked_pre)
        marked_tgt_str = ''.join(marked_tgt)

        # 打印当前三元组的结果
        print(f"第 {i + 1} 组数据：")
        print(f"差异位置：{diff_indices}")
        print(f"ori: {marked_ori_str}")
        print(f"pre: {marked_pre_str}")
        print(f"tgt: {marked_tgt_str}")
        print('-' * 40)

        if write_file:
            # 写入当前三元组的结果
            f.write(f"第 {i + 1} 组数据：\n")
            f.write(f"差异位置：{diff_indices}\n")
            f.write(f"ori: {marked_ori_str}\n")
            f.write(f"pre: {marked_pre_str}\n")
            f.write(f"tgt: {marked_tgt_str}\n")
            f.write('-' * 40 + '\n')

        # 更新全局差异词对统计
        for pair, count in diff_pairs.items():
            global_diff_pairs[pair] += count

    # 打印全局差异词对统计
    print("全局差异词对统计：")
    if write_file:
        f.write("全局差异词对统计：\n")
    for (a, b), count in global_diff_pairs.items():
        percentage = (count / total_diff) * 100
        print(f"{a}->{b}: 出现次数={count}, 占比={percentage:.2f}%")

        if write_file:
            f.write(f"{a}->{b}: 出现次数={count}, 占比={percentage:.2f}%\n")
        # 写入全局差异词对统计
    f.close()
