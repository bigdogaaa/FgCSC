import regex as re
import json

pattern1 = r'```json\s*(.*?)\s*```'
pattern2 = r'\{(?:[^{}]|(?R))*\}'

def extract_json_from_text(text):
    """
    抽取文本中的JSON数据。支持以下两种情形：
    1. 被```json包裹的JSON内容
    2. 独立的JSON内容
    """
    # 尝试匹配第一种情况：```json...```
    match1 = re.search(pattern1, text, re.DOTALL)
    if match1:
        json_str = match1.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            try:
                return eval(json_str)
            except:
                raise ValueError(f"JSON解析失败: {e}")

    # 如果没有找到第一种情况，尝试查找第二种情况: 独立的 JSON 内容
    # 匹配所有可能的 JSON 对象（以 "{" 开头且可能跨行）
    matches = re.findall(pattern2, text, re.DOTALL)

    for possible_json in matches:
        # 去除可能的前后空格或换行符
        possible_json = possible_json.strip()
        try:
            return json.loads(possible_json)
        except json.JSONDecodeError:
            try:
                return eval(possible_json)
            except:
                pass  # 如果解析失败，继续下一个

    # 如果所有尝试都失败
    raise ValueError("未找到有效的JSON内容")


if __name__ == '__main__':

    # 示例用法
    text1 = """
    从语义上看，这些表达都属于演唱会的不同方式，只是表达的动词不同。因此，语义相似性应该较高，但具体分数可能根据表达的准确性而有所不同。

接下来评估语言流畅度。句子0中的“听演唱会”可能更常用，而“唱演唱会”可能更准确，因为“唱”更直接关联到演唱会的表演。句子1中的“看演唱会”可能更适用于电影或电视节目，而不像演唱会是现场表演，所以流畅度可能会稍低。

综合来看，句子0和1的语义相似性较高，流畅度方面，句子0可能稍好，因为“听”更广泛，而句子1中的“看”可能不如“听”常用。句子2中的“唱”更准确，但流畅度可能稍低，因为“唱”可能在某些情况下不如“听”或“看”流畅。

现在，我需要给每个候选句子打分。语义相似性在0.8到0.9之间，因为表达方式略有不同，但都接近原始句子。流畅度方面，句子0可能得0.85，句子1得0.8，句子2得0.85，因为“唱”可能稍微影响流畅度。

最后，整理结果，确保分数在0到1之间，并且解释每个得分的原因。
</think>

```json
{
"0": {
'semantic': 0.89,
'fluency': 0.85
},
"1": {
'semantic': 0.88,
'fluency': 0.80
},
"2": {
'semantic': 0.87,
'fluency': 0.85
}
}
```
       """

    result = extract_json_from_text(text1)
    print(result)
    print(type(result))

