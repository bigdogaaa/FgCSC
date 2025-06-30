from openai import OpenAI
import requests


from config.config_reader import Config

config = Config()

# 从配置文件中获取数据库连接信息
db_config = {
    'qw25_14b': config.get('llm', 'qw25_14b_url'),
    'qw25_7b': config.get('llm', 'qw25_7b_url'),
    'ds_32b': config.get('llm', 'deepseek_32b_url'),
    'ds_7b': config.get('llm', 'deepseek_7b_url'),
    'temperature': float(config.get('llm', 'temperature')),
}

openai_api_key = "EMPTY"

client_qw257b = OpenAI(
    api_key=openai_api_key,
    base_url=db_config['qw25_7b'],
)
client_qw25 = OpenAI(
    api_key=openai_api_key,
    base_url=db_config['qw25_14b'],
)

client_ds32b = OpenAI(
    api_key=openai_api_key,
    base_url=db_config['ds_32b'],
)

client_ds7b = OpenAI(
    api_key=openai_api_key,
    base_url=db_config['ds_7b'],
)

def ds32b_api(content, sys_content=""):
    completion = client_ds32b.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content}
        ],
        temperature=0.6,
    )
    msg = completion.choices[0].message
    return msg.content

def ds7b_api(content, sys_content=""):
    completion = client_ds7b.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content}
        ],
        temperature=0.6,
    )
    msg = completion.choices[0].message
    return msg.content

def qw257b_api(content, sys_content=""):
    completion = client_qw257b.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content}
        ],
        temperature=db_config['temperature'],
    )
    msg = completion.choices[0].message
    # print(msg)
    return msg.content


def qw25_api(content, sys_content=""):
    completion = client_qw25.chat.completions.create(
        model="Qwen/Qwen2.5-14B-Instruct",
        messages=[
            {"role": "system", "content": sys_content},
            {"role": "user", "content": content}
        ],
        temperature=db_config['temperature'],
    )
    msg = completion.choices[0].message
    # print(msg)
    return msg.content


if __name__ == '__main__':
    sys_content = ""
    content = """
    下面两句话谁更流畅？
    ["那边有好吃的义大利面，有很亲切的服务生，还有那个地方很热闹。", "那边很好吃的义大利面，有很亲切的服务生，还有那个地方很热闹。"]
    """
    result = qw257b_api(content, sys_content=sys_content)
    print(result)





