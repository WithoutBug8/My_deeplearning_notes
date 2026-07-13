import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "你是滔博Top E-Sport"},
        {"role": "assistant", "content": "好的，我对滔博战队非常了解，请问你想了解哪方面的信息？"},
        {"role": "user", "content": "请问陀螺和滔博之间有什么关系？"},
    ]
)

print(response.choices[0].message.content)
