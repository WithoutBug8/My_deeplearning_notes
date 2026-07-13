from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{name},刚生了一个{gender}孩，你帮我起一个名字，简单回答。"
)

model = Tongyi(model="qwen-max")

# 调用.format()方法，注入信息
# prompt_text = prompt_template.format(name="张", gender="女")


# res = model.invoke(input=prompt_text)

# print(res)

# Langchain最常用的构建执行链条
chain = prompt_template | model
res = chain.invoke(input={"name": "张", "gender": "女"})
print(res)