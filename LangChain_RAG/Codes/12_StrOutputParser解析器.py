'''
输出解析器（Output Parser）的作用：
1. 大语言模型返回的结果通常不是普通字符串，而是 AIMessage 等消息对象。
2. 输出解析器负责将模型输出转换成程序更容易处理的格式，例如字符串、JSON、列表等。
3. 本例使用的是 StrOutputParser，它会提取模型回复的文本内容（字符串），方便后续继续传递给下一个组件。

本例执行流程：
PromptTemplate -> ChatTongyi -> StrOutputParser -> ChatTongyi

执行过程如下：
① PromptTemplate 根据输入变量生成提示词；
② ChatTongyi 根据提示词生成一个名字；
③ StrOutputParser 将模型返回的 AIMessage 提取为纯文本字符串；
④ 该字符串作为新的输入再次发送给 ChatTongyi，让模型继续加工或润色结果。
'''
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template("我邻居姓：{name}，刚生了一个{gender}宝宝，请帮我起一个名字。")

chain = prompt | model | parser | model

res = chain.invoke({"name": "张", "gender": "男"})
print(res.content)