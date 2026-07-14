'''
在Chain的链中，我们有时候想要自己写一个函数放到链中，所以自定义函数是非常重要的，
这里引入RunnableLambda类，可以将普通函数转换为Runnable接口示例
'''

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda

# 创建解析器
str_parser = StrOutputParser()

# 创建模型
model = ChatTongyi(model="qwen3-max")

# 创建第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{name}，刚生了一个{gender}宝宝，请帮我起一个名字,不要额外信息"
)

# 创建第二个提示词模板
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我解析含义。"
)

# 创建runnableLambda类，传入的参数AIMessage，返回的函数是dict字典格式
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})


Chain = first_prompt | model | my_func | second_prompt | model | str_parser

for chunk in Chain.stream({"name":"张","gender":"女"}):
    print(chunk, end="", flush=True)