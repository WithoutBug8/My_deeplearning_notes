from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

# 创建解析器
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# 创建模型
model = ChatTongyi(model="qwen3-max")

# 创建第一个提示词模板
first_prompt = PromptTemplate.from_template(
    "我邻居姓：{name}，刚生了一个{gender}宝宝，请帮我起一个名字,"
    "并封装为JSON格式返回给我，要求key是name，value是你起的名字，请严格遵守格式要求。"
)

# 创建第二个提示词模板
second_prompt = PromptTemplate.from_template(
    "姓名：{name}，请帮我解析含义。"
)

# 构建链
chain = first_prompt | model | json_parser | second_prompt | model | str_parser

for chunk in chain.stream({"name": "张", "gender": "男"}):
    print(chunk, end="", flush=True)