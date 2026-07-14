'''
2026年7月14日，这个函数已经被废弃了，注意不要再使用了
'''
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


model = ChatTongyi(model="qwen3-max")

prompt = ChatPromptTemplate.from_messages([

    ("system", "你需要根据会话历史来回答用户问题。"),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "请回答如下问题：{input}")

])

str_parser = StrOutputParser()

def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

base_chain = prompt | print_prompt | model | str_parser

store = {}   # 空字典，key是session，value是InMemoryChatMessageHistory类对象

# 实现通过会话ID获取获取InMemoryChatMessageHistory类对象函数
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]


# 创建一个新的链，对原有的链增强功能：附加上历史消息啊
conversation_chain = RunnableWithMessageHistory(
    base_chain,                         # 被增强的原有链
    get_history,                        # 通过会话id获取InMemoryChatMessageHistory类对象
    input_messages_key="input",         # 表示用户在模版中输入的占位符
    history_messages_key="chat_history" # 表示用户输入在模版中的占位符
)


if __name__ == '_main_':
    # 固定格式，添加LangChain的配置，为当前程序配置sessionID
    session_config = {
        "configurable": {
            "session_id":"user_001"
        }
    }
    res = conversation_chain.invoke({"input":"小明有两个猫"},session_config)
    print("第一次执行", res)

    res = conversation_chain.invoke({"input":"小刚有一个狗"},session_config)
    print("第二次执行", res)

    res = conversation_chain.invoke({"input":"总共有几个宠物"},session_config)
    print("第三次执行", res)