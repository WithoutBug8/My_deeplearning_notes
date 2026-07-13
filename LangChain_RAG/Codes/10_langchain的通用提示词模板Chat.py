from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个诗人，可以作诗。"),
    MessagesPlaceholder("history"),
    ("human", "请再来一首诗"),
])

history_data = [
    ("human", "你来写一首诗歌"),
    ("ai", "世人都晓神仙好，惟有功名忘不了！古今将相在何方？荒冢一堆草没了。世人都晓神仙好，只有金银忘不了！终朝只恨聚无多，及到多时眼闭了。世人都晓神仙好，只有娇妻忘不了！君生日日说恩情，君死又随人去了。世人都晓神仙好，只有儿孙忘不了！痴心父母古来多，孝顺儿孙谁见了。"),
    ("human", "好诗，好诗啊！请再来一个"),
    ("ai", "力拔山兮气盖世，时不利兮骓不逝。骓不逝兮可奈何，虞兮虞兮奈若何！"),
]

# 注入对话的历史信息,注意invoke函数的返回值不是字符串
prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()

model = ChatTongyi(model="qwen3-max")
print(model.invoke(input=prompt_text).content) #content只取模型回复的内容