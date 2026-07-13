##########################################这里是调用云端的API##########################################
from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max")

# stream方法获得流式输出
res = model.stream(input="你好，请问在LPL职业联赛在BLG上单Bin的黑称有什么？ 我知道48-bin对吗？")

for chunk in res:
    print(chunk, end="", flush=True)


##########################################这里是调用本机的Ollama模型##########################################
# from langchain_ollama import OllamaLLM

# model = OllamaLLM(
#       model="qwen3:8b",
#       base_url="http://127.0.0.1:11434"
#   )

# res = model.stream(input="你好，请问在LPL职业联赛在BLG上单Bin的黑称有什么？ 我知道48-bin对吗？")

# for chunk in res:
#     print(chunk, end="", flush=True)