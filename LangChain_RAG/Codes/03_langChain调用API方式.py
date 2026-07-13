from langchain_community.llms.tongyi import Tongyi

model = Tongyi(model="qwen-max")

res = model.invoke(input="你好，请问在LPL职业联赛在BLG上单Bin的黑称有什么？")

print(res)

