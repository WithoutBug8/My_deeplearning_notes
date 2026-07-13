from langchain_community.llms.tongyi import Tongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


model = Tongyi(model="qwen-max")

# messages = [
#     SystemMessage(content="你是一个LPL多年的老观众，你对LPL职业联赛的选手和战队非常熟悉。"),
#     HumanMessage(content="你好，请问在LPL职业联赛在IG的上单选手TheShy的黑称有什么？"),
#     AIMessage(content="你好！在LPL职业联赛中，BLG战队的上单选手Bin的黑称主要是“48-bin”。这个称号来源于他在比赛中的表现和一些粉丝的调侃。希望这个信息对你有帮助！"),
#     HumanMessage(content="按照你上面说的，那么WBG战队的Breath选手的黑称是什么？")
# ]

###############################################也可以换成这种的简写形式############################################################
messages = [
    ("system", "你是一个LPL多年的老观众，你对LPL职业联赛的选手和战队非常熟悉。"),
    ("human", "你好，请问在LPL职业联赛中IG的上单选手TheShy的黑称有什么？"),
    ("ai", "你好！在LPL职业联赛中，BLG战队的上单选手Bin的黑称主要是“48-bin”。这个称号来源于他在比赛中的表现和一些粉丝的调侃。希望这个信息对你有帮助！"),
    ("human", "按照你上面说的，那么WBG战队的Breath选手的黑称是什么？")
]

res = model.stream(input=messages)


for chunk in res:
    print(chunk, end="", flush=True)