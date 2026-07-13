from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi

# 通用的示例模板
example_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")

# 示例的动态数据注入，要求是list字典格式
examples_data = [
    {"word": "快乐", "antonym": "伤心"},
    {"word": "大", "antonym": "小"}
]

few_shot_template = FewShotPromptTemplate(
    example_prompt = example_template, # 示例数据的模板
    examples = examples_data, # 示例数据(动态数据), list字典格式
    prefix = "请告知我单词的反义词，我提供如下的示例", # 示例之前的提示词
    suffix = "现在，请告诉我单词“{input_word}”的反义词是什么？", # 示例之后的提示词
    input_variables = ["input_word"] # 输入变量,声明在前缀或者后缀需要注入的变量名
)

prompt_text = few_shot_template.invoke(input={"input_word": "好看"})

model = Tongyi(model="qwen-max")
print(model.invoke(input=prompt_text))