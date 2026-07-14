'''
我们需要借助jq这个包才能实现JSONLoader

jq语法具体是：
1. `.`表示整个JSON对象
2. `[]`代表的是数组
举个例子`.name`抽取name属性；`.[]`获得其中的字典
'''

from langchain_community.document_loaders import JSONLoader
 
loader = JSONLoader(
    file_path="./LangChain_RAG/Codes/data/stu_json_lines.json",
    jq_schema=".name",
    text_content=False,     # 告知JSONLoader 我抽取的内容不是字符串
    json_lines=True         # 告知JSONLoader 这是一个JSONLines文件（每一行都是一个独立的标准JSON）
)
 
document = loader.load()
print(document)
