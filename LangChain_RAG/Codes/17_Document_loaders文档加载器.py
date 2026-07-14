'''
所有文档加载器有一个统一的接口
1. load(): 一次性加载全部文档
2. lazy_load(): 延迟流式传输文档，对付大型数据集，避免内存溢出
'''
from langchain_community.document_loaders import CSVLoader
 
 
loader = CSVLoader(
    file_path="./LangChain_RAG/Codes/data/stu.csv",
    csv_args={
        "delimiter": ",",       # 指定分隔符
        "quotechar": '"',       # 指定带有分隔符文本的引号包围是单引号还是双引号
        # 如果数据原本有表头，就不要下面的代码，如果没有可以使用
        "fieldnames": ['name', 'age', 'gender', '爱好']
    },
    encoding="utf-8"            # 指定编码为UTF-8
)
 
# 批量加载 .load()   ->  [Document, Document, ...]
# documents = loader.load()
#
# for document in documents:
#     print(type(document), document)
 
# 懒加载  .lazy_load()  迭代器[Document]
for document in loader.lazy_load():
    print(document)