'''
PyPDFLoader加载器：是加载PDF用的，使用前请先安装
pip install pypdf
'''
from langchain_community.document_loaders import PyPDFLoader
 
loader = PyPDFLoader(
    file_path="./LangChain_RAG/Codes/data/pdf2.pdf",
    mode="single",          # 默认是page模式，每个页面形成一个Document文档对象，
                            # single模式，不管有多少页，只返回1个Document对象
    password="itheima"      # PDF 有密码的在这里设置
)
 
i = 0
# 文件比较大，使用懒加载模式
for doc in loader.lazy_load():
    i += 1
    print(doc)
    print("="*20, i)