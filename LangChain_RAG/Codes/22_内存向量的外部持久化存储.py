from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma

# Chroma 向量数据库(底层是基于SQLite)很轻巧的一个数据库
vector_store = Chroma(
    collection_name="test",                                       # 当前向量存储的名字，类似于数据库表名
    embedding_function=DashScopeEmbeddings(),                     # 嵌入模型
    persist_directory="./LangChain_RAG/Codes/chroma_db"           # 存储的位置
)


loader = CSVLoader(
    file_path='./LangChain_RAG/Codes/data/info.csv',
    encoding='UTF-8',
    source_column="source"     # 指定数据的来源，这样原来默认的来源是info.csv; 现在就是source中的数据了
)

documents = loader.load()

# 向量的存储的增
vector_store.add_documents(
    documents=documents,                                    # 被添加的文档，类型list[document]
    ids=["id"+str(i) for i in range(1, len(documents)+1)]  # 给体检的文档提供ID(字符串)list[str]
)
# 向量的存储的删
vector_store.delete(
    ["id1","id2"]
)
# 向量的存储的查，返回的类型是list[document]
result = vector_store.similarity_search(
    "Python是不是简单易学呀",
    k=3                 # 要几个检索的结果
)
print(result)

