##########################################这里是调用云端的API##########################################
from langchain_community.embeddings import DashScopeEmbeddings

# 创建模型对象,不传参数默认的是text-embeddings-v1模型
model = DashScopeEmbeddings()

print(model.embed_query("你好，我喜欢你！"))
print(model.embed_documents(["你好，我喜欢你！","你好，我不喜欢你！","不好，我喜欢你！"]))



##########################################这里是调用本机的Ollama模型##########################################
# from langchain_ollama import OllamaEmbeddings

# model = OllamaEmbeddings(
#     model="qwen3-embedding:4b",
#     base_url="http://127.0.0.1:11434"
# )

# print(model.embed_query("你好，我喜欢你！"))
# print(model.embed_documents(["你好，我喜欢你！","你好，我不喜欢你！","不好，我喜欢你！"]))
