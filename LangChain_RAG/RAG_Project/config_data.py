from networkx.algorithms.similarity import simrank_similarity

md5_path = "./data/md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./data/chroma_db"

# Spliter文本分割器
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。","！","？"," ",""]
max_split_char_number = 1000

# 相似度检索的阈值，每次返回匹配的文档数量
similarity_threshold = 2

# 配置model
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# session状态配置
session_config = {
        "configurable": {
            "session_id": "user_001",
        }
}
