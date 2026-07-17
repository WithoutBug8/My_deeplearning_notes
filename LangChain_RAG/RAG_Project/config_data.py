
md5_path = "./data/md5.text"

# Chroma
collection_name = "rag"
persist_directory = "./data/chroma_db"

# Spliter文本分割器
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。","！","？"," ",""]
max_split_char_number = 1000