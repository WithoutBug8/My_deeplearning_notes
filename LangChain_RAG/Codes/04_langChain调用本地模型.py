from langchain_ollama import OllamaLLM
  
  
model = OllamaLLM(
      model="qwen3:8b",
      base_url="http://127.0.0.1:11434"
  )

res = model.invoke(input="你好，请问在LPL职业联赛在BLG上单Bin的黑称有什么？")

print(res)

