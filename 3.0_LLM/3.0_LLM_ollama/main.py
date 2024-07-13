# 需要先安装依赖，本次使用qwen2
# ollama pull qwen2:7b
# ollama run qwen2:7b
import ollama

response = ollama.chat(model='qwen2:7b', messages=[
  {
    'role': 'user',
    'content': '写一首五言绝句',
  },
])
print(response['message']['content'])