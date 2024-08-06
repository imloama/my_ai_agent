from ollama import AsyncClient
import asyncio

ollama_host = 'http://localhost:11434'
ollama_model='qwen2:1.5b'

async def chat_by_ollama(question):
    message = {'role': 'user', 'content': question}
    response = await AsyncClient(host=ollama_host).chat(model=ollama_model, messages=[message])
    print(response)
    print( response["message"]["content"])
    return response

if __name__ == '__main__':
    asyncio.run(chat_by_ollama("今天是2024年8月6日，今天是星期几？"))