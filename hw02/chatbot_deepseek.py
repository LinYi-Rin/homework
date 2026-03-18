import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("VOLC_API_KEY")
BOT_ID = os.getenv("VOLC_BOT_ID")
API_URL = f"https://ark.cn-beijing.volces.com/api/v3/bots/{BOT_ID}/chat/completions"

def chat_with_deepseek(user_input):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-r1-250124",
        "messages": [{"role": "user", "content": user_input}],
        "stream": False
    }
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"请求失败：{response.status_code}, {response.text}"

if __name__ == "__main__":
    print("DeepSeek Chatbot 已启动，输入 'quit' 退出对话")
    while True:
        user_input = input("你：")
        if user_input.lower() == "quit":
            print("对话结束")
            break
        reply = chat_with_deepseek(user_input)
        print(f"DeepSeek：{reply}")
