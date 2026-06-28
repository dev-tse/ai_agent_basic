from groq import Groq
from dotenv import load_dotenv
import os
import json
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conn = sqlite3.connect('document_store.db', check_same_thread=False)
cursor = conn.cursor()
model = SentenceTransformer('all-MiniLM-L6-v2')

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chunks TEXT,
               filename TEXT
               )
""")



def get_weather(city: str) -> str:
    fake_weather = {
        "Алматы": "+18°C, облачно",
        "Астана": "+12°C, ветрено"
    }
    return fake_weather.get(city, "Нет данных по городу")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vector_search(query, knowledge_base,top_k=9):
    vector_query = model.encode(query)
    results = []
    for i, chunk in enumerate(knowledge_base):
        vector_chunk = model.encode(chunk)
        score = cosine_similarity(vector_query, vector_chunk)
        results.append((score, chunk))
    results = sorted(results, key=lambda x:x[0], reverse=True)
    results = [chunk for score, chunk in results]
    return "\n".join(results[:top_k])


# Храним документ в памяти
document_store = {
    "text": None,
    "chunks": [],
    "filename": None
}


def search_documents(query):
    rows = cursor.execute("SELECT * FROM documents").fetchall()
    chunks = json.loads(rows[0][1])
    result = vector_search(query, chunks)
    return result


available_functions = {
    "get_weather": get_weather,
    "search_documents": search_documents
}

tools = [
    {
        "type":"function",
        "function": {
            "name": "get_weather",
            "description": "Выдает инфо погоды по городам",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Город на который нужна погода"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Выдает информацию на заданный вопрос, отвечай только на основе того, что вернул инструмент, не добавляй ничего от себя",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "вопрос на который нужен ответ"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

messages = [{"role": "user", "content":"Какая погода в городе Алматы"}]
def get_model_answer():
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

message = get_model_answer()
if message.tool_calls == None:
    print(message)
else:
    while message.tool_calls:
        func_name = message.tool_calls[0].function.name
        func_argument = json.loads(message.tool_calls[0].function.arguments)
        real_func = available_functions[func_name]
        # message_assist = {"role": "assistant", "content": message}
        result = real_func(**func_argument)
        message_user = {"role": "tool", "content": result, "tool_call_id": message.tool_calls[0].id}
        messages.append(message)
        messages.append(message_user)
        message = get_model_answer()
        print(message)

