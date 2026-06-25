from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_weather(city: str) -> str:
    fake_weather = {
        "Алматы": "+18°C, облачно",
        "Астана": "+12°C, ветрено"
    }
    return fake_weather.get(city, "Нет данных по городу")

available_functions = {
    "get_weather": get_weather
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
    }
]
messages = [{"role": "user", "content":"Какая погода в Алматы?"}]
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
)
message = response.choices[0].message
if (message.tool_calls):
    func_name = message.tool_calls[0].function.name
    func_argument = json.loads(message.tool_calls[0].function.arguments)
    real_func = available_functions[func_name]
    # message_assist = {"role": "assistant", "content": message}
    result = real_func(**func_argument)
    message_user = {"role": "tool", "content": result, "tool_call_id": message.tool_calls[0].id}
    messages.append(message)
    messages.append(message_user)
    new_response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools
    )
    new_message = new_response.choices[0].message
    print(new_message)