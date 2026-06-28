# AI Agent Basic — Tool Calling & Agentic Loop

Минимальный, но полноценный AI-агент на Python с использованием Groq API. Демонстрирует ключевые концепции agentic AI: tool calling, agentic loop, RAG как инструмент агента.

## Что внутри

- **Agentic loop** — модель сама решает, нужен ли инструмент, и может вызывать их несколько раз подряд до получения финального ответа
- **Два инструмента:**
  - `get_weather(city)` — демо-инструмент с фиктивными данными
  - `search_documents(query)` — RAG-поиск по загруженным документам (SQLite + sentence-transformers + cosine similarity)
- **Защита от галлюцинаций** — инструкция в `description` инструмента, требующая отвечать только на основе найденного контекста

## Технологии

- Python 3.9
- Groq API (LLaMA 3.3 70B / GPT-OSS 120B)
- SQLite — хранение документов
- sentence-transformers — векторные embeddings (`all-MiniLM-L6-v2`)
- NumPy — cosine similarity

## Архитектура

\```
Вопрос пользователя
    → Модель решает: нужен инструмент?
        → Да: вызывается реальная Python-функция
            → Результат возвращается модели (role: "tool")
            → Модель решает: достаточно информации?
        → Нет: модель отвечает напрямую
\```

## Известные ограничения (и как решены)

- **Нестабильность tool calling** на некоторых моделях — решено переходом с `llama-3.3-70b-versatile` (deprecated) на `openai/gpt-oss-120b`
- **Качество векторного поиска** зависит от размера чанков и `top_k` — для большей точности увеличен `top_k` до 9
- **Галлюцинации** — модель могла дополнять найденную информацию своими общими знаниями; решено явной инструкцией в `description` инструмента

## Запуск

\```bash
python3 -m venv venv
source venv/bin/activate
pip install groq python-dotenv sentence-transformers numpy

# создать .env с GROQ_API_KEY=...

python agent_test.py
\```
