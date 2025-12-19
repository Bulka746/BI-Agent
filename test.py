#!/usr/bin/env python
# coding: utf-8



from langchain_openai import ChatOpenAI
import json
import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from trino import dbapi
from trino.exceptions import TrinoUserError
from trino.exceptions import TrinoExternalError
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os
import io
from datetime import timedelta

"""
Additional imports for integration with MinIO

This agent now supports logging interactions and generated artefacts to a
MinIO-compatible object storage. The following imports enable connection
and file uploads to MinIO. If the `minio` library is not installed, the
logging functions will gracefully degrade and simply print a message.

The environment variables `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY` and `MINIO_BUCKET_NAME` can be used to configure the
MinIO client. Defaults are provided for local development.

```
export MINIO_ENDPOINT="localhost:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin"
export MINIO_BUCKET_NAME="bi-agent-logs"
```

A unique `chat_id` and human‑readable `chat_name` are generated when the
agent is started. A step counter tracks the sequence of messages for
logging purposes.
"""

try:
    from minio import Minio
    from minio.error import S3Error 
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

if MINIO_AVAILABLE:
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "bi-agent-logs")
    minio_client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
    except Exception as bucket_err:
        print("Warning: unable to verify or create MinIO bucket:", bucket_err)
else:
    minio_client = None
    MINIO_BUCKET = None

def log_to_minio(state: dict, type_of_content: str, content: str) -> str | None:
    """
    Persist a single interaction step to MinIO.

    Each call increments the `step_num` stored in `state` and writes a JSON
    record describing the event. The JSON object includes:

      - chat_id: Unique identifier for the entire conversation
      - chat_name: Human‑readable name for the chat
      - type_of_content: One of "initial_prompt", "clarification_questions",
        "clarification_response", "analytics", "image_link"
      - content: Free form text or URL, depending on the type
      - step_num: Monotonically increasing counter within this chat

    When the MinIO client is unavailable, the function prints a notice and
    returns `None`.

    Parameters
    ----------
    state : dict
        The current agent state. Must contain `chat_id`, `chat_name`, and
        `step_num` keys. `step_num` will be incremented in place.
    type_of_content : str
        Categorical label describing the payload being logged.
    content : str
        Arbitrary text to log (for image files this should be a URL).

    Returns
    -------
    str | None
        The object name used in MinIO if logging succeeded, otherwise None.
    """
    if not MINIO_AVAILABLE or minio_client is None or MINIO_BUCKET is None:
        print("MinIO client unavailable; skipping persistence for", type_of_content)
        state["step_num"] = state.get("step_num", 0) + 1
        return None
    step_num = state.get("step_num", 0) + 1
    state["step_num"] = step_num
    chat_id = state.get("chat_id")
    chat_name = state.get("chat_name")
    record = {
        "chat_id": chat_id,
        "chat_name": chat_name,
        "type_of_content": type_of_content,
        "content": content,
        "step_num": step_num,
    }
    object_name = f"{chat_id}/{step_num}_{type_of_content}.json"
    data = json.dumps(record, ensure_ascii=False).encode("utf-8")
    try:
        minio_client.put_object(
            MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="application/json",
        )
    except Exception as upload_err:
        print("Warning: failed to log to MinIO:", upload_err)
        return None
    return object_name

def save_figure_to_minio(state: dict) -> str | None:
    """
    Save the current matplotlib figure into MinIO and return a public link.

    This function captures the active Matplotlib figure, writes it to an
    in-memory buffer as PNG and uploads it to MinIO. A pre‑signed URL is
    generated with a default expiry (7 days). If generation of the URL
    fails, a fallback path composed of endpoint, bucket and object name is
    returned.

    Parameters
    ----------
    state : dict
        The current agent state containing chat identifiers. `step_num` will
        be incremented within `log_to_minio` when called afterwards.

    Returns
    -------
    str | None
        A URL pointing to the uploaded image if successful, else None.
    """
    if not MINIO_AVAILABLE or minio_client is None or MINIO_BUCKET is None:
        print("MinIO client unavailable; skipping image upload")
        return None
    chat_id = state.get("chat_id")

    next_step = state.get("step_num", 0) + 1
    object_name = f"{chat_id}/{next_step}_viz.png"

    buf = io.BytesIO()
    try:
        fig = plt.gcf()
    except Exception:

        fig = None
    if fig is not None:
        fig.savefig(buf, format="png")
    else:

        plt.savefig(buf, format="png")
    buf.seek(0)
    try:
        minio_client.put_object(
            MINIO_BUCKET,
            object_name,
            buf,
            buf.getbuffer().nbytes,
            content_type="image/png",
        )
    except Exception as upload_err:
        print("Warning: failed to upload image to MinIO:", upload_err)
        return None

    try:
        url = minio_client.presigned_get_object(
            MINIO_BUCKET, object_name, expires=timedelta(days=7)
        )
    except Exception:

        proto = "https" if minio_client._secure else "http"
        url = f"{proto}://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"
    return url




SYSTEM_PROMPT = """
Ты аналитический модуль внутри автоматизированного пайплайна анализа данных.
Твоя задача — строго и последовательно выполнять аналитические функции,
в зависимости от этапа. Твоя основная задача строить аналитику для бизнеса.

Модель поведения:
- Уточняй данные на естественном языке;
- Ты являешься аналитическим помошником для менеджмента.

Не выходи за рамки ответственности текущего узла.

Контекст среды:
- Источник данных — SQL через Trino,
- Схема базы данных подаётся явно,
- Визуализация выполняется через matplotlib.

Правила:
- При уточнении данных, учитывай что ты разговариваешь с бизнесом: не используй технические тармины, название столбцов, таблиц, схем и sql-термины;
- Перед написание SQL-запроса подумай, как можно сделать его максимально точным и подробным;
- Используй только таблицы/поля полученные их схемы;
- Не придумывай данные, только фичи к данным, если считаешь нужным;
- Не делай бизнес-выводов без запроса и данных;
- При построение SQL-запроса, помни о будущей визуализации;
- Возвращай только структурированный результат согласно контракту узла;
- При написании запроса к названию таблицы добавляй iceberg.gold. => iceberg.gold.table_name.
""" 



from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-pN7mbpwi3acKv4tu4iG8Uw",
    base_url="https://api.artemox.com/v1",  # или свой прокси / gateway
    model="gpt-5.1",
    temperature=0.1,
    max_tokens = 1000
)



class AgentState(TypedDict):
    user_input: str
    merged_input: str | None

    schema: dict | None 

    intent: str | None  
    clarification_required: bool | None
    questions: list[str] | None

    sql_query: str | None
    query_result: object | None
    analytics: str | None
    viz_code: str | None
    sql_error: str | None
    sql_fix_attempts: int

    chat_id: str
    chat_name: str
    step_num: int




def schema_to_text(schema: dict | None) -> str:
    if not schema:
        return "Схема данных недоступна."

    lines = []
    for table, columns in schema.get("tables", {}).items():
        lines.append(f"Таблица {table}:")
        for col, meta in columns.items():
            col_type = meta.get("type")
            comment = meta.get("comment")
            if comment:
                lines.append(f"- {col} ({col_type}): {comment}")
            else:
                lines.append(f"- {col} ({col_type})")
        lines.append("")  # пустая строка между таблицами

    return "\n".join(lines)

def safe_json_loads(text: str) -> dict:
    """
    Безопасно извлекает JSON-объект из ответа LLM.
    Если JSON не найден — выбрасывает ValueError с понятным сообщением.
    """
    if not text or not text.strip():
        raise ValueError("LLM returned empty response")

    # Ищем первый JSON-объект вида {...}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in LLM response:\n{text}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in LLM response:\n{match.group()}"
        ) from e










import json

from trino import dbapi

def schema_introspection_node(state: AgentState) -> dict:
    print(">>> ENTERED")

    conn = dbapi.connect(
        host="localhost",
        port=8081,
        user="trino_user",
        catalog="iceberg",
        schema="gold",
    )

    cursor = conn.cursor()

    # 1️⃣ список таблиц
    cursor.execute("SHOW TABLES FROM iceberg.gold")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {
        "layer": "gold",
        "catalog": "iceberg",
        "tables": {}
    }

    # 2️⃣ DESCRIBE + comments
    for table in tables:
        cursor.execute(f"DESCRIBE iceberg.gold.{table}")
        rows = cursor.fetchall()

        columns = {}

        for row in rows:
            column_name = row[0]
            data_type = row[1]
            comment = row[3] if len(row) > 3 else None

            # отсекаем служебные строки
            if not column_name or column_name.startswith("#"):
                continue

            columns[column_name] = {
                "type": data_type,
                "comment": comment
            }

        schema["tables"][table] = columns

    cursor.close()
    conn.close()


    return {
        "schema": schema
    }


#Узел получения запроса и данных
def intent_node(state: AgentState) -> dict:
    text = state.get("merged_input") or state["user_input"]

    schema = state.get("schema")

    schema_text = schema_to_text(schema)

    prompt = f"""
STAGE: INTENT
Доступные данные:
{schema_text}

USER_QUERY:
{text}

Проверь, хватает ли данных для генерации SQL.

Минимально требуется:
- Аналитическая цель
- Бизнес-метрики
- Аналитическая логика запроса

Верни JSON строго в формате:
{{
  "clarification_required": boolean,
  "questions": [string]
}}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    parsed = json.loads(response.content)

    print("\n=== INTENT LLM ===")
    print(response.content)


    if parsed.get("clarification_required") and parsed.get("questions"):
        questions_text = "\n".join(parsed["questions"])
        log_to_minio(state, "clarification_questions", questions_text)

    return {
        "clarification_required": parsed["clarification_required"],
        "questions": parsed.get("questions"),
    }

#Узел маршрута, по условию хватает ли данных для запроса
def route_after_intent(state: AgentState) -> str:
    """
    Маршрутизация (НЕ LLM):
    - если нужны уточнения -> clarification_node
    - иначе -> sql_exec_node
    """
    if state.get("clarification_required"):
        return "clarification_node"
    return "sql_generation_node"

#Цикл повтора
def clarification_node(state: AgentState) -> dict:
    """
    Одна нода для уточнения:
    - показывает вопросы
    - принимает один ввод пользователя
    - добавляет его к merged_input
    """
    print("\n❓ Не хватает данных. Уточни, пожалуйста:")

    for q in (state.get("questions") or []):
        print(f"- {q}")

    clarification = input("\nВведите уточнение одним сообщением:\n> ").strip()

    base = state.get("merged_input") or state["user_input"]
    merged = f"{base}\n\nУТОЧНЕНИЕ:\n{clarification}"

    log_to_minio(state, "clarification_response", clarification)

    return {
        "merged_input": merged
    }

# def sql_planning_node(state: AgentState) -> dict:
#     """
#     Пока заглушка.
#     Тут ты дальше сделаешь генерацию SQL под Trino на основе merged_input + схемы.
#     """
#     final_context = state.get("merged_input") or state["user_input"]

#     print("\n✅ Данных достаточно. Переходим к SQL_PLANNING.")
#     print("\n=== FINAL CONTEXT ===")
#     print(final_context)

#     # заглушка
#     return {}

#Генерация sql
def sql_generation_node(state: AgentState) -> dict:
    """
    Генерирует SQL-запрос (Trino SQL) на основе финального контекста.
    """
    final_context = state.get("merged_input") or state["user_input"]
    schema = state.get("schema")

    schema_text = schema_to_text(schema)
    print("\n✅ Данных достаточно. Переходим к SQL_PLANNING.")
    print("\n=== FINAL CONTEXT ===")
    print(final_context)

    final_text = state.get("merged_input") or state["user_input"]

    prompt = f"""
STAGE: SQL_GENERATION

Доступные данные:
{schema_text}

USER_REQUEST:
{final_text}

Требования:
- Сгенерируй SQL-запрос для Trino
- Округляй все числа до двух точек после запятой
- Используй только те таблицы и поля, которые указаны пользователем
- Если период указан текстом (например, "весь 2021 год") — корректно преобразуй в фильтр по датам
- Если метрика указана — используй её
- Если группировка указана — добавь GROUP BY
- Если чего-то не хватает, сделай разумное допущение, но НЕ задавай вопросов

Верни результат СТРОГО в JSON формате:
{{
  "sql_query": "string"
}}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    print("\n=== SQL LLM ===")
    print(response.content)

    parsed = safe_json_loads(response.content)

    return {
        "sql_query": parsed["sql_query"]
    }

from trino import dbapi


def sql_exec_node(state: AgentState) -> dict:
    sql = state.get("sql_query")
    if not sql:
        return {
            "sql_error": "sql_exec_node: sql_query is empty"
        }

    print("\n🚀 EXECUTING SQL IN TRINO")
    print(sql)

    try:
        conn = dbapi.connect(
            host="localhost",
            port=8081,
            user="trino_user",
            catalog="iceberg",
            schema="gold",
        )

        cursor = conn.cursor()
        cursor.execute(sql)

        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

        cursor.close()
        conn.close()

        return {
            "query_result": {
                "columns": columns,
                "rows": rows
            },
            "sql_error": None   # 🔥 ВАЖНО: явно сбрасываем ошибку
        }

    except TrinoUserError as e:
        # ❗ SQL ошибка (синтаксис, поля, GROUP BY и т.д.)
        error_text = f"TrinoUserError: {e}"

        print("\n❌ TRINO USER ERROR")
        print(error_text)

        return {
            "sql_error": error_text,
            "query_result": None
        }

    except TrinoExternalError as e:
        # ❗ Ошибка движка / кластера
        error_text = f"TrinoExternalError: {e}"

        print("\n❌ TRINO EXTERNAL ERROR")
        print(error_text)

        return {
            "sql_error": error_text,
            "query_result": None
        }

    except Exception as e:
        # ❗ Любая другая ошибка (network, Python, etc.)
        error_text = f"Unexpected error in sql_exec_node: {e}"

        print("\n❌ UNEXPECTED ERROR")
        print(error_text)

        return {
            "sql_error": error_text,
            "query_result": None
        }

def route_after_sql_exec(state: AgentState) -> str:
    sql_error = state.get("sql_error")
    attempts = state.get("sql_fix_attempts", 0)

    print("DEBUG ROUTER")
    print("sql_error repr:", repr(state.get("sql_error")))
    print("attempts:", state.get("sql_fix_attempts"))
    # ❌ Есть реальная ошибка
    if sql_error is not None:
        if attempts >= 3:
            print("\n⛔ SQL fix attempts limit reached")
            return END
        return "sql_error_fix_node"

    # ✅ Ошибки нет → идём дальше
    return "viz_planning_node"


def sql_error_fix_node(state: AgentState) -> dict:
    """
    Исправляет SQL-запрос на основе ошибки Trino.
    Максимум 3 попытки.
    """

    sql_query = state.get("sql_query")
    sql_error = state.get("sql_error")
    attempts = state.get("sql_fix_attempts", 0)

    if not sql_query or not sql_error:
        raise ValueError("sql_error_fix_node: missing sql_query or sql_error")


    user_text = state.get("merged_input") or state["user_input"]
    schema = state.get("schema")
    schema_text = schema_to_text(schema)

    prompt = f"""
STAGE: SQL_ERROR_FIX

ТРЕБОВАНИЯ ПОЛЬЗОВАТЕЛЯ:
{user_text}

СХЕМА ДАННЫХ:
{schema_text}

ОРИГИНАЛЬНЫЙ SQL:
{sql_query}

ОШИБКА TRINO:
{sql_error}


ЗАДАЧА:
- Проанализируй ошибку Trino
- Исправь SQL-запрос
- НЕ меняй бизнес-логику запроса
- Верни корректный SQL для Trino

Верни результат СТРОГО в JSON формате:
{{
  "sql_query": "string"
}}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    print("\n=== SQL ERROR FIX LLM RAW ===")
    print(response.content)

    parsed = safe_json_loads(response.content)

    return {
        "sql_query": parsed["sql_query"],
        "sql_error": None,                 # сбрасываем ошибку
        "sql_fix_attempts": attempts + 1,  # увеличиваем счётчик
    }



def viz_planning_node(state: AgentState) -> dict:
    qr = state.get("query_result")

    if not qr:
        raise ValueError("viz_planning_node: query_result is empty")

    columns = qr["columns"]
    rows = qr["rows"][:10]  # показываем LLM только пример

    prompt = f"""
STAGE: VISUALIZATION_PLANNING

ДАННЫЕ:
Колонки: {columns}
Пример строк:
{rows}

Твоя задача:
1. Определи наиболее подходящий тип графика
2. Напиши Python-код для matplotlib
3. Сформулируй краткий аналитический вывод по данным

Ограничения:
- код будет исполняться в окружении, где уже есть:
  - pandas as pd
  - matplotlib.pyplot as plt
  - DataFrame df (создан заранее)
- НЕ импортируй os, sys, subprocess
- НЕ читай и не пиши файлы
- используй ТОЛЬКО df

Верни результат СТРОГО в JSON формате:
{{
  "viz_code": "string",
  "analytics": "string"
}}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    print("\n=== VIZ LLM RAW ===")
    print(response.content)

    parsed = safe_json_loads(response.content)


    if parsed.get("analytics"):
        log_to_minio(state, "analytics", parsed["analytics"])

    return {
        "viz_code": parsed["viz_code"],
        "analytics": parsed["analytics"],
    }

def viz_exec_node(state: AgentState) -> dict:

    qr = state.get("query_result")
    code = state.get("viz_code")

    if not qr or not code:
        raise ValueError("viz_exec_node: missing query_result or viz_code")

    # создаём DataFrame
    df = pd.DataFrame(qr["rows"], columns=qr["columns"])

    print("\n=== EXECUTING VIZ CODE ===")
    print(code)

    # безопасное окружение выполнения
    exec_globals = {
        "pd": pd,
        "plt": plt,
        "df": df,
    }


    original_show = plt.show
    def _noop_show(*args, **kwargs):
        plt.draw()
    plt.show = _noop_show

    try:
        exec(code, exec_globals)
    finally:
        plt.show = original_show


    try:
        plt.draw()
    except Exception:
        pass

    image_url = save_figure_to_minio(state)
    if image_url:
        log_to_minio(state, "image_link", image_url)

    try:
        original_show()
    except Exception:
        pass

    return {}






graph = StateGraph(AgentState)

graph.add_node("schema_introspection_node", schema_introspection_node)
graph.add_node("intent_node", intent_node)
graph.add_node("clarification_node", clarification_node)
graph.add_node("sql_generation_node", sql_generation_node)
graph.add_node("sql_exec_node", sql_exec_node)
graph.add_node("sql_error_fix_node", sql_error_fix_node)
graph.add_node("viz_planning_node", viz_planning_node)
graph.add_node("viz_exec_node", viz_exec_node)

# строим последовательность выполнения
graph.add_edge(START, "schema_introspection_node")
graph.add_edge("schema_introspection_node","intent_node")
graph.add_conditional_edges(
    "intent_node",
    route_after_intent,
    {
        "clarification_node": "clarification_node",
        "sql_generation_node": "sql_generation_node",
    }
)
# цикл уточнений
graph.add_edge("clarification_node", "intent_node")
graph.add_edge("sql_generation_node", "sql_exec_node")
graph.add_conditional_edges(
    "sql_exec_node", 
    route_after_sql_exec,
    {
        "sql_error_fix_node": "sql_error_fix_node",
        "viz_planning_node": "viz_planning_node",
    }

)
graph.add_edge("sql_error_fix_node", "sql_exec_node")
graph.add_edge("viz_planning_node", "viz_exec_node")
graph.add_edge("viz_exec_node", END)



compiled = graph.compile()




if __name__ == "__main__":
    user_query = input("Введите запрос:")


    chat_id = str(uuid.uuid4())
    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    chat_name = f"chat_{timestamp_str}"

    initial_state: AgentState = {
        "user_input": user_query,
        "schema": None,
        "merged_input": None,

        "clarification_required": None,
        "questions": None,

        "sql_query": None,

        "query_result": None,
        "analytics": None,
        "viz_code": None,
        "sql_error": None,
        "sql_fix_attempts": 0,

        "chat_id": chat_id,
        "chat_name": chat_name,
        "step_num": 0,
    }

    print("USER INPUT:", initial_state["user_input"])

    log_to_minio(initial_state, "initial_prompt", user_query)

    result_state = compiled.invoke(initial_state)

    print("\n=== FINAL STATE ===")
    for k, v in result_state.items():
        print(f"{k}: {v}")






