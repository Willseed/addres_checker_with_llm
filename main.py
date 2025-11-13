import os
import csv
import difflib
from dotenv import load_dotenv
from openai import OpenAI

# 替換為你要使用的 HF LLM 模型
MODEL_NAME = "openai/gpt-oss-20b"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,   # 從 .env 取得 HF_TOKEN
)

# ====== 從 CSV 載入地址資料庫 ======
address_db = []
with open("address_list.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 組成「完整地址」= 城市 + 行政區 + 路名 + 範圍
        full_address = f"{row['CITY']}{row['AREA']}{row['ROAD']}{row['SCOOP']}"
        # 組成「完整地址 — 郵遞區號」格式
        address_db.append(f"{full_address} — {row['ZIPCODE']}")


# ====== 先用 Python 取 Top-K 最接近的地址 ======
def get_top_k_candidates(user_input, db_list, k=10):
    # 用 difflib 做快速粗匹配（你之後可以改成更高級的）
    scored = difflib.get_close_matches(user_input, db_list, n=k, cutoff=0)
    return scored


# ====== 使用者輸入地址（示例） ======
USER_INPUT = "永和中正123"

# 找候選地址
top_candidates = get_top_k_candidates(USER_INPUT, address_db, k=10)

# 形成可塞給 LLM 的文字
ADDRESS_LIST_SNIPPET = "\n".join([f"{i+1}. {addr}" for i, addr in enumerate(top_candidates)])


# ====== System Prompt ======
system_prompt = """
你是一位具備 25 年台灣 GIS 與繁體中文 NLP 經驗的地址正規化專家。
請根據候選地址列表，找出最可能的正確地址。

規則：
- 輸入地址可能錯字、漏字、行政區未寫完整
- 僅能從本次給定的候選列表選出最可能的一筆
- 必須輸出完整地址 + 其前三碼郵遞區號
- 若多筆可能匹配，選擇最合適的一筆並提供理由
"""

# ====== User Prompt ======
user_prompt = f"""
候選地址列表（Top-K）：

{ADDRESS_LIST_SNIPPET}

使用者輸入地址：
{USER_INPUT}

請依照以下 JSON 格式輸出：

{{
  "matched_address": "完整地址",
  "zipcode_3": "前三碼郵遞區號"
}}
"""

# ====== 呼叫 HuggingFace Router ======
completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

# ====== 輸出結果 ======
print(completion.choices[0].message.content)