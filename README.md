# 🚀 LangGraph 即時串流 API

## 📂 目錄結構與職責

整個 `chat` 模組嚴格遵守「關注點分離 (Separation of Concerns)」:

* `port.py`: 定義了外部 Request 對內溝通的 Pydantic 嚴格格式驗證。
* `domain.py`: 存放不可變狀態 (`frozen dataclass`) 與純粹的商業截取邏輯，跟外部框架完全脫鉤。
* `config.py`: 環境變數載入，決定目前的模型供應商。
* `adapter.py`: 所有的「髒活」都在這裡。它載入 LangChain、實作 Graph Nodes、實體化 Tools、並把 Graph 生出來的晦澀資料，利用 `StreamEventHandler` 轉譯給前端。
* `router.py`: 把 Python 內部的 Async Generator 翻譯成 `text/event-stream` (SSE)。

---

## 🛠️ 如何啟動與測試

### 1. 安裝套件
確保你在虛擬環境下安裝好以下套件：
```bash
uv pip install fastapi uvicorn pydantic langchain-openai
# 若要使用 Gemini，請再額外安裝:
# uv pip install langchain-google-genai
```

### 2. 環境變數設定
請在根目錄確認存在 `.env` 檔案，支援以下變數：
```env
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..." # vertex 要另外調整程式碼

# 決定要用哪家模型？ (openai / google)
MODEL_PROVIDER="openai" 

MODEL_NAME="gpt-5.4-mini" 
```

### 3. 啟動伺服器
請退回擁有 `main.py` 的目錄，並輸入：
```bash
uv run uvicorn main:app --reload
```
看到 `Uvicorn running on http://127.0.0.1:8000` 即代表啟動成功。

### 4. 發射測試 (Curl)

**情境 A：單次回合 (Single Turn)**
打開另一個 Terminal 視窗，打入這個指令見證奇蹟：

```bash
curl -N -s -X POST http://127.0.0.1:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{
           "user_input": "請問原價120元打85折是幾元？從台北用 express 寄下來要多久時間？並幫我總結一下。",
           "chat_mode": "plan"
         }'
```

終端機的輸出：
* `{"event": "thought", ...}`
* `{"event": "tool_progress", ...}`
* `{"event": "llm_progress", ...}`
* `{"event": "final_response", ...}`

---

### 5. 多回合工具記憶 (Multi-turn Tool History)

如果要讓大模型跟 LangGraph「記得」上一回合的對話以及工具執行結果，請在 `chat_history` 傳入包含 `tool_calls` 和 `role: "tool"` 的物件：

```bash
curl -N -s -X POST http://127.0.0.1:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{
       "user_input": "上面的打折算出來是102元對吧？好，那我現在還需要加上高雄 express 的運費。",
       "chat_mode": "plan",
       "chat_history": [
           {
               "role": "user",
               "content": "請問原價120元打85折是幾元？"
           },
           {
               "role": "assistant",
               "content": "我幫您計算一下...",
               "tool_calls": [
                   {
                       "name": "calculate_sale_price",
                       "args": {"original_price": 120, "discount_percent": 15},
                       "id": "call_123456"
                   }
               ]
           },
           {
               "role": "tool",
               "name": "calculate_sale_price",
               "tool_call_id": "call_123456",
               "content": "{\"final_price\": 102.0, \"currency\": \"USD\"}"
           }
       ]
     }'
```

模型此時會完美識別上一回合的答案 (`102.0`)，並直接回答新的問題！
