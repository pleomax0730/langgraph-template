# LangGraph Streaming API Service

這是一個企業級支援非同步 LangGraph 串流基礎設施。本專案透過 `FastAPI` 與 `Server-Sent Events (SSE)` 建構出高效擴展的 API 層，並針對大型專案嚴格貫徹了 Clean Architecture (六角形架構) 規範。

## 架構核心理念

* **原生非同步高併發 (Native Async I/O)**: 在底層全面採用原生 `asyncio` 與 `FastAPI` Generator，取代傳統阻塞式的執行緒佇列 (Thread Blocking Queues)，確保伺服器單機承載高併發請求。同步且易發生 I/O 阻塞的舊型工具，均會受到內部 `asyncio.to_thread` 執行緒池保護，防止主動線程雪崩。
* **多型語言模型配接器 (Polymorphic LLM Adapter)**: 導入模型工廠與依賴注入 (DI) 設計，使得替換底層 `OpenAI` 與 `Google Gemini` 服務不需動及核心業務邏輯。
* **運算開銷最佳化 (O(1) Streaming Interceptors)**: 面對長推理 (Reasoning Blocks) 串流產生的資源消耗，於 `StreamEventHandler` 中引進了啟發式短路及即時緩衝區清理策略，消除了正則運算式的複雜度效能瓶頸。
* **深度解耦的依賴注入 (Dependency Injection)**: 所有的應用層用例 (Use Cases) 皆經由 FastAPI `Depends` 動態掛載，與 HTTP Router 徹底分離。開發團隊可隨時重寫底層 Mock 服務以達成高效單元測試。

---

## 目錄結構與模組職責

整個 `chat` 模組透過「關注點分離」嚴格切分：

* `port.py`: 定義了嚴格的 Pydantic 驗證綱要 (Schemas)，構成系統對外的資料交換合約 (Data Transfer Objects)。
* `domain.py`: 存放單純負責運算或狀態管理的不可變資料結構 (`frozen dataclass`)，確保隔離外部框架的侵入。
* `config.py`: 配置層。
* `adapters/`: 基礎設施配接器 (Infrastructure Adapters)，封裝所有對外系統之溝通細節。
  * `llm_factory.py`: 動態決定及實體化第三方模型連接物件。
  * `tools_registry.py`: 外部工具定義中樞與非同步安全執行器。
  * `graph_nodes.py`: LangGraph 狀態機核心節點。
  * `graph_builder.py`: 負責將所有節點編譯為執行圖 (StateGraph)。
* `use_cases/chat_streamer.py`: 應用場景層 (Application Services)。主要負責將內部複雜 Graph Data 轉譯為前端所需之標準格式。
* `dependencies.py`: FastAPI 的依賴提供者模組 (DI Provider)，管理 Use Case 實例化。
* `router.py`: 控制器層，純粹負責 HTTP 路由分配。

---

## 環境配置與服務啟動

### 1. 套件安裝
請使用 `uv` 或是您偏好的套件管理器進行安裝：
```bash
uv sync
```

### 2. 環境變數設定
請於專案根目錄建立 `.env` 檔案，支援以下標準設定檔：
```env
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..." # Vertex AI 整合需額外修改 Adapter API 初始化

# 選擇服務供應商 (openai / google)
MODEL_PROVIDER="openai" 

MODEL_NAME="gpt-5.4-mini" 
```

### 3. 本機服務啟動
透過 `uvicorn` 啟動：
```bash
uv run uvicorn main:app --reload
```
預設服務端點位址為 `http://127.0.0.1:8000`。

---

## API 介接文件與整合範例

### 範例 A：無狀態單次請求 (Single Turn Execution)
針對預設的獨立單機測試，可直接以命令列測試 SSE (Server-Sent Events) 輸出端點：

```bash
curl -N -s -X POST http://127.0.0.1:8000/chat/stream \
     -H "Content-Type: application/json" \
     -d '{
           "user_input": "請問原價120元打85折是幾元？從台北用 express 寄下來要多久時間？並幫我總結一下。",
           "chat_mode": "plan"
         }'
```

**預期回傳的 Event Stream 結構：**
* `{"event": "thought", ...}`
* `{"event": "tool_progress", ...}`
* `{"event": "llm_progress", ...}`
* `{"event": "final_response", ...}`

---

### 範例 B：具備歷史記憶的連續請求 (Stateful Multi-turn Execution)

為了支援生產環境下之多輪對話 (如 ChatGPT)，若對話歷史涉及曾經執行過之「工具請求 (Tool Call)」與對應的「工具回傳結果 (Tool Result)」，前端客戶端需於 Payload 中提供包含 `tool_calls` 與 `role: "tool"` 格式的歷史紀錄陣列，以便 LLM 繼承先前的邏輯情境。

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
               "content": "",
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
以此格式發送，伺服器即可完美解析上游之變數狀態 (`102.0`) ，接續回答客戶需求。

---

### 範例 C：非串流同步請求 (JSON Synchronous Execution)

若不需要打字機效果，可使用 `/chat/invoke` 端點直接獲取最終結果：

```bash
curl -s -X POST http://127.0.0.1:8000/chat/invoke \
     -H "Content-Type: application/json" \
     -d '{
           "user_input": "120元打85折是多少？",
           "chat_mode": "plan"
         }'
```

**預期回傳：**
```json
{
  "response": "120元打85折後的價格是 102元。",
  "usage": {
    "input_tokens": 120,
    "output_tokens": 45
  }
}
```

