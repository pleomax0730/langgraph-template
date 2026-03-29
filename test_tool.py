import httpx

payload = {
    "user_input": "上面的打折算出來是102元對吧？好，那我現在還需要加上高雄 express 的運費。",  # noqa: E501
    "chat_mode": "plan",
    "chat_history": [
        {"role": "user", "content": "請問原價120元打85折是幾元？"},
        {
            "role": "assistant",
            "content": "我幫您計算一下...",
            "tool_calls": [
                {
                    "name": "calculate_sale_price",
                    "args": {"original_price": 120, "discount_percent": 15},
                    "id": "call_123456",
                }
            ],
        },
        {
            "role": "tool",
            "name": "calculate_sale_price",
            "tool_call_id": "call_123456",
            "content": '{"original_price": 120, "discount_percent": 15, "discount_amount": 18.0, "final_price": 102.0, "currency": "USD"}',  # noqa: E501
        },
    ],
}

res = httpx.post("http://127.0.0.1:8000/chat/stream", json=payload, timeout=30.0)
for line in res.iter_lines():
    if line.strip():
        print(line)
