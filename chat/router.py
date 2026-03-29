import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .port import ChatCommandRequest
from .use_cases.chat_streamer import stream_execute

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream_endpoint(request: ChatCommandRequest):
    # Adapter transforms external HTTP payload to Internal Domain types
    async def sse_event_generator() -> AsyncGenerator[str]:
        async for event_dict in stream_execute(
            user_input=request.user_input,
            chat_mode=request.chat_mode,
            chat_history_dicts=request.chat_history,
        ):
            # SSE Standard format compliance
            yield f"data: {json.dumps(event_dict, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
