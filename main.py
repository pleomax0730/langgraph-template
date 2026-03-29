import uvicorn
from fastapi import FastAPI

from chat.router import router as chat_router

app = FastAPI(
    title="LangGraph Chat API Service",
    description="LangGraph Stream API",
    version="1.0.0",
)

# Register Endpoints
app.include_router(chat_router)


@app.get("/")
def health_check():
    return {"status": "up", "message": "Ready to chat!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
