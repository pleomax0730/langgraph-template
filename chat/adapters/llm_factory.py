from langchain_core.language_models.chat_models import BaseChatModel

from ..config import settings


def build_llm(provider: str, model_name: str) -> BaseChatModel:
    if provider.lower() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name, streaming=True
        )  # vertex ai 要調整 init 方式

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model_name=model_name,
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "high", "summary": "detailed"},
        streaming=True,
    )


MODEL = build_llm(settings.MODEL_PROVIDER, settings.MODEL_NAME)
