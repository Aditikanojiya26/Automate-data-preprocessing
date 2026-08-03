try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None


def create_llm(model: str = "gpt-5-mini", **kwargs):
    if ChatOpenAI is None:
        raise ImportError(
            "langchain-openai is not installed. Install it to use the OpenAI helper."
        )

    return ChatOpenAI(model=model, temperature=0, **kwargs)


def build_fallback_llm(**kwargs):
    primary = create_llm(model="gpt-5-mini", **kwargs)
    fallback1 = create_llm(model="gpt-5.5-mini", **kwargs)
    fallback2 = create_llm(model="gpt-5.5-nano", **kwargs)

    return primary.with_fallbacks([fallback1, fallback2])