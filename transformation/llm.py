import os
from typing import Any


def build_llm() -> Any:
    """Return an LLM instance based on environment configuration.

    Defaults to OpenAI models. Set ``LLM_PROVIDER`` to ``"google"`` or
    ``"ollama"`` to use Gemini or an Ollama-backed model instead. For OpenAI,
    the API key is read from ``OPENAI_API_KEY``. For Gemini, the API key is
    read from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``. For Ollama,
    ``OLLAMA_HOST`` or ``OLLAMA_HOST_PC`` must be set.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        from langchain_ollama.llms import OllamaLLM

        host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_HOST_PC")
        if not host:
            raise EnvironmentError("Set OLLAMA_HOST or OLLAMA_HOST_PC")
        model_name = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")
        return OllamaLLM(
            model=model_name,
            base_url=host,
            options={"num_ctx": 8192},
            temperature=0.0,
        )

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAI

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return GoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.0)

    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.0)
