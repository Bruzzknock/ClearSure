import os
from typing import Any


class _ChatCompletionWrapper:
    """Minimal interface over the OpenAI client exposing ``invoke``."""

    def __init__(self, client, model: str, *, temperature: float = 0.0, max_tokens: int = 1000) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content


def build_llm() -> Any:
    """Return an LLM instance based on environment configuration.

    Defaults to OpenAI models. Set ``LLM_PROVIDER`` to ``"google"`` or
    ``"ollama"`` to use Gemini or an Ollama-backed model instead. For OpenAI,
    the API key is read from ``OPENAI_API_KEY`` and the endpoint from
    ``OPENAI_ENDPOINT`` (or ``AZURE_OPENAI_ENDPOINT``); ``OPENAI_API_VERSION``
    may also be set. For Gemini, the API key is read from ``GEMINI_API_KEY``
    or ``GOOGLE_API_KEY``. For Ollama, ``OLLAMA_HOST`` or ``OLLAMA_HOST_PC``
    must be set.
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

    from openai import AzureOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY")
    endpoint = os.environ.get("OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise EnvironmentError("Set OPENAI_ENDPOINT or AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return _ChatCompletionWrapper(client, model_name, temperature=0.0)
