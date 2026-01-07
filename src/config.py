import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

# .env 자동 로드 (루트에 .env가 있을 경우)
load_dotenv()


@dataclass
class AzureSettings:
    endpoint: str
    api_key: str
    chat_deployment: str
    embedding_deployment: str
    api_version: str = "2024-05-01-preview"


@dataclass
class AppSettings:
    azure: AzureSettings
    persist_dir: str = "data/chroma"
    sk_corpus_path: str = "data/sk_news.json"
    global_corpus_path: str = "data/global_trends.json"


@lru_cache
def get_settings() -> AppSettings:
    def _env(*names: str, default: str = "") -> str:
        for name in names:
            value = os.getenv(name)
            if value:
                return value.rstrip("/") if "ENDPOINT" in name else value
        return default

    azure = AzureSettings(
        # 우선순위: AZURE_* > AOAI_*
        endpoint=_env("AZURE_OPENAI_ENDPOINT", "AOAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_API_KEY", "AOAI_API_KEY"),
        chat_deployment=_env(
            "AZURE_OPENAI_CHAT_DEPLOYMENT",
            "AOAI_DEPLOY_GPT4O_MINI",
            "AOAI_DEPLOY_GPT4O",
        ),
        embedding_deployment=_env(
            "AZURE_OPENAI_EMBED_DEPLOYMENT",
            "AOAI_DEPLOY_EMBED_3_SMALL",
            "AOAI_DEPLOY_EMBED_3_LARGE",
            "AOAI_DEPLOY_EMBED_ADA",
        ),
        api_version=_env("AZURE_OPENAI_API_VERSION", default="2024-05-01-preview"),
    )
    return AppSettings(azure=azure)

