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
    azure = AzureSettings(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""),
        embedding_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
    )
    return AppSettings(azure=azure)

