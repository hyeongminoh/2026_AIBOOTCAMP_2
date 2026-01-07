from __future__ import annotations

import os
from typing import Any

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.config import get_settings


def get_embeddings() -> Any:
    """
    Azure AOAI(기본) 또는 OpenAI 퍼블릭 API를 선택적으로 사용.
    USE_OPENAI_EMBEDDINGS=true 이면 OpenAIEmbeddings, 아니면 AzureOpenAIEmbeddings.
    """
    settings = get_settings()
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "").lower() in {"1", "true", "yes"}

    if use_openai:
        return OpenAIEmbeddings(
            model=settings.azure.embedding_deployment or "text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY", settings.azure.api_key),
        )

    return AzureOpenAIEmbeddings(
        model=settings.azure.embedding_deployment,
        azure_endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        api_version=settings.azure.api_version,
    )

