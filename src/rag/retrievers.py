from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from src.config import get_settings


def _read_docs(path: Path, source_type: str) -> List[Document]:
    with path.open() as f:
        raw = json.load(f)
    docs: List[Document] = []
    for item in raw:
        docs.append(
            Document(
                page_content=item.get("body", ""),
                metadata={
                    "title": item.get("title"),
                    "date": item.get("date"),
                    "source": item.get("source"),
                    "tags": ",".join(item.get("tags", [])),
                    "source_type": source_type,
                },
            )
        )
    return docs


def get_retriever() -> Chroma:
    settings = get_settings()
    persist_dir = Path(settings.persist_dir)
    embeddings = AzureOpenAIEmbeddings(
        model=settings.azure.embedding_deployment,
        azure_endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        api_version=settings.azure.api_version,
    )
    if persist_dir.exists():
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    # fallback: in-memory build from seed docs
    seed_docs: List[Document] = []
    seed_docs.extend(_read_docs(Path(settings.sk_corpus_path), "sk"))
    seed_docs.extend(_read_docs(Path(settings.global_corpus_path), "global"))
    return Chroma.from_documents(seed_docs, embedding=embeddings)

