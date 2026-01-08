from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import get_settings
from src.rag.embeddings import get_embeddings


def _read_docs(path: Path, source_type: str) -> List[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: List[Document] = []
    for record in data:
        body = record.get("body", "")
        if not body:
            continue
        metadata = {
            "title": record.get("title"),
            "date": record.get("date"),
            "source": record.get("source"),
            "tags": ",".join(record.get("tags", [])),
            "source_type": source_type,
        }
        docs.append(Document(page_content=body, metadata=metadata))
    return docs


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def get_retriever() -> Chroma:
    raise NotImplementedError("Chroma retriever deprecated; use get_faiss_index instead.")


@lru_cache
def get_faiss_index(source_type: str) -> FAISS:
    settings = get_settings()
    embeddings = get_embeddings()
    index_dir = Path(settings.persist_dir) / source_type
    if index_dir.exists():
        return FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    # fallback in-memory
    docs = _read_docs(
        Path(settings.sk_corpus_path if source_type == "sk" else settings.global_corpus_path),
        source_type,
    )
    return FAISS.from_documents(_split_docs(docs), embedding=embeddings)

