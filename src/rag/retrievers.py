from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import get_settings
from src.rag.embeddings import get_embeddings


def _read_docs(path: Path, source_type: str) -> List[Document]:
    def _metadata_func(record: dict, metadata: dict) -> dict:
        metadata.update(
            {
                "title": record.get("title"),
                "date": record.get("date"),
                "source": record.get("source"),
                "tags": ",".join(record.get("tags", [])),
                "source_type": source_type,
            }
        )
        return metadata

    loader = JSONLoader(
        file_path=str(path),
        jq_schema=".[]",
        content_key="body",
        metadata_func=_metadata_func,
    )
    return loader.load()


def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def get_retriever() -> Chroma:
    settings = get_settings()
    persist_dir = Path(settings.persist_dir)
    embeddings = get_embeddings()
    if persist_dir.exists():
        return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    # fallback: in-memory build from seed docs
    seed_docs: List[Document] = []
    seed_docs.extend(_read_docs(Path(settings.sk_corpus_path), "sk"))
    seed_docs.extend(_read_docs(Path(settings.global_corpus_path), "global"))
    return Chroma.from_documents(_split_docs(seed_docs), embedding=embeddings)

