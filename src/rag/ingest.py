from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import get_settings


def load_json_corpus(path: Path, source_type: str) -> List[Document]:
    with path.open() as f:
        raw = json.load(f)
    docs: List[Document] = []
    for item in raw:
        metadata = {
            "title": item.get("title"),
            "date": item.get("date"),
            "source": item.get("source"),
            "tags": ",".join(item.get("tags", [])),
            "source_type": source_type,
        }
        text = item.get("body", "")
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def build_vectorstore(docs: Iterable[Document], persist_dir: Path) -> None:
    settings = get_settings()
    embeddings = AzureOpenAIEmbeddings(
        model=settings.azure.embedding_deployment,
        azure_endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        api_version=settings.azure.api_version,
    )
    Chroma.from_documents(list(docs), embedding=embeddings, persist_directory=str(persist_dir))
    persist_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Vector store persisted to {persist_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vectorstore from sample corpora.")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(get_settings().persist_dir),
        help="Directory to persist Chroma DB",
    )
    parser.add_argument("--sk-corpus", type=Path, default=Path(get_settings().sk_corpus_path))
    parser.add_argument(
        "--global-corpus", type=Path, default=Path(get_settings().global_corpus_path)
    )
    args = parser.parse_args()

    docs: List[Document] = []
    docs.extend(load_json_corpus(args.sk_corpus, source_type="sk"))
    docs.extend(load_json_corpus(args.global_corpus, source_type="global"))
    build_vectorstore(docs, args.persist_dir)


if __name__ == "__main__":
    main()

