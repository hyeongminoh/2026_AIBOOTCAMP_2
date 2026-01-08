from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import get_settings
from src.rag.embeddings import get_embeddings


def load_json_corpus(path: Path, source_type: str) -> List[Document]:
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


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(split_docs: Iterable[Document], persist_dir: Path) -> None:
    embeddings = get_embeddings()
    persist_dir.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(list(split_docs), embedding=embeddings)
    db.save_local(folder_path=str(persist_dir))
    print(f"[ingest] FAISS index persisted to {persist_dir}")


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

    appendix_path = Path("data/appendix-keywords.txt")
    if appendix_path.exists():
        file = appendix_path.read_text(encoding="utf-8")
        print(f"[ingest] appendix-keywords loaded ({len(file)} chars)")
    else:
        file = ""
        print("[ingest] appendix-keywords.txt not found (optional)")

    sk_docs: List[Document] = load_json_corpus(args.sk_corpus, source_type="sk")
    gl_docs: List[Document] = load_json_corpus(args.global_corpus, source_type="global")
    build_vectorstore(split_docs(sk_docs), args.persist_dir / "sk")
    build_vectorstore(split_docs(gl_docs), args.persist_dir / "it")


if __name__ == "__main__":
    main()

