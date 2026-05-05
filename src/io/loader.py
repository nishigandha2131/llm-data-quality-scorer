"""
Data loaders — accepts JSONL, CSV, Parquet, TXT, and JSON.
All return a list of Document objects with auto-detected text columns.
"""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from src.models import Document

_TEXT_COLUMNS = ("text", "content", "body", "document", "passage", "article")


def _detect_text_col(columns: list[str]) -> str:
    for col in columns:
        if col.lower() in _TEXT_COLUMNS:
            return col
    return columns[0]


def _row_to_doc(row: dict, text_col: str, source: str) -> Document:
    text = str(row.get(text_col, "")).strip()
    doc_id = str(row.get("id", str(uuid.uuid4())))
    metadata = {k: v for k, v in row.items() if k not in (text_col, "id")}
    return Document(id=doc_id, text=text, source=source, metadata=metadata)


# ── Per-format loaders ────────────────────────────────────────────────────────

def load_jsonl(path: Union[str, Path]) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = (
                    obj.get("text")
                    or obj.get("content")
                    or obj.get("body")
                    or obj.get("passage")
                    or str(obj)
                )
                doc_id = str(obj.get("id", str(uuid.uuid4())))
                meta = {k: v for k, v in obj.items() if k not in ("text", "content", "body", "id")}
                docs.append(Document(id=doc_id, text=str(text), source=str(path), metadata=meta))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSONL line {i}")
    return docs


def load_csv(path: Union[str, Path]) -> list[Document]:
    df = pd.read_csv(path, dtype=str).fillna("")
    text_col = _detect_text_col(list(df.columns))
    return [_row_to_doc(row.to_dict(), text_col, str(path)) for _, row in df.iterrows()]


def load_parquet(path: Union[str, Path]) -> list[Document]:
    df = pd.read_parquet(path).fillna("")
    df = df.astype(str)
    text_col = _detect_text_col(list(df.columns))
    return [_row_to_doc(row.to_dict(), text_col, str(path)) for _, row in df.iterrows()]


def load_txt(path: Union[str, Path], split_by: str = "\n\n") -> list[Document]:
    text = Path(path).read_text(encoding="utf-8")
    chunks = [c.strip() for c in text.split(split_by) if c.strip()]
    return [Document(id=str(uuid.uuid4()), text=chunk, source=str(path)) for chunk in chunks]


# ── Streamlit-friendly bytes loader ──────────────────────────────────────────

def load_from_bytes(content: bytes, filename: str) -> list[Document]:
    """
    Load documents from raw file bytes — used by the Streamlit uploader.
    Supports: .jsonl, .json, .csv, .parquet, .txt
    """
    ext = Path(filename).suffix.lower()

    if ext == ".jsonl":
        lines = content.decode("utf-8", errors="replace").strip().split("\n")
        docs: list[Document] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or obj.get("body") or str(obj)
                doc_id = str(obj.get("id", str(uuid.uuid4())))
                meta = {k: v for k, v in obj.items() if k not in ("text", "content", "body", "id")}
                docs.append(Document(id=doc_id, text=str(text), source=filename, metadata=meta))
            except Exception:
                logger.warning(f"Skipping malformed JSONL line {i}")
        return docs

    elif ext == ".json":
        data = json.loads(content.decode("utf-8", errors="replace"))
        if isinstance(data, list):
            docs = []
            for item in data:
                if isinstance(item, str):
                    docs.append(Document(id=str(uuid.uuid4()), text=item, source=filename))
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("body") or str(item)
                    docs.append(Document(id=str(item.get("id", uuid.uuid4())), text=str(text), source=filename))
            return docs
        elif isinstance(data, dict):
            text = data.get("text") or data.get("content") or str(data)
            return [Document(id=str(uuid.uuid4()), text=str(text), source=filename)]
        return []

    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
        text_col = _detect_text_col(list(df.columns))
        return [_row_to_doc(row.to_dict(), text_col, filename) for _, row in df.iterrows()]

    elif ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(content)).fillna("").astype(str)
        text_col = _detect_text_col(list(df.columns))
        return [_row_to_doc(row.to_dict(), text_col, filename) for _, row in df.iterrows()]

    elif ext == ".txt":
        text = content.decode("utf-8", errors="replace")
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        return [Document(id=str(uuid.uuid4()), text=chunk, source=filename) for chunk in chunks]

    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .jsonl .json .csv .parquet .txt")
