from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagChunk:
    text: str
    source: str


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = max(0, end - overlap)
        if start == end:
            break
    return chunks


def load_documents(kb_dir: Path) -> List[RagChunk]:
    chunks: List[RagChunk] = []
    for path in sorted(kb_dir.glob("**/*")):
        if path.is_dir():
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            text = _read_text_file(path)
        elif path.suffix.lower() in {".pdf"}:
            text = _read_pdf(path)
        else:
            continue

        text = _clean_text(text)
        if not text:
            continue

        for chunk in _chunk_text(text):
            chunks.append(RagChunk(text=chunk, source=path.name))
    return chunks


class RagIndex:
    def __init__(self, chunks: List[RagChunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def query(self, text: str, top_k: int = 3) -> List[RagChunk]:
        if not self.chunks or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


def build_rag_index(kb_dir: Path) -> RagIndex:
    chunks = load_documents(kb_dir)
    return RagIndex(chunks)
