from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DOCS_DIR = ROOT / "documents"


TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")


@dataclass
class RagChunk:
    source: str
    title: str
    text: str
    score: float = 0.0


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    suffixes = ("으로", "에서", "에게", "보다", "처럼", "까지", "부터", "하고", "라는", "이야", "인가", "은", "는", "이", "가", "을", "를", "의", "와", "과", "로")
    for raw in TOKEN_RE.findall(text):
        token = raw.lower()
        tokens.append(token)
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                tokens.append(token[: -len(suffix)])
                break
    return tokens


def load_env_file(path: Path | None = None) -> None:
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = value


def split_markdown(path: Path) -> list[RagChunk]:
    text = path.read_text(encoding="utf-8")
    chunks: list[RagChunk] = []
    current_title = path.stem
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("# ") or line.startswith("## "):
            if current_lines:
                chunks.append(RagChunk(path.name, current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = line.lstrip("#").strip()
        current_lines.append(line)
    if current_lines:
        chunks.append(RagChunk(path.name, current_title, "\n".join(current_lines).strip()))
    return [chunk for chunk in chunks if chunk.text]


def load_chunks() -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks.extend(split_markdown(path))
    return chunks


def search(question: str, top_k: int = 5) -> list[RagChunk]:
    chunks = load_chunks()
    query_tokens = tokenize(question)
    if not query_tokens:
        return chunks[:top_k]

    doc_freq: dict[str, int] = {}
    chunk_tokens = []
    for chunk in chunks:
        tokens = tokenize(chunk.title + "\n" + chunk.text)
        chunk_tokens.append(tokens)
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    scored: list[RagChunk] = []
    total_docs = max(1, len(chunks))
    query_set = set(query_tokens)
    for chunk, tokens in zip(chunks, chunk_tokens):
        if not tokens:
            continue
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        score = 0.0
        for token in query_set:
            tf = counts.get(token, 0)
            if not tf:
                continue
            idf = math.log((total_docs + 1) / (doc_freq.get(token, 0) + 1)) + 1
            score += (1 + math.log(tf)) * idf

        phrase_bonus = 0.0
        lowered = (chunk.title + "\n" + chunk.text).lower()
        for token in query_set:
            if len(token) >= 3 and token in lowered:
                phrase_bonus += 0.15
        score += phrase_bonus
        if chunk.source == "example_questions.md":
            score *= 0.25

        if score > 0:
            scored.append(RagChunk(chunk.source, chunk.title, chunk.text, score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k] if scored else chunks[:top_k]


def build_prompt(question: str, top_k: int = 5) -> tuple[str, list[RagChunk]]:
    chunks = search(question, top_k=top_k)
    context = "\n\n".join(
        f"[{idx}] {chunk.source} / {chunk.title}\n{chunk.text}"
        for idx, chunk in enumerate(chunks, start=1)
    )
    prompt = f"""너는 서울시 관광 지역 내 숙박시설 화재위험 대시보드의 설명 도우미다.
아래 검색 문맥만 근거로 답한다. 문맥에 없는 내용은 추측하지 말고, 대시보드에서 확인해야 한다고 말한다.
답변은 한국어로 짧고 실무적으로 작성한다.

질문:
{question}

검색 문맥:
{context}

답변 형식:
- 핵심 답변
- 근거
- 대시보드에서 확인할 위치
"""
    return prompt, chunks


def offline_answer(question: str, top_k: int = 5) -> str:
    _, chunks = build_prompt(question, top_k=top_k)
    payload = {
        "question": question,
        "sources": [
            {
                "source": chunk.source,
                "title": chunk.title,
                "score": round(chunk.score, 4),
                "text": chunk.text[:800],
            }
            for chunk in chunks
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("question", nargs="+")
    arg_parser.add_argument("--top-k", type=int, default=5)
    arg_parser.add_argument("--offline", action="store_true")
    return arg_parser
