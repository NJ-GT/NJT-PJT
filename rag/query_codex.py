from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

try:
    from .rag_core import build_prompt, load_env_file, offline_answer, parser
except ImportError:
    from rag_core import build_prompt, load_env_file, offline_answer, parser


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def ask_codex(question: str, top_k: int = 5) -> str:
    load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")

    prompt, _ = build_prompt(question, top_k=top_k)
    payload = {
        "model": os.environ.get("OPENAI_RAG_MODEL", "gpt-5.2"),
        "input": prompt,
        "max_output_tokens": 900,
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise RuntimeError(
                "OpenAI API가 429(요청 한도/크레딧/모델 접근 제한)을 반환했습니다. "
                "API 키는 읽혔고 서버까지 도달했습니다. 결제/크레딧/Rate limit 또는 OPENAI_RAG_MODEL 값을 확인하세요. "
                f"상세: {detail}"
            ) from exc
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc

    if data.get("output_text"):
        return data["output_text"]

    texts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "\n".join(text for text in texts if text).strip()


def main() -> None:
    args = parser().parse_args()
    question = " ".join(args.question)
    if args.offline:
        print(offline_answer(question, top_k=args.top_k))
        return
    try:
        print(ask_codex(question, top_k=args.top_k))
    except Exception as exc:
        print(f"Codex/OpenAI RAG 오류: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
