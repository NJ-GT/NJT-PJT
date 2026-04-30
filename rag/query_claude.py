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


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def ask_claude(question: str, top_k: int = 5) -> str:
    load_env_file()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 필요합니다.")

    prompt, _ = build_prompt(question, top_k=top_k)
    payload = {
        "model": os.environ.get("ANTHROPIC_RAG_MODEL", "claude-sonnet-4-5"),
        "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if "credit balance is too low" in detail.lower():
            raise RuntimeError("Claude API 키는 읽혔지만 Anthropic 계정 크레딧이 부족합니다.") from exc
        raise RuntimeError(f"Claude API HTTP {exc.code}: {detail}") from exc

    texts = [item.get("text", "") for item in data.get("content", []) if item.get("type") == "text"]
    return "\n".join(text for text in texts if text).strip()


def main() -> None:
    args = parser().parse_args()
    question = " ".join(args.question)
    if args.offline:
        print(offline_answer(question, top_k=args.top_k))
        return
    try:
        print(ask_claude(question, top_k=args.top_k))
    except Exception as exc:
        print(f"Claude RAG 오류: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
