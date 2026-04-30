from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .query_claude import ask_claude
    from .query_codex import ask_codex
    from .rag_core import offline_answer
except ImportError:
    from query_claude import ask_claude
    from query_codex import ask_codex
    from rag_core import offline_answer


class RagHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw or "{}")
            question = str(payload.get("question", "")).strip()
            top_k = int(payload.get("top_k", 5))
            if not question:
                self._json({"error": "question 값이 필요합니다."}, status=400)
                return

            if self.path == "/search":
                self._json(json.loads(offline_answer(question, top_k=top_k)))
            elif self.path == "/codex":
                self._json({"answer": ask_codex(question, top_k=top_k)})
            elif self.path == "/claude":
                self._json({"answer": ask_claude(question, top_k=top_k)})
            else:
                self._json({"error": "지원하지 않는 endpoint입니다."}, status=404)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def do_GET(self) -> None:
        self._json(
            {
                "service": "dashboard-rag",
                "endpoints": ["POST /search", "POST /codex", "POST /claude"],
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RagHandler)
    print(f"RAG API: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
