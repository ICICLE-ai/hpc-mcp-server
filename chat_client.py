import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER_URL = os.environ.get("HPC_MCP_SERVER_URL", "http://127.0.0.1:8000")
CHAT_URL = f"{SERVER_URL.rstrip('/')}/chat"


def ask(prompt: str) -> str:
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    request = Request(
        CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))

    return body["response"]


def main() -> None:
    print(f"Connected to {CHAT_URL}")
    print("Type a question, or type exit to quit.")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break

        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        try:
            print(ask(prompt))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"HTTP error {exc.code}: {detail}", file=sys.stderr)
        except URLError as exc:
            print(f"Could not connect to {CHAT_URL}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
