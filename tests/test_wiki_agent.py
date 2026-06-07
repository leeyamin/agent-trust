"""Send a Wikipedia query to the running wiki agent and print the response."""

import uuid

import httpx

URL = "http://127.0.0.1:8001/"


def send_message(text: str) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [{"text": text}],
            }
        },
    }
    headers = {"A2A-Version": "1.0"}
    response = httpx.post(URL, json=payload, headers=headers, timeout=120)
    data = response.json()

    if "result" in data:
        text = data["result"]["message"]["parts"][0]["text"]
        print(text)
    else:
        print(data)
    print()


def main() -> None:
    while True:
        text = input("> ")
        send_message(text)


if __name__ == "__main__":
    main()
