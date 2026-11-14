"""A stand-in for the Anthropic client that replays saved replies, so checks are free and repeatable.

fixtures/replies.json holds {"recorded": ..., "model": ..., "source": ..., "replies": [...]}. Each reply has
"match", text that must appear in the conversation's last user message, an optional "turn" (how many
assistant messages came before), and "response", a Messages API response as JSON.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


class NoSavedReply(LookupError):
    pass


def _obj(value: Any, key: str = "") -> Any:
    """Turn response JSON into objects with attributes, as the SDK returns. Tool inputs stay dictionaries."""
    if key == "input":
        return value
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _obj(v, k) for k, v in value.items()})
    if isinstance(value, list):
        return [_obj(item) for item in value]
    return value


def _text(content: Any) -> str:
    """Every piece of text in a message's content: plain strings, text blocks, documents and tool results."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            for field in ("text", "content"):
                if field in block:
                    parts.append(_text(block[field]))
            source = block.get("source")
            if isinstance(source, dict) and isinstance(source.get("data"), str):
                parts.append(source["data"])
        else:
            parts.append(_text(getattr(block, "text", "") or ""))
    return "\n".join(parts)


class _Messages:
    def __init__(self, client: "ReplayClient") -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._answer(kwargs)

    def parse(self, **kwargs: Any) -> Any:
        """Like client.messages.parse: the reply's JSON text is validated against output_format."""
        response = self._client._answer(kwargs)
        output_format = kwargs.get("output_format")
        response.parsed_output = None
        if response.stop_reason == "end_turn" and output_format is not None:
            text = next(block.text for block in response.content if block.type == "text")
            response.parsed_output = output_format.model_validate_json(text)
        return response


class ReplayClient:
    def __init__(self, saved: dict) -> None:
        self.saved = saved
        self.calls: list[dict] = []
        self.messages = _Messages(self)

    def _answer(self, kwargs: dict) -> Any:
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        users = [m for m in messages if m.get("role") == "user"]
        if not users:
            raise NoSavedReply("Send at least one user message.")
        last = _text(users[-1].get("content"))
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        matches = [
            reply
            for reply in self.saved["replies"]
            if reply["match"] in last and reply.get("turn", turn) == turn
        ]
        if not matches:
            raise NoSavedReply(
                "No saved reply matches this request. The checks replay saved replies, so send the text you were "
                f"given inside the user message, unchanged. Last user message began: {last[:80]!r}"
            )
        reply = max(matches, key=lambda r: len(r["match"]))
        return _obj(json.loads(json.dumps(reply["response"])))
