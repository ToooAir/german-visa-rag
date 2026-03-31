"""Unit tests for src/models/chat.py"""

import pytest
from pydantic import ValidationError

from src.models.chat import (
    ChatCompletion,
    ChatCompletionStream,
    Choice,
    Message,
    MessageRole,
    Usage,
)

# ─── MessageRole ──────────────────────────────────────────────────────────────


class TestMessageRole:
    def test_values(self):
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"

    def test_is_str(self):
        assert isinstance(MessageRole.USER, str)


# ─── Message ──────────────────────────────────────────────────────────────────


class TestMessage:
    def test_basic_creation(self):
        msg = Message(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"
        assert msg.name is None

    def test_with_name(self):
        msg = Message(role=MessageRole.ASSISTANT, content="hi", name="bot")
        assert msg.name == "bot"

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="x")  # type: ignore[arg-type]


# ─── Choice ───────────────────────────────────────────────────────────────────


class TestChoice:
    def test_defaults(self):
        msg = Message(role=MessageRole.ASSISTANT, content="ok")
        choice = Choice(message=msg)
        assert choice.index == 0
        assert choice.finish_reason is None

    def test_custom_values(self):
        msg = Message(role=MessageRole.ASSISTANT, content="done")
        choice = Choice(index=2, message=msg, finish_reason="stop")
        assert choice.index == 2
        assert choice.finish_reason == "stop"


# ─── Usage ────────────────────────────────────────────────────────────────────


class TestUsage:
    def test_fields(self):
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


# ─── ChatCompletion ───────────────────────────────────────────────────────────


class TestChatCompletion:
    def _make(self) -> ChatCompletion:
        msg = Message(role=MessageRole.ASSISTANT, content="answer")
        choice = Choice(message=msg, finish_reason="stop")
        usage = Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        return ChatCompletion(
            id="chatcmpl-abc123",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[choice],
            usage=usage,
        )

    def test_object_type(self):
        cc = self._make()
        assert cc.object == "chat.completion"

    def test_fields(self):
        cc = self._make()
        assert cc.id == "chatcmpl-abc123"
        assert cc.model == "gpt-4o-mini"
        assert len(cc.choices) == 1
        assert cc.usage.total_tokens == 15


# ─── ChatCompletionStream ─────────────────────────────────────────────────────


class TestChatCompletionStream:
    def test_object_type(self):
        chunk = ChatCompletionStream(
            id="chatcmpl-xyz",
            created=1700000001,
            model="gpt-4o-mini",
            choices=[{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0]["delta"]["content"] == "hi"
