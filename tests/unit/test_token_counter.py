"""Unit tests for src/llm/token_counter.py"""

import pytest

from src.llm.token_counter import TokenCounter, get_token_counter


class TestTokenCounter:
    @pytest.fixture
    def counter(self):
        return TokenCounter()

    def test_count_text_returns_positive_int(self, counter):
        count = counter.count_text("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_count_text_longer_text_has_more_tokens(self, counter):
        short = counter.count_text("hi")
        long = counter.count_text("hi " * 50)
        assert long > short

    def test_count_text_empty_string(self, counter):
        assert counter.count_text("") == 0

    def test_count_messages_basic(self, counter):
        messages = [{"role": "user", "content": "hello"}]
        count = counter.count_messages(messages)
        assert count > 0

    def test_count_messages_more_messages_more_tokens(self, counter):
        one = counter.count_messages([{"role": "user", "content": "hello"}])
        two = counter.count_messages(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        )
        assert two > one

    def test_estimate_cost_known_model(self, counter):
        cost = counter.estimate_cost(1000, 500, model="gpt-4o-mini")
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_unknown_model_is_zero(self, counter):
        cost = counter.estimate_cost(1000, 500, model="unknown-model-xyz")
        assert cost == 0.0

    def test_estimate_cost_zero_tokens(self, counter):
        assert counter.estimate_cost(0, 0, model="gpt-4o") == 0.0

    def test_estimate_query_cost(self, counter):
        cost = counter.estimate_query_cost("what is chancenkarte?", "It is a visa type.", model="gpt-4o-mini")
        assert isinstance(cost, float)
        assert cost >= 0


class TestGetTokenCounter:
    def test_returns_instance(self):
        result = get_token_counter()
        assert isinstance(result, TokenCounter)

    def test_singleton(self):
        a = get_token_counter()
        b = get_token_counter()
        assert a is b
