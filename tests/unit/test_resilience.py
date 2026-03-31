"""Unit tests for src/utils/resilience.py"""

import time

import pytest

from src.utils.resilience import CircuitBreaker, async_retry

# ─── CircuitBreaker ────────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        assert cb.is_open is False
        assert cb.failures == 0
        assert cb.last_failure_time is None

    def test_is_available_when_closed(self):
        cb = CircuitBreaker()
        assert cb.is_available() is True

    def test_record_success_resets_state(self):
        cb = CircuitBreaker()
        cb.failures = 4
        cb.is_open = True
        cb.record_success()
        assert cb.failures == 0
        assert cb.is_open is False

    def test_record_failure_increments_counter(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2
        assert cb.is_open is False

    def test_record_failure_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

    def test_is_not_available_when_open_within_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, timeout=9999)
        cb.record_failure()
        cb.last_failure_time = time.time()
        assert cb.is_available() is False

    def test_resets_after_timeout_elapses(self):
        cb = CircuitBreaker(failure_threshold=1, timeout=1)
        cb.record_failure()
        cb.is_open = True
        cb.last_failure_time = time.time() - 2  # 2 s ago > 1 s timeout
        assert cb.is_available() is True
        assert cb.is_open is False
        assert cb.failures == 0

    def test_multiple_successes_keep_circuit_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_success()
        cb.record_success()
        assert cb.is_available() is True


# ─── async_retry ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_retry_succeeds_without_retrying():
    calls = []

    @async_retry(max_attempts=3, exceptions=(ValueError,))
    async def succeed():
        calls.append(1)
        return "ok"

    result = await succeed()
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_retry_reraises_after_max_attempts_1():
    """With max_attempts=1 there is no retry wait — fails immediately."""

    @async_retry(max_attempts=1, exceptions=(ValueError,))
    async def always_fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await always_fail()


@pytest.mark.asyncio
async def test_async_retry_does_not_catch_unmatched_exception():
    """TypeError is not in the retry list — should propagate instantly."""

    @async_retry(max_attempts=3, exceptions=(ValueError,))
    async def raise_type_error():
        raise TypeError("not a value error")

    with pytest.raises(TypeError):
        await raise_type_error()
