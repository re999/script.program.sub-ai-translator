import pytest

from api.errors import ProviderError
from core import retry
from core.retry import call_with_retries, TranslationCancelled, MAX_TRANSPORT_ATTEMPTS


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(retry.time, "sleep", recorded.append)
    monkeypatch.setattr(retry.random, "uniform", lambda low, high: high)
    return recorded


def failing(*errors, result="ok"):
    calls = []

    def request():
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result

    return request, calls


def never_cancelled():
    return False


def test_success_needs_single_attempt(sleeps):
    request, calls = failing()
    assert call_with_retries(request, never_cancelled, print) == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_exponential_backoff_is_bounded(sleeps):
    error = ProviderError("Fake", "transient", "busy", status=503)
    request, calls = failing(error, error, error)

    assert call_with_retries(request, never_cancelled, lambda m: None) == "ok"

    assert len(calls) == 4
    assert sum(sleeps) == pytest.approx(1 + 2 + 4)
    assert max(sleeps) <= retry.CANCEL_POLL_SECONDS


def test_backoff_jitter_stays_within_ceiling():
    for attempt in range(1, 10):
        delay = retry.backoff_delay(attempt)
        ceiling = min(retry.MAX_BACKOFF_SECONDS, retry.BASE_DELAY_SECONDS * 2 ** (attempt - 1))
        assert ceiling / 2 <= delay <= ceiling


def test_retry_after_overrides_backoff(sleeps):
    request, _ = failing(ProviderError("Fake", "rate_limit", "slow", status=429, retry_after=3.5))

    call_with_retries(request, never_cancelled, lambda m: None)

    assert sum(sleeps) == pytest.approx(3.5)


def test_excessive_retry_after_fails_instead_of_blocking(sleeps):
    error = ProviderError("Fake", "rate_limit", "slow", status=429, retry_after=retry.MAX_RETRY_AFTER_SECONDS + 1)
    request, calls = failing(error)

    with pytest.raises(ProviderError):
        call_with_retries(request, never_cancelled, lambda m: None)

    assert len(calls) == 1
    assert sleeps == []


def test_attempts_are_finite(sleeps):
    error = ProviderError("Fake", "network", "TimeoutError")
    request, calls = failing(*[error] * 20)

    with pytest.raises(ProviderError):
        call_with_retries(request, never_cancelled, lambda m: None)

    assert len(calls) == MAX_TRANSPORT_ATTEMPTS


@pytest.mark.parametrize("category", ["auth", "quota", "invalid_request", "server"])
def test_non_retryable_categories_fail_immediately(sleeps, category):
    request, calls = failing(ProviderError("Fake", category, "nope", status=400))

    with pytest.raises(ProviderError):
        call_with_retries(request, never_cancelled, lambda m: None)

    assert len(calls) == 1
    assert sleeps == []


def test_unexpected_exceptions_are_not_retried(sleeps):
    request, calls = failing(KeyError("bug"))

    with pytest.raises(KeyError):
        call_with_retries(request, never_cancelled, lambda m: None)

    assert len(calls) == 1


def test_cancellation_interrupts_retry_wait(sleeps):
    request, calls = failing(ProviderError("Fake", "rate_limit", "slow", status=429, retry_after=60))
    cancelled = lambda: len(sleeps) >= 3

    with pytest.raises(TranslationCancelled):
        call_with_retries(request, cancelled, lambda m: None)

    assert len(calls) == 1
    assert sum(sleeps) < 1


def test_cancellation_before_request_skips_it(sleeps):
    request, calls = failing()

    with pytest.raises(TranslationCancelled):
        call_with_retries(request, lambda: True, lambda m: None)

    assert calls == []


def test_logs_contain_category_and_delay_but_no_payload(sleeps):
    messages = []
    request, _ = failing(ProviderError("Fake", "rate_limit", "slow", status=429, retry_after=2))

    call_with_retries(request, never_cancelled, messages.append)

    assert messages == ["request attempt 1/5 failed: Fake rate_limit error (HTTP 429): slow; retrying in 2.0s"]
