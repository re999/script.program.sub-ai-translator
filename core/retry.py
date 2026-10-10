import random
import time

from api.errors import ProviderError

MAX_TRANSPORT_ATTEMPTS = 5
BASE_DELAY_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
MAX_RETRY_AFTER_SECONDS = 90.0
CANCEL_POLL_SECONDS = 0.2


class TranslationCancelled(Exception):
    def __init__(self):
        super().__init__("Translation interrupted by client")


def call_with_retries(request, is_cancelled, log, max_attempts=MAX_TRANSPORT_ATTEMPTS):
    for attempt in range(1, max_attempts + 1):
        raise_if_cancelled(is_cancelled)
        try:
            return request()
        except ProviderError as error:
            delay = retry_delay(error, attempt)
            if delay is None or attempt == max_attempts:
                log(f"request attempt {attempt}/{max_attempts} failed, giving up: {error.summary()}")
                raise
            log(f"request attempt {attempt}/{max_attempts} failed: {error.summary()}; retrying in {delay:.1f}s")
            wait_unless_cancelled(delay, is_cancelled)


def retry_delay(error, attempt):
    if not error.retryable:
        return None
    if error.retry_after is None:
        return backoff_delay(attempt)
    return error.retry_after if error.retry_after <= MAX_RETRY_AFTER_SECONDS else None


def backoff_delay(attempt):
    ceiling = min(MAX_BACKOFF_SECONDS, BASE_DELAY_SECONDS * 2 ** (attempt - 1))
    return random.uniform(ceiling / 2, ceiling)


def wait_unless_cancelled(delay, is_cancelled):
    remaining = delay
    while remaining > 0:
        raise_if_cancelled(is_cancelled)
        step = min(CANCEL_POLL_SECONDS, remaining)
        time.sleep(step)
        remaining -= step
    raise_if_cancelled(is_cancelled)


def raise_if_cancelled(is_cancelled):
    if is_cancelled():
        raise TranslationCancelled()
