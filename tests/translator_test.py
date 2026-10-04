import shutil
import threading
import time
from pathlib import Path

import pytest

from api.errors import ProviderError
from core import retry
from core.providers import Provider
from core.retry import TranslationCancelled
from core.srt import parse_srt
from core.translation import translate_subtitles, TranslationIncomplete, MAX_CONTENT_ATTEMPTS

TEST_DIR = Path(__file__).parent
SAMPLE_FILE = TEST_DIR / "data" / "sample.srt"


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.srt"
    shutil.copy(SAMPLE_FILE, path)
    return path


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(retry.time, "sleep", recorded.append)
    return recorded


def translate_lines(item):
    return {"id": item["id"], "lines": [line.upper() for line in item["lines"]]}


def translate_all(items):
    return {"translations": [translate_lines(item) for item in items]}


class RecordingProvider:
    def __init__(self, respond):
        self.respond = respond
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, instructions, items):
        with self.lock:
            self.requests.append([item["id"] for item in items])
            call_number = len(self.requests)
        return self.respond(items, call_number)

    def provider(self):
        return Provider("Fake", "fake-model", self)


def run(path, recording, **kwargs):
    kwargs.setdefault("log", lambda message: None)
    return translate_subtitles(str(path), "PL", recording.provider(), **kwargs)


def output_blocks(result_path):
    return parse_srt(result_path)


def assert_complete_translation(source_path, result_path):
    source = parse_srt(str(source_path))
    result = output_blocks(result_path)
    assert [(b["index"], b["start"], b["end"]) for b in result] == [(b["index"], b["start"], b["end"]) for b in source]
    assert [b["lines"] for b in result] == [[line.upper() for line in b["lines"]] for b in source]


def test_complete_structured_batches_succeed(sample):
    recording = RecordingProvider(lambda items, n: translate_all(items))

    result_path = run(sample, recording)

    assert Path(result_path) == sample.with_name("sample.pl.translated.srt")
    assert_complete_translation(sample, result_path)
    assert sorted(len(r) for r in recording.requests) == [2, 15, 15]


def test_source_file_is_not_modified(sample):
    original = sample.read_bytes()

    run(sample, RecordingProvider(lambda items, n: translate_all(items)))

    assert sample.read_bytes() == original


def test_missing_last_item_retries_only_that_item(tmp_path):
    srt = tmp_path / "three.srt"
    srt.write_text(
        "13\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "14\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "15\n00:00:05,000 --> 00:00:06,000\nAgain\n",
        encoding="utf-8",
    )
    recording = RecordingProvider(lambda items, n: translate_all(items[:-1] if n == 1 else items))

    result_path = run(srt, recording)

    assert recording.requests == [[0, 1, 2], [2]]
    assert_complete_translation(srt, result_path)


def test_missing_middle_item_retries_only_that_item(tmp_path):
    srt = tmp_path / "three.srt"
    srt.write_text(
        "7\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "8\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "9\n00:00:05,000 --> 00:00:06,000\nAgain\n",
        encoding="utf-8",
    )
    recording = RecordingProvider(lambda items, n: translate_all([i for i in items if n > 1 or i["id"] != 1]))

    result_path = run(srt, recording)

    assert recording.requests == [[0, 1, 2], [1]]
    assert_complete_translation(srt, result_path)


def test_fourteen_of_fifteen_resends_only_the_missing_item(sample):
    recording = RecordingProvider(lambda items, n: translate_all([i for i in items if i["id"] != 9 or len(items) == 1]))

    result_path = run(sample, recording, parallel=1)

    assert [9] in recording.requests
    assert all(9 not in r or r == [9] or len(r) == 15 for r in recording.requests)
    assert len(recording.requests) == 4
    assert_complete_translation(sample, result_path)


def test_wrong_line_count_is_retried(sample):
    def respond(items, n):
        result = translate_all(items)
        if n == 1:
            result["translations"][1]["lines"] = ["ONE LINE INSTEAD OF TWO"]
        return result

    recording = RecordingProvider(respond)

    result_path = run(sample, recording, parallel=1)

    assert recording.requests[1] == [1]
    assert_complete_translation(sample, result_path)


def test_duplicate_id_is_not_silently_accepted(sample):
    def respond(items, n):
        result = translate_all(items)
        if n == 1:
            result["translations"].append({"id": 0, "lines": ["DUPLICATE"]})
        return result

    recording = RecordingProvider(respond)

    result_path = run(sample, recording, parallel=1)

    assert recording.requests[1] == [0]
    assert_complete_translation(sample, result_path)


def test_malformed_structured_response_is_retried(sample):
    recording = RecordingProvider(lambda items, n: None if n == 1 else translate_all(items))

    result_path = run(sample, recording, parallel=1)

    assert recording.requests[0] == recording.requests[1]
    assert_complete_translation(sample, result_path)


def test_extra_unrequested_ids_cannot_corrupt_output(sample):
    def respond(items, n):
        result = translate_all(items)
        result["translations"] += [{"id": 999, "lines": ["BOGUS"]}, {"id": (items[0]["id"] + 15) % 32, "lines": ["WRONG BATCH"]}]
        return result

    result_path = run(sample, RecordingProvider(respond), parallel=1)

    assert_complete_translation(sample, result_path)


def test_exhausted_content_retries_fail_without_output(sample):
    recording = RecordingProvider(lambda items, n: translate_all([i for i in items if i["id"] != 20]))

    with pytest.raises(TranslationIncomplete) as error:
        run(sample, recording, parallel=1)

    assert error.value.unresolved_ids == [20]
    assert recording.requests.count([20]) == MAX_CONTENT_ATTEMPTS - 1
    assert not sample.with_name("sample.pl.translated.srt").exists()
    assert not sample.with_name("sample.pl.translated.srt.part").exists()


def test_parallel_batches_cannot_silently_lose_a_batch(sample):
    recording = RecordingProvider(lambda items, n: {"translations": []} if items[0]["id"] >= 15 else translate_all(items))

    with pytest.raises(TranslationIncomplete):
        run(sample, recording, parallel=3)

    assert not sample.with_name("sample.pl.translated.srt").exists()


def test_failing_parallel_batch_fails_whole_translation(sample):
    def respond(items, n):
        if items[0]["id"] == 15:
            raise ProviderError("Fake", "invalid_request", "bad request", status=400)
        return translate_all(items)

    with pytest.raises(ProviderError):
        run(sample, RecordingProvider(respond), parallel=3)

    assert not sample.with_name("sample.pl.translated.srt").exists()


@pytest.mark.parametrize("error", [
    ProviderError("Fake", "rate_limit", "slow down", status=429),
    ProviderError("Fake", "transient", "unavailable", status=503),
    ProviderError("Fake", "network", "TimeoutError: timed out"),
])
def test_transient_failures_are_retried(sample, sleeps, error):
    def respond(items, n):
        if n == 1:
            raise error
        return translate_all(items)

    recording = RecordingProvider(respond)

    result_path = run(sample, recording, parallel=1)

    assert recording.requests[0] == recording.requests[1]
    assert sum(sleeps) > 0
    assert_complete_translation(sample, result_path)


def test_retry_after_is_honored(sample, sleeps):
    def respond(items, n):
        if n == 1:
            raise ProviderError("Fake", "rate_limit", "slow down", status=429, retry_after=7)
        return translate_all(items)

    run(sample, RecordingProvider(respond), parallel=1)

    assert sum(sleeps) == pytest.approx(7)


@pytest.mark.parametrize("error", [
    ProviderError("Fake", "auth", "invalid key", status=401),
    ProviderError("Fake", "auth", "forbidden", status=403),
    ProviderError("Fake", "quota", "insufficient_quota", status=429),
])
def test_non_transient_failures_are_not_retried(sample, sleeps, error):
    def respond(items, n):
        raise error

    recording = RecordingProvider(respond)

    with pytest.raises(ProviderError) as raised:
        run(sample, recording, parallel=1)

    assert raised.value is error
    assert len(recording.requests) == 1
    assert sleeps == []
    assert not sample.with_name("sample.pl.translated.srt").exists()


def test_cancel_after_first_batch(sample):
    first_call = threading.Event()

    def respond(items, n):
        first_call.set()
        time.sleep(0.3)
        return translate_all(items)

    recording = RecordingProvider(respond)

    with pytest.raises(TranslationCancelled, match="Translation interrupted by client"):
        run(sample, recording, parallel=1, check_cancelled=first_call.is_set)

    time.sleep(0.5)
    assert len(recording.requests) == 1
    assert not sample.with_name("sample.pl.translated.srt").exists()


def test_cancel_during_retry_wait_stops_promptly(sample, monkeypatch):
    cancel = threading.Event()

    def respond(items, n):
        cancel.set()
        raise ProviderError("Fake", "rate_limit", "slow down", status=429, retry_after=60)

    started = time.monotonic()
    with pytest.raises(TranslationCancelled):
        run(sample, RecordingProvider(respond), parallel=1, check_cancelled=cancel.is_set)

    assert time.monotonic() - started < 5


def test_progress_is_reported_until_complete(sample):
    progress = []

    run(sample, RecordingProvider(lambda items, n: translate_all(items)), report_progress=lambda done, total: progress.append((done, total)))

    assert progress[-1] == (3, 3)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)


def test_non_sequential_srt_indices_and_timestamps_are_preserved(tmp_path):
    srt = tmp_path / "odd.srt"
    srt.write_text(
        "101\n01:00:01,000 --> 01:00:02,250\nFirst\nsecond line\n\n"
        "5\n01:00:03,000 --> 01:00:04,000\nBack in time\n\n"
        "5\n01:00:05,000 --> 01:00:06,000\nDuplicate index\n",
        encoding="utf-8",
    )

    result_path = run(srt, RecordingProvider(lambda items, n: translate_all(items)))

    assert Path(result_path).read_text(encoding="utf-8") == (
        "101\n01:00:01,000 --> 01:00:02,250\nFIRST\nSECOND LINE\n\n"
        "5\n01:00:03,000 --> 01:00:04,000\nBACK IN TIME\n\n"
        "5\n01:00:05,000 --> 01:00:06,000\nDUPLICATE INDEX"
    )


def test_mock_provider_round_trips_sample(sample):
    from api import mock

    result_path = translate_subtitles(str(sample), "PL", Provider("Mock (Test)", "mock-model", mock.translate), log=lambda m: None)

    assert output_blocks(result_path) == parse_srt(str(sample))
