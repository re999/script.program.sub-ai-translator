import io
import json
import socket
import urllib.error
from email.message import Message

import pytest

from api import gemini, mock, openai
from api.errors import ProviderError
from api.transport import parse_retry_after, status_category
from core.prompt import TRANSLATION_SCHEMA

ITEMS = [{"id": 4, "lines": ["Hello", "there"]}]
TRANSLATED = {"translations": [{"id": 4, "lines": ["Cześć", "tam"]}]}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def http_error(status, body, headers=None):
    message = Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("https://example.invalid", status, "error", message, io.BytesIO(json.dumps(body).encode()))


@pytest.fixture
def urlopen(monkeypatch):
    calls = []
    outcome = {}

    def fake_urlopen(request, timeout=None):
        calls.append({
            "url": request.full_url,
            "headers": {k.lower(): v for k, v in request.header_items()},
            "body": json.loads(request.data.decode()),
            "timeout": timeout,
        })
        if "error" in outcome:
            raise outcome["error"]
        return FakeResponse(json.dumps(outcome["response"]).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fake_urlopen.calls = calls
    fake_urlopen.outcome = outcome
    return fake_urlopen


def openai_response(text):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        ],
    }


def gemini_response(text):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]}


def test_openai_uses_responses_api_with_strict_schema(urlopen):
    urlopen.outcome["response"] = openai_response(json.dumps(TRANSLATED))

    payload = openai.translate("Translate to Polish", ITEMS, "gpt-5.6-luna", "sk-secret")

    call = urlopen.calls[0]
    assert payload == TRANSLATED
    assert call["url"] == "https://api.openai.com/v1/responses"
    assert call["headers"]["authorization"] == "Bearer sk-secret"
    assert call["timeout"] == openai.TIMEOUT_SECONDS
    assert call["body"]["model"] == "gpt-5.6-luna"
    assert call["body"]["instructions"] == "Translate to Polish"
    assert json.loads(call["body"]["input"]) == {"items": ITEMS}
    assert call["body"]["text"]["format"] == {
        "type": "json_schema", "name": "subtitle_translations", "strict": True, "schema": TRANSLATION_SCHEMA,
    }
    assert "tools" not in call["body"]
    assert call["body"]["store"] is False


@pytest.mark.parametrize("response", [
    openai_response('{"translations": [{"id": 4'),
    {"status": "incomplete", "output": []},
    {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]},
])
def test_openai_malformed_output_yields_no_payload(urlopen, response):
    urlopen.outcome["response"] = response
    assert openai.translate("x", ITEMS, "gpt-5.6-luna", "key") is None


def test_gemini_uses_structured_output_and_header_auth(urlopen):
    urlopen.outcome["response"] = gemini_response(json.dumps(TRANSLATED))

    payload = gemini.translate("Translate to Polish", ITEMS, "gemini-3.8-flash", "g-secret")

    call = urlopen.calls[0]
    assert payload == TRANSLATED
    assert call["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    assert "g-secret" not in call["url"]
    assert call["headers"]["x-goog-api-key"] == "g-secret"
    assert call["timeout"] == gemini.TIMEOUT_SECONDS
    assert call["body"]["systemInstruction"] == {"parts": [{"text": "Translate to Polish"}]}
    assert json.loads(call["body"]["contents"][0]["parts"][0]["text"]) == {"items": ITEMS}
    assert call["body"]["generationConfig"] == {
        "responseMimeType": "application/json", "responseJsonSchema": TRANSLATION_SCHEMA,
    }


@pytest.mark.parametrize("response", [
    gemini_response("not json"),
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
])
def test_gemini_malformed_output_yields_no_payload(urlopen, response):
    urlopen.outcome["response"] = response
    assert gemini.translate("x", ITEMS, "gemini-3.8-flash", "key") is None


def test_gemini_ignores_thought_parts(urlopen):
    urlopen.outcome["response"] = {"candidates": [{"content": {"parts": [
        {"text": "thinking...", "thought": True},
        {"text": json.dumps(TRANSLATED)},
    ]}}]}
    assert gemini.translate("x", ITEMS, "gemini-3.8-flash", "key") == TRANSLATED


@pytest.mark.parametrize("status,body,headers,category,retry_after", [
    (429, {"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}}, {"Retry-After": "12"}, "rate_limit", 12.0),
    (429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}, {}, "quota", None),
    (401, {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}, {}, "auth", None),
    (403, {"error": {"message": "Country not supported"}}, {}, "auth", None),
    (404, {"error": {"message": "The model does not exist", "code": "model_not_found"}}, {}, "invalid_request", None),
    (400, {"error": {"message": "Invalid schema"}}, {}, "invalid_request", None),
    (503, {"error": {"message": "Overloaded"}}, {}, "transient", None),
    (500, {}, {}, "transient", None),
])
def test_openai_errors_are_normalized(urlopen, status, body, headers, category, retry_after):
    urlopen.outcome["error"] = http_error(status, body, headers)

    with pytest.raises(ProviderError) as raised:
        openai.translate("x", ITEMS, "gpt-5.6-luna", "sk-secret")

    error = raised.value
    assert (error.provider, error.status, error.category, error.retry_after) == ("OpenAI", status, category, retry_after)
    assert error.retryable == (category in ("rate_limit", "transient"))
    assert "sk-secret" not in str(error)


def gemini_error(status, code_name, message, details):
    return {"error": {"code": status, "message": message, "status": code_name, "details": details}}


@pytest.mark.parametrize("status,body,category,retry_after", [
    (429, gemini_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
    ]), "rate_limit", 17.0),
    (429, gemini_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3600s"},
    ]), "quota", 3600.0),
    (400, gemini_error(400, "INVALID_ARGUMENT", "API key not valid.", [
        {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"},
    ]), "auth", None),
    (403, gemini_error(403, "PERMISSION_DENIED", "Permission denied", []), "auth", None),
    (404, gemini_error(404, "NOT_FOUND", "models/x is not found", []), "invalid_request", None),
    (503, gemini_error(503, "UNAVAILABLE", "The model is overloaded", []), "transient", None),
    (504, {}, "transient", None),
])
def test_gemini_errors_are_normalized(urlopen, status, body, category, retry_after):
    urlopen.outcome["error"] = http_error(status, body)

    with pytest.raises(ProviderError) as raised:
        gemini.translate("x", ITEMS, "gemini-3.8-flash", "g-secret")

    error = raised.value
    assert (error.provider, error.status, error.category, error.retry_after) == ("Gemini", status, category, retry_after)
    assert "g-secret" not in str(error)


@pytest.mark.parametrize("failure", [
    urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
    socket.timeout("timed out"),
    TimeoutError("timed out"),
    ConnectionResetError(104, "reset"),
])
def test_network_failures_are_retryable(urlopen, failure):
    urlopen.outcome["error"] = failure

    with pytest.raises(ProviderError) as raised:
        openai.translate("x", ITEMS, "gpt-5.6-luna", "key")

    assert raised.value.category == "network"
    assert raised.value.retryable


def test_non_json_response_body_is_retryable(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse(b"<html>gateway</html>"))

    with pytest.raises(ProviderError) as raised:
        gemini.translate("x", ITEMS, "gemini-3.8-flash", "key")

    assert raised.value.category == "malformed"
    assert raised.value.retryable


def test_retry_after_parsing():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_status_categories():
    assert [status_category(s) for s in (408, 429, 500, 502, 503, 504)] == ["transient", "rate_limit", "transient", "transient", "transient", "transient"]
    assert [status_category(s) for s in (400, 401, 403, 404, 422, 501)] == ["invalid_request", "auth", "auth", "invalid_request", "invalid_request", "server"]


def test_mock_echoes_structured_translations():
    assert mock.translate("x", ITEMS) == {"translations": ITEMS}
