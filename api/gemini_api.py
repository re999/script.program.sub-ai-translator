import re
import urllib.parse

from core.prompt import TRANSLATION_SCHEMA, serialize_items
from .errors import ProviderError
from .transport import post_json, decode_json_or_none, error_object, status_category, parse_retry_after

PROVIDER = "Gemini"
URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_SECONDS = 120
DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)s$")


def translate(instructions, items, model, api_key):
    response = post_json(
        URL_TEMPLATE.format(model=urllib.parse.quote(model, safe="")),
        {"x-goog-api-key": api_key},
        build_request(instructions, items),
        TIMEOUT_SECONDS,
        PROVIDER,
        normalize_error,
    )
    return extract_payload(response)


def build_request(instructions, items):
    return {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": [{"text": serialize_items(items)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": TRANSLATION_SCHEMA,
        },
    }


def extract_payload(response):
    candidates = dicts(response.get("candidates"))
    content = candidates[0].get("content") if candidates else None
    parts = dicts(content.get("parts")) if isinstance(content, dict) else []
    return decode_json_or_none("".join(part.get("text", "") for part in parts if not part.get("thought")))


def dicts(value):
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def normalize_error(status, body, headers):
    error = error_object(body)
    details = dicts(error.get("details"))
    return ProviderError(
        PROVIDER,
        error_category(status, details),
        error.get("message") or "request failed",
        status=status,
        retry_after=retry_after(headers, details),
    )


def error_category(status, details):
    if status == 429 and daily_quota_exhausted(details):
        return "quota"
    if status == 400 and any(detail.get("reason") == "API_KEY_INVALID" for detail in details):
        return "auth"
    return status_category(status)


def daily_quota_exhausted(details):
    violations = [
        violation
        for detail in details_of_type(details, "QuotaFailure")
        for violation in dicts(detail.get("violations"))
    ]
    return any("PerDay" in str(violation.get("quotaId", "")) for violation in violations)


def retry_after(headers, details):
    from_header = parse_retry_after(headers.get("Retry-After"))
    if from_header is not None:
        return from_header
    delays = [detail.get("retryDelay") for detail in details_of_type(details, "RetryInfo")]
    return parse_duration(delays[0]) if delays else None


def details_of_type(details, type_name):
    return [detail for detail in details if str(detail.get("@type", "")).endswith(f".{type_name}")]


def parse_duration(value):
    match = DURATION_PATTERN.match(str(value or ""))
    return float(match.group(1)) if match else None
