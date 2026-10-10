import re

from core.prompt import TRANSLATION_SCHEMA, serialize_items
from .errors import ProviderError
from .transport import post_json, decode_json_or_none, error_object, status_category, parse_retry_after

PROVIDER = "Gemini"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TIMEOUT_SECONDS = 120
FAILED_STATUSES = ("failed", "cancelled")
DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)s$")
THINKING_LEVELS = {
    "gemini-3.5-flash-lite": "minimal",
    "gemini-3.8-flash": "low",
}


def translate(instructions, items, model, api_key):
    response = post_json(
        URL,
        {"x-goog-api-key": api_key},
        build_request(instructions, items, model),
        TIMEOUT_SECONDS,
        PROVIDER,
        normalize_error,
    )
    raise_if_failed(response)
    return extract_payload(response)


def build_request(instructions, items, model):
    request = {
        "model": model,
        "store": False,
        "system_instruction": instructions,
        "input": serialize_items(items),
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": TRANSLATION_SCHEMA,
        },
    }
    if model in THINKING_LEVELS:
        request["generation_config"] = {"thinking_level": THINKING_LEVELS[model]}
    return request


def raise_if_failed(response):
    status = response.get("status")
    if status in FAILED_STATUSES:
        message = error_object(response).get("message") or f"interaction {status}"
        raise ProviderError(PROVIDER, "transient", message, status=200)


def extract_payload(response):
    if response.get("status") not in (None, "completed"):
        return None
    return decode_json_or_none("".join(output_texts(response)))


def output_texts(response):
    return [
        content.get("text", "")
        for step in dicts(response.get("steps"))
        if step.get("type") == "model_output"
        for content in dicts(step.get("content"))
        if content.get("type") == "text"
    ]


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
