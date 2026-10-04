from core.prompt import TRANSLATION_SCHEMA, serialize_items
from .errors import ProviderError
from .transport import post_json, decode_json_or_none, error_object, status_category, parse_retry_after

PROVIDER = "OpenAI"
URL = "https://api.openai.com/v1/responses"
TIMEOUT_SECONDS = 120
SCHEMA_NAME = "subtitle_translations"
QUOTA_CODES = ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active")


def translate(instructions, items, model, api_key):
    response = post_json(
        URL,
        {"Authorization": f"Bearer {api_key}"},
        build_request(instructions, items, model),
        TIMEOUT_SECONDS,
        PROVIDER,
        normalize_error,
    )
    return extract_payload(response)


def build_request(instructions, items, model):
    return {
        "model": model,
        "instructions": instructions,
        "input": serialize_items(items),
        "text": {
            "format": {
                "type": "json_schema",
                "name": SCHEMA_NAME,
                "strict": True,
                "schema": TRANSLATION_SCHEMA,
            }
        },
        "store": False,
    }


def extract_payload(response):
    return decode_json_or_none("".join(output_texts(response)))


def output_texts(response):
    return [
        part.get("text", "")
        for message in dicts(response.get("output"))
        if message.get("type") == "message"
        for part in dicts(message.get("content"))
        if part.get("type") == "output_text"
    ]


def dicts(value):
    return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []


def normalize_error(status, body, headers):
    error = error_object(body)
    return ProviderError(
        PROVIDER,
        error_category(status, error),
        error.get("message") or "request failed",
        status=status,
        retry_after=parse_retry_after(headers.get("Retry-After")),
    )


def error_category(status, error):
    if status == 429 and (error.get("code") in QUOTA_CODES or error.get("type") in QUOTA_CODES):
        return "quota"
    return status_category(status)
