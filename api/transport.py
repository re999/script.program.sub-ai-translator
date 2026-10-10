import http.client
import json
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

from .errors import ProviderError

TRANSIENT_STATUSES = (408, 500, 502, 503, 504)
AUTH_STATUSES = (401, 403)


def post_json(url, headers, body, timeout, provider, normalize_error):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise normalize_error(error.code, read_error_body(error), error.headers or {}) from None
    except (OSError, http.client.HTTPException) as error:
        raise ProviderError(provider, "network", f"{type(error).__name__}: {error}") from None
    envelope = decode_json_or_none(raw.decode("utf-8", errors="replace"))
    if not isinstance(envelope, dict):
        raise ProviderError(provider, "malformed", "response body is not a JSON object", status=200)
    return envelope


def read_error_body(error):
    try:
        body = decode_json_or_none(error.read().decode("utf-8", errors="replace"))
    except (OSError, http.client.HTTPException):
        return {}
    return body if isinstance(body, dict) else {}


def error_object(body):
    error = body.get("error")
    return error if isinstance(error, dict) else {}


def decode_json_or_none(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def status_category(status):
    if status == 429:
        return "rate_limit"
    if status in TRANSIENT_STATUSES:
        return "transient"
    if status in AUTH_STATUSES:
        return "auth"
    if 400 <= status < 500:
        return "invalid_request"
    return "server"


def parse_retry_after(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None
