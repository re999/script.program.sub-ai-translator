from .config import MODELS, LANGUAGES, DEFAULT_PARALLEL_REQUESTS, DEFAULT_PRICE_PER_1000_TOKENS
import xbmcaddon
from xbmcaddon import Addon
import xbmc
from api import mock, openai, gemini
from backoff import rate_limited_backoff_on_429

addon = Addon("script.program.sub-ai-translator")

_GEMINI_MODELS_CACHE = None

PROVIDERS = {
    "OpenAI": {
        "get_config": lambda: {
            "provider": "OpenAI",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("api_key"),
            "model": get_enum("model", MODELS),
            "price_per_1000_tokens": float(addon.getSetting("price_per_1000_tokens") or DEFAULT_PRICE_PER_1000_TOKENS),
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": 3 #max(1, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS))
        },
        "call_fn": rate_limited_backoff_on_429(min_interval=0, retries=3, base_delay=1.0, max_delay=8.0)(lambda prompt, model, api_key: openai(prompt, model, api_key)
)
    },
    "Gemini": {
        "get_config": lambda: {
            "provider": "Gemini",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("gemini_api_key"),
            "model": resolve_gemini_model(addon, logger=lambda msg: xbmc.log(msg, xbmc.LOGDEBUG)),
            "price_per_1000_tokens": 0.0,
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": 1
        },
        "call_fn": rate_limited_backoff_on_429()(lambda prompt, model, api_key: gemini(prompt, model, api_key, logger=lambda msg: xbmc.log(msg, xbmc.LOGDEBUG)))
    },
    "Mock (Test)": {
        "get_config": lambda: {
            "provider": "Mock (Test)",
            "lang": get_effective_lang(),
            "api_key": "",
            "model": "mock-model",
            "price_per_1000_tokens": 0.0,
            "use_mock": True,
            "parallel": max(1, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS))
        },
        "call_fn": mock
    }
}

def get_enum(setting_id, options):
    try:
        idx = int(addon.getSetting(setting_id))
        return options[idx] if 0 <= idx < len(options) else ""
    except Exception:
        return ""

def get_effective_lang():
    lang = get_enum("target_lang", LANGUAGES)
    return addon.getSetting("custom_lang") if lang == "Other" else lang

def get():
    provider_options = list(PROVIDERS.keys())
    provider = get_enum("provider", provider_options)
    return PROVIDERS.get(provider, PROVIDERS["Mock (Test)"])["get_config"]()

def get_call_fn():
    provider_options = list(PROVIDERS.keys())
    provider = get_enum("provider", provider_options)
    return PROVIDERS.get(provider, PROVIDERS["Mock (Test)"])["call_fn"]

def list_gemini_models(api_key, logger):
    global _GEMINI_MODELS_CACHE
    if _GEMINI_MODELS_CACHE is not None:
        return _GEMINI_MODELS_CACHE

    try:
        models = list_models(api_key)
        _GEMINI_MODELS_CACHE = models
        logger(f"[GEMINI] Available models: {models}")
        return models
    except Exception as e:
        logger(f"[GEMINI] Failed to list models: {e}")
        return []

def resolve_gemini_model(addon, logger):
    api_key = addon.getSetting("gemini_api_key")

    legacy_models = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash",
    ]

    try:
        idx = int(addon.getSetting("gemini_model"))
    except Exception:
        idx = 2

    if idx >= 3:
        try:
            tier_idx = int(addon.getSetting("gemini_tier"))
        except Exception:
            tier_idx = 0

        tier = "pro" if tier_idx == 1 else "flash"

        available = list_gemini_models(api_key, logger)
        if not available:
            return "gemini-2.0-flash"

        def score(m):
            ml = m.lower()
            return (
                "preview" in ml or "exp" in ml,
                0 if ("2.5" in ml or "3" in ml) else 1 if "2.0" in ml else 2,
            )

        candidates = [m for m in available if tier in m.lower()]
        candidates.sort(key=score)

        if candidates:
            return candidates[0]

        fallback = sorted(available, key=score)
        flash = [m for m in fallback if "flash" in m.lower()]
        return flash[0] if flash else fallback[0]

    selected = legacy_models[idx] if 0 <= idx < len(legacy_models) else "gemini-2.0-flash"

    if selected.startswith("gemini-1.5-"):
        return "gemini-2.0-flash"

    available = list_gemini_models(api_key, logger)
    if available and selected not in available:
        return "gemini-2.0-flash"

    return selected


def list_models(api_key):
    import urllib.request, json
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": api_key}
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        data = json.loads(res.read().decode())
        return [
            m["name"].split("/")[-1]
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
