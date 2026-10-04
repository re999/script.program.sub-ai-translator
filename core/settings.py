from functools import partial

from .config import LANGUAGES, DEFAULT_PARALLEL_REQUESTS, DEFAULT_PRICE_PER_1000_TOKENS
from .models import resolve_openai_model, resolve_gemini_model
from .providers import Provider
import xbmcaddon
from xbmcaddon import Addon
from api import mock, openai, gemini

addon = Addon("script.program.sub-ai-translator")

PROVIDERS = {
    "OpenAI": {
        "get_config": lambda: {
            "provider": "OpenAI",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("api_key"),
            "model": resolve_openai_model(get_index("model")),
            "price_per_1000_tokens": float(addon.getSetting("price_per_1000_tokens") or DEFAULT_PRICE_PER_1000_TOKENS),
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": 3 #max(1, int(addon.getSetting("parallel_requests") or DEFAULT_PARALLEL_REQUESTS))
        },
        "translate": openai.translate
    },
    "Gemini": {
        "get_config": lambda: {
            "provider": "Gemini",
            "lang": get_effective_lang(),
            "api_key": addon.getSetting("gemini_api_key"),
            "model": resolve_gemini_model(get_index("gemini_model")),
            "price_per_1000_tokens": 0.0,
            "use_mock": addon.getSettingBool("use_mock"),
            "parallel": 1
        },
        "translate": gemini.translate
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
        "translate": mock.translate
    }
}

def get_index(setting_id):
    try:
        return int(addon.getSetting(setting_id))
    except Exception:
        return None

def get_enum(setting_id, options):
    idx = get_index(setting_id)
    return options[idx] if idx is not None and 0 <= idx < len(options) else ""

def get_effective_lang():
    lang = get_enum("target_lang", LANGUAGES)
    return addon.getSetting("custom_lang") if lang == "Other" else lang

def selected_provider():
    provider_options = list(PROVIDERS.keys())
    provider = get_enum("provider", provider_options)
    return PROVIDERS.get(provider, PROVIDERS["Mock (Test)"])

def get():
    return selected_provider()["get_config"]()

def get_provider(cfg):
    translate = PROVIDERS[cfg["provider"]]["translate"]
    return Provider(cfg["provider"], cfg["model"], partial(translate, model=cfg["model"], api_key=cfg["api_key"]))
