import pytest

from conftest import FakeAddon, settings_definitions
from api import gemini, mock, openai
from core import settings
from core.models import (
    OPENAI_MODEL_CHOICES, GEMINI_MODEL_CHOICES, resolve_openai_model, resolve_gemini_model,
)

PREVIOUS_SETTING_IDS = [
    "provider", "api_key", "model", "price_per_1000_tokens", "gemini_api_key", "gemini_model",
    "target_lang", "custom_lang", "style_hint", "use_mock", "parallel_requests",
]
PREVIOUS_PROVIDER_VALUES = ["OpenAI", "Gemini", "Mock (Test)"]
PREVIOUS_OPENAI_VALUES = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
PREVIOUS_GEMINI_VALUES = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-2.0-flash", "Auto"]
PREVIOUS_LANGUAGE_VALUES = ["English", "Polish", "German", "Dutch", "Spanish", "Italian", "Other"]
CURRENT_OPENAI_MODELS = {"gpt-5.6-luna", "gpt-5.6-terra"}
CURRENT_GEMINI_MODELS = {"gemini-3.8-flash"}


@pytest.fixture
def stored(monkeypatch):
    values = {}
    monkeypatch.setattr(FakeAddon, "stored", values)
    return values


def enum_values(setting_id):
    return settings_definitions()[setting_id].get("values").split("|")


def enum_default(setting_id):
    return int(settings_definitions()[setting_id].get("default"))


def test_previous_setting_ids_still_exist():
    assert set(PREVIOUS_SETTING_IDS) <= set(settings_definitions())


def test_provider_and_language_enums_keep_their_order():
    assert enum_values("provider") == PREVIOUS_PROVIDER_VALUES
    assert list(settings.PROVIDERS) == PREVIOUS_PROVIDER_VALUES
    assert enum_values("target_lang") == PREVIOUS_LANGUAGE_VALUES


@pytest.mark.parametrize("setting_id,previous,choices", [
    ("model", PREVIOUS_OPENAI_VALUES, OPENAI_MODEL_CHOICES),
    ("gemini_model", PREVIOUS_GEMINI_VALUES, GEMINI_MODEL_CHOICES),
])
def test_model_enums_only_append_new_entries(setting_id, previous, choices):
    labels = enum_values(setting_id)
    assert len(labels) == len(choices)
    assert all(labels[i].startswith(name) for i, name in enumerate(previous))
    assert all(label.startswith(choice) for label, (choice, _) in zip(labels, choices))


def test_fresh_install_defaults_point_to_new_models():
    assert OPENAI_MODEL_CHOICES[enum_default("model")][0] == "gpt-5.6-luna"
    assert GEMINI_MODEL_CHOICES[enum_default("gemini_model")][0] == "gemini-3.8-flash"


@pytest.mark.parametrize("index,expected", [(0, "gpt-5.6-luna"), (1, "gpt-5.6-terra"), (2, "gpt-5.6-terra"), (3, "gpt-5.6-luna"), (4, "gpt-5.6-terra")])
def test_openai_indices_resolve_deterministically(index, expected):
    assert resolve_openai_model(index) == expected


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_gemini_indices_resolve_to_current_model(index):
    assert resolve_gemini_model(index) == "gemini-3.8-flash"


@pytest.mark.parametrize("index", [None, -1, 5, 99])
def test_unknown_indices_fall_back_to_default_never_empty(index):
    assert resolve_openai_model(index) == "gpt-5.6-luna"
    assert resolve_gemini_model(index) == "gemini-3.8-flash"


def test_no_legacy_model_id_is_ever_sent():
    assert {model for _, model in OPENAI_MODEL_CHOICES} == CURRENT_OPENAI_MODELS
    assert {model for _, model in GEMINI_MODEL_CHOICES} == CURRENT_GEMINI_MODELS


@pytest.mark.parametrize("saved_model,expected", [("0", "gpt-5.6-luna"), ("1", "gpt-5.6-terra"), ("2", "gpt-5.6-terra")])
def test_previous_version_openai_settings_keep_working(stored, saved_model, expected):
    stored.update({
        "provider": "0", "api_key": "sk-previous", "model": saved_model, "price_per_1000_tokens": "0.002",
        "target_lang": "2", "custom_lang": "", "use_mock": "false", "parallel_requests": "5",
    })

    cfg = settings.get()
    provider = settings.get_provider(cfg)

    assert cfg["provider"] == "OpenAI"
    assert cfg["api_key"] == "sk-previous"
    assert cfg["lang"] == "German"
    assert cfg["model"] == expected
    assert cfg["price_per_1000_tokens"] == 0.002
    assert cfg["parallel"] == 3
    assert (provider.name, provider.model) == ("OpenAI", expected)
    assert provider.translate.func is openai.translate
    assert provider.translate.keywords == {"model": expected, "api_key": "sk-previous"}


@pytest.mark.parametrize("saved_model", ["0", "1", "2", "3"])
@pytest.mark.parametrize("saved_tier", ["0", "1", None])
def test_previous_version_gemini_settings_keep_working(stored, saved_model, saved_tier):
    stored.update({"provider": "1", "gemini_api_key": "g-previous", "gemini_model": saved_model, "target_lang": "1"})
    if saved_tier is not None:
        stored["gemini_tier"] = saved_tier

    cfg = settings.get()
    provider = settings.get_provider(cfg)

    assert cfg["provider"] == "Gemini"
    assert cfg["api_key"] == "g-previous"
    assert cfg["lang"] == "Polish"
    assert cfg["model"] == "gemini-3.8-flash"
    assert cfg["parallel"] == 1
    assert provider.translate.func is gemini.translate
    assert provider.translate.keywords == {"model": "gemini-3.8-flash", "api_key": "g-previous"}


def test_previous_version_mock_settings_keep_working(stored):
    stored.update({"provider": "2", "parallel_requests": "5", "target_lang": "6", "custom_lang": "Esperanto"})

    cfg = settings.get()

    assert cfg["provider"] == "Mock (Test)"
    assert cfg["parallel"] == 5
    assert cfg["lang"] == "Esperanto"
    assert settings.get_provider(cfg).translate.func is mock.translate


def test_fresh_install_uses_new_defaults(stored):
    cfg = settings.get()

    assert cfg["provider"] == "OpenAI"
    assert cfg["model"] == "gpt-5.6-luna"
    assert cfg["lang"] == "Polish"

    stored["provider"] = "1"
    assert settings.get()["model"] == "gemini-3.8-flash"


def test_corrupted_model_values_never_yield_empty_model(stored):
    stored.update({"provider": "0", "model": "garbage"})
    assert settings.get()["model"] == "gpt-5.6-luna"

    stored.update({"provider": "1", "gemini_model": "42"})
    assert settings.get()["model"] == "gemini-3.8-flash"
