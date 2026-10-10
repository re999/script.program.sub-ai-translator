import math
from typing import NamedTuple


class Price(NamedTuple):
    input_per_million: float
    output_per_million: float


FREE = Price(0.0, 0.0)

MODEL_PRICES = {
    "gpt-5.6-luna": Price(0.20, 1.20),
    "gpt-5.6-terra": Price(2.00, 12.00),
    "gemini-3.8-flash": Price(0.75, 3.75),
    "gemini-3.5-flash-lite": Price(0.30, 2.50),
}
LEGACY_DEFAULT_PRICE_PER_1000_TOKENS = 0.001

OPENAI_MODEL_CHOICES = (
    ("gpt-3.5-turbo", "gpt-5.6-luna"),
    ("gpt-4", "gpt-5.6-terra"),
    ("gpt-4-turbo", "gpt-5.6-terra"),
    ("gpt-5.6-luna", "gpt-5.6-luna"),
    ("gpt-5.6-terra", "gpt-5.6-terra"),
)
OPENAI_DEFAULT_INDEX = 3

GEMINI_MODEL_CHOICES = (
    ("gemini-1.5-flash-latest", "gemini-3.8-flash"),
    ("gemini-1.5-pro-latest", "gemini-3.8-flash"),
    ("gemini-2.0-flash", "gemini-3.8-flash"),
    ("Auto", "gemini-3.8-flash"),
    ("gemini-3.8-flash", "gemini-3.8-flash"),
    ("gemini-3.5-flash-lite", "gemini-3.5-flash-lite"),
)
GEMINI_DEFAULT_INDEX = 5


def resolve_model(choices, saved_index, default_index):
    index = saved_index if isinstance(saved_index, int) and 0 <= saved_index < len(choices) else default_index
    return choices[index][1]


def resolve_openai_model(saved_index):
    return resolve_model(OPENAI_MODEL_CHOICES, saved_index, OPENAI_DEFAULT_INDEX)


def resolve_gemini_model(saved_index):
    return resolve_model(GEMINI_MODEL_CHOICES, saved_index, GEMINI_DEFAULT_INDEX)


def model_price(model):
    return MODEL_PRICES.get(model, FREE)


def price_override(price_per_1000_tokens):
    try:
        per_1000 = float(price_per_1000_tokens)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(per_1000) or per_1000 <= 0 or per_1000 == LEGACY_DEFAULT_PRICE_PER_1000_TOKENS:
        return None
    return Price(per_1000 * 1000, per_1000 * 1000)
