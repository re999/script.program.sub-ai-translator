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
)
GEMINI_DEFAULT_INDEX = 4


def resolve_model(choices, saved_index, default_index):
    index = saved_index if isinstance(saved_index, int) and 0 <= saved_index < len(choices) else default_index
    return choices[index][1]


def resolve_openai_model(saved_index):
    return resolve_model(OPENAI_MODEL_CHOICES, saved_index, OPENAI_DEFAULT_INDEX)


def resolve_gemini_model(saved_index):
    return resolve_model(GEMINI_MODEL_CHOICES, saved_index, GEMINI_DEFAULT_INDEX)
