import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api import gemini, openai
from core.providers import Provider
from core.srt import parse_srt
from core.translation import translate_subtitles

PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-5.6-luna", openai.translate),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.8-flash", gemini.translate),
}


def smoke(name, lang):
    key_variable, model_variable, default_model, translate = PROVIDERS[name]
    api_key = os.environ.get(key_variable)
    if not api_key:
        print(f"{name}: skipped, {key_variable} is not set")
        return
    model = os.environ.get(model_variable, default_model)
    provider = Provider(name, model, lambda instructions, items: translate(instructions, items, model, api_key))
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "sample.srt"
        shutil.copy(ROOT / "tests" / "data" / "sample.srt", source)
        result = translate_subtitles(str(source), lang, provider, parallel=2)
        blocks = parse_srt(result)
        assert len(blocks) == len(parse_srt(str(source)))
        print(f"{name}/{model}: translated {len(blocks)} blocks; first: {blocks[0]['lines']}")


if __name__ == "__main__":
    for provider_name in sys.argv[1:] or list(PROVIDERS):
        smoke(provider_name, os.environ.get("SMOKE_LANG", "Polish"))
