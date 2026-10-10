from .config import BATCH_SIZE
from .models import FREE
from .prompt import build_instructions, build_items, serialize_items
from .srt import parse_srt, group_blocks

OUTPUT_TOKENS_PER_INPUT_TOKEN = 1.2

def estimate_cost(path, lang, price=FREE):
    blocks = parse_srt(path)
    batches = group_blocks(build_items(enumerate(blocks)), BATCH_SIZE)
    instructions = build_instructions(lang)

    prompts = [instructions + "\n" + serialize_items(batch) for batch in batches]
    chars = sum(len(p) for p in prompts)
    tokens = chars // 4
    output_tokens = round(tokens * OUTPUT_TOKENS_PER_INPUT_TOKEN)
    usd = round((tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000, 4)

    return {
        "chars": chars,
        "tokens": tokens,
        "output_tokens": output_tokens,
        "usd": usd,
        "prompts": prompts
    }
