from .config import BATCH_SIZE
from .prompt import build_instructions, build_items, serialize_items
from .srt import parse_srt, group_blocks

def estimate_cost(path, lang, price_per_1000=0):
    blocks = parse_srt(path)
    batches = group_blocks(build_items(enumerate(blocks)), BATCH_SIZE)
    instructions = build_instructions(lang)

    prompts = [instructions + "\n" + serialize_items(batch) for batch in batches]
    chars = sum(len(p) for p in prompts)
    tokens = chars // 4
    usd = round(2 * tokens / 1000 * price_per_1000, 4)

    return {
        "chars": chars,
        "tokens": tokens,
        "usd": usd,
        "prompts": prompts
    }
