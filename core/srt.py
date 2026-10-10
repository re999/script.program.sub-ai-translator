import os
import re

# Blank lines can occur inside cue text; only another cue header ends it.
SRT_REGEX = (
    r"(?m)^(\d+)[ \t]*\n([\d:,]+)[ \t]+-->[ \t]+([\d:,]+)[ \t]*(?:\n|\Z)"
    r"([\s\S]*?)(?=^\d+[ \t]*\n[\d:,]+[ \t]+-->[ \t]+[\d:,]+[ \t]*(?:\n|\Z)|\Z)"
)

def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()
    return [
        {
            "index": int(m[0]),
            "start": m[1].strip(),
            "end": m[2].strip(),
            "lines": [line for line in m[3].strip().splitlines() if line.strip()]
        }
        for m in re.findall(SRT_REGEX, content)
    ]

def write_srt(blocks, path):
    lines = [
        f"{block['index']}\n{block['start']} --> {block['end']}\n" + "\n".join(block["lines"])
        for block in blocks
    ]
    write_atomically("\n\n".join(lines), path)

def write_atomically(content, path):
    temp_path = f"{path}.part"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def group_blocks(blocks, size):
    return [blocks[i:i + size] for i in range(0, len(blocks), size)]
