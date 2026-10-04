import os
import re

SRT_REGEX = r"(\d+)\s+([\d:,]+)\s+-->\s+([\d:,]+)\s+([\s\S]+?)(?=\n\n|\Z)"

def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        content = f.read()
    return [
        {
            "index": int(m[0]),
            "start": m[1].strip(),
            "end": m[2].strip(),
            "lines": m[3].strip().splitlines()
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
