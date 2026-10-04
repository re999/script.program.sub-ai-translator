from collections import Counter


def valid_translations(items, payload):
    entries = translation_entries(payload)
    occurrences = Counter(entry["id"] for entry in entries)
    by_id = {entry["id"]: entry for entry in entries}
    return {
        item["id"]: list(by_id[item["id"]]["lines"])
        for item in items
        if occurrences[item["id"]] == 1 and lines_match(item["lines"], by_id[item["id"]].get("lines"))
    }


def translation_entries(payload):
    translations = payload.get("translations") if isinstance(payload, dict) else None
    if not isinstance(translations, list):
        return []
    return [entry for entry in translations if isinstance(entry, dict) and is_id(entry.get("id"))]


def is_id(value):
    return isinstance(value, int) and not isinstance(value, bool)


def lines_match(source_lines, lines):
    return (
        isinstance(lines, list)
        and len(lines) == len(source_lines)
        and all(is_translated_line(line, source) for line, source in zip(lines, source_lines))
    )


def is_translated_line(line, source):
    if not isinstance(line, str) or "\n" in line or "\r" in line:
        return False
    return bool(line.strip()) or not source.strip()
