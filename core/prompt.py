import json

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "lines": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "lines"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def build_instructions(lang):
    return (
        f"Translate each subtitle item into {lang}.\n"
        "- Preserve the meaning and the tone of the dialogue.\n"
        "- Do not add commentary, notes or explanations.\n"
        "- Return exactly one translation for every requested id.\n"
        "- Return the same number of lines as the source item.\n"
        "- Do not merge or split subtitle items."
    )


def build_items(indexed_blocks):
    return [{"id": index, "lines": list(block["lines"])} for index, block in indexed_blocks]


def serialize_items(items):
    return json.dumps({"items": items}, ensure_ascii=False)
