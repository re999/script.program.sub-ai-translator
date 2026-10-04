def translate(instructions, items, model=None, api_key=None):
    return {"translations": [{"id": item["id"], "lines": list(item["lines"])} for item in items]}
