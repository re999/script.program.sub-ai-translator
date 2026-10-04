from core.validation import valid_translations

ITEMS = [
    {"id": 3, "lines": ["One", "Two"]},
    {"id": 4, "lines": ["Three"]},
]

def payload(*entries):
    return {"translations": list(entries)}

def test_complete_payload_is_accepted():
    result = valid_translations(ITEMS, payload({"id": 3, "lines": ["Jeden", "Dwa"]}, {"id": 4, "lines": ["Trzy"]}))
    assert result == {3: ["Jeden", "Dwa"], 4: ["Trzy"]}

def test_missing_item_is_unresolved():
    assert valid_translations(ITEMS, payload({"id": 4, "lines": ["Trzy"]})) == {4: ["Trzy"]}

def test_wrong_line_count_is_rejected():
    assert valid_translations(ITEMS, payload({"id": 3, "lines": ["Jeden Dwa"]}, {"id": 4, "lines": ["Trzy"]})) == {4: ["Trzy"]}

def test_duplicate_id_is_rejected():
    result = valid_translations(ITEMS, payload(
        {"id": 3, "lines": ["Jeden", "Dwa"]},
        {"id": 3, "lines": ["Raz", "Dwa"]},
        {"id": 4, "lines": ["Trzy"]},
    ))
    assert result == {4: ["Trzy"]}

def test_unrequested_id_is_ignored():
    result = valid_translations(ITEMS, payload({"id": 99, "lines": ["X"]}, {"id": 4, "lines": ["Trzy"]}))
    assert result == {4: ["Trzy"]}

def test_blank_or_multiline_strings_are_rejected():
    result = valid_translations(ITEMS, payload({"id": 3, "lines": ["Jeden", "  "]}, {"id": 4, "lines": ["Trzy\nCztery"]}))
    assert result == {}

def test_invalid_structures_are_rejected():
    assert valid_translations(ITEMS, None) == {}
    assert valid_translations(ITEMS, ["not", "an", "object"]) == {}
    assert valid_translations(ITEMS, {"translations": "nope"}) == {}
    assert valid_translations(ITEMS, payload({"id": "4", "lines": ["Trzy"]}, {"id": True, "lines": ["Trzy"]})) == {}
    assert valid_translations(ITEMS, payload({"id": 4, "lines": "Trzy"}, {"id": 3, "lines": [1, 2]})) == {}
    assert valid_translations(ITEMS, payload({"id": 4})) == {}

def test_blank_source_line_may_stay_blank():
    items = [{"id": 1, "lines": ["Hello", " "]}]
    assert valid_translations(items, payload({"id": 1, "lines": ["Cześć", ""]})) == {1: ["Cześć", ""]}
