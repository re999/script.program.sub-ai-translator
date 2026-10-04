import json
import pytest
from core.srt import parse_srt, group_blocks, write_srt
from core.prompt import build_instructions, build_items, serialize_items, TRANSLATION_SCHEMA
from core.translation import translated_path, format_ids
from pathlib import Path

def test_parse_srt_with_valid_blocks(tmp_path):
    # Given: an .srt file with three valid subtitle blocks
    srt_content = (
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWorld\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nAgain\n"
    )
    file_path = tmp_path / "test.srt"
    file_path.write_text(srt_content, encoding="utf-8")

    # When: parsing the file
    result = parse_srt(str(file_path))

    # Then: result should contain 3 blocks with correct structure
    assert len(result) == 3
    assert result[0]["index"] == 1
    assert result[1]["lines"] == ["World"]

def test_grouping_function():
    # Given: a list of 10 items
    items = list(range(10))

    # When: grouped by 3
    result = group_blocks(items, 3)

    # Then: result should be grouped into 4 groups (3 + 3 + 3 + 1)
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

def test_build_items_keeps_ids_and_separate_lines():
    # Given: indexed subtitle blocks
    blocks = [(5, {"lines": ["Hello", "world"]}), (12, {"lines": ["Another block"]})]

    # When: building request items
    items = build_items(blocks)

    # Then: every block becomes one item with an integer id and a list of lines
    assert items == [{"id": 5, "lines": ["Hello", "world"]}, {"id": 12, "lines": ["Another block"]}]

def test_serialize_items_produces_items_payload():
    # Given: request items with non-ASCII text
    items = [{"id": 1, "lines": ["Cześć"]}]

    # When: serializing
    text = serialize_items(items)

    # Then: the payload is JSON with an items list and readable characters
    assert json.loads(text) == {"items": items}
    assert "Cześć" in text

def test_build_instructions_mentions_language_and_rules():
    # When: building instructions
    instructions = build_instructions("Polish")

    # Then: the target language and structural rules are present
    assert "Polish" in instructions
    assert "same number of lines" in instructions
    assert "Do not merge or split" in instructions

def test_translation_schema_is_strict():
    # Then: schema requires ids and lines and forbids extra properties at every level
    item_schema = TRANSLATION_SCHEMA["properties"]["translations"]["items"]
    assert TRANSLATION_SCHEMA["required"] == ["translations"]
    assert TRANSLATION_SCHEMA["additionalProperties"] is False
    assert item_schema["required"] == ["id", "lines"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["id"] == {"type": "integer"}

def test_write_srt_creates_valid_file(tmp_path):
    # Given: subtitle blocks
    blocks = [
        {"index": 1, "start": "00:00:01,000", "end": "00:00:02,000", "lines": ["Hi"]},
        {"index": 2, "start": "00:00:03,000", "end": "00:00:04,000", "lines": ["There"]},
    ]
    out_file = tmp_path / "test.srt"

    # When: writing the file
    write_srt(blocks, str(out_file))

    # Then: the content should match the expected .srt structure
    content = out_file.read_text(encoding="utf-8").strip()
    expected = """1
00:00:01,000 --> 00:00:02,000
Hi

2
00:00:03,000 --> 00:00:04,000
There"""
    assert content == expected
    assert [p.name for p in tmp_path.iterdir()] == ["test.srt"]

def test_translated_path_never_equals_source_path():
    # Then: upper-case extensions and ".srt" in directory names do not break the output name
    assert translated_path("/media/Movie.SRT", "Polish") == "/media/Movie.polish.translated.srt"
    assert translated_path("/media/a.srt.dir/movie.srt", "PL") == "/media/a.srt.dir/movie.pl.translated.srt"

def test_format_ids_compacts_ranges():
    assert format_ids([4, 0, 1, 2, 7, 8]) == "0-2, 4, 7-8"
    assert format_ids([]) == ""
