import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .config import BATCH_SIZE
from .prompt import build_instructions, build_items
from .retry import TranslationCancelled, call_with_retries
from .srt import group_blocks, parse_srt, write_srt
from .validation import valid_translations

MAX_CONTENT_ATTEMPTS = 3
CANCEL_POLL_SECONDS = 0.2


class TranslationIncomplete(Exception):
    def __init__(self, unresolved_ids):
        self.unresolved_ids = list(unresolved_ids)
        super().__init__(f"Translation incomplete, unresolved subtitle ids: {format_ids(self.unresolved_ids)}")


def format_ids(ids):
    ranges = []
    for current in sorted(ids):
        if ranges and current == ranges[-1][1] + 1:
            ranges[-1][1] = current
        else:
            ranges.append([current, current])
    return ", ".join(str(start) if start == end else f"{start}-{end}" for start, end in ranges)


def ids_of(items):
    return [item["id"] for item in items]


def translate_batch(items, instructions, provider, is_cancelled, log):
    resolved = {}
    pending = items
    for content_attempt in range(1, MAX_CONTENT_ATTEMPTS + 1):
        context = f"{provider.name}/{provider.model} ids [{format_ids(ids_of(pending))}] content attempt {content_attempt}/{MAX_CONTENT_ATTEMPTS}"
        payload = call_with_retries(
            lambda: provider.translate(instructions, pending),
            is_cancelled,
            lambda message: log(f"{context}: {message}"),
        )
        resolved.update(valid_translations(pending, payload))
        pending = [item for item in pending if item["id"] not in resolved]
        if not pending:
            return resolved
        log(f"{context}: unresolved ids [{format_ids(ids_of(pending))}]")
    raise TranslationIncomplete(ids_of(pending))


def translate_batches(batches, instructions, provider, parallel, report_progress, check_cancelled, log):
    cancelled = threading.Event()
    executor = ThreadPoolExecutor(max_workers=max(1, parallel))
    try:
        futures = [
            executor.submit(translate_batch, batch, instructions, provider, cancelled.is_set, log)
            for batch in batches
        ]
        return collect_results(futures, report_progress, check_cancelled, cancelled)
    finally:
        cancelled.set()
        executor.shutdown(wait=False)


def collect_results(futures, report_progress, check_cancelled, cancelled):
    resolved = {}
    pending = set(futures)
    while pending:
        if check_cancelled and check_cancelled():
            cancelled.set()
            raise TranslationCancelled()
        done, pending = wait(pending, timeout=CANCEL_POLL_SECONDS, return_when=FIRST_COMPLETED)
        for future in done:
            resolved.update(future.result())
        if done and report_progress:
            report_progress(len(futures) - len(pending), len(futures))
    return resolved


def merge_translations(blocks, translations):
    missing = [index for index in range(len(blocks)) if index not in translations]
    if missing:
        raise TranslationIncomplete(missing)
    return [{**block, "lines": translations[index]} for index, block in enumerate(blocks)]


def translated_path(path, lang):
    base, _ = os.path.splitext(path)
    return f"{base}.{lang.lower()}.translated.srt"


def translate_subtitles(
    path,
    lang,
    provider,
    report_progress=None,
    check_cancelled=None,
    parallel=3,
    log=print
):
    blocks = parse_srt(path)
    batches = group_blocks(build_items(enumerate(blocks)), BATCH_SIZE)
    log(f"{provider.name}/{provider.model}: translating {len(blocks)} subtitles in {len(batches)} batches")
    translations = translate_batches(
        batches, build_instructions(lang), provider, parallel,
        report_progress, check_cancelled, log
    )
    merged = merge_translations(blocks, translations)
    new_path = translated_path(path, lang)
    write_srt(merged, new_path)
    return new_path
