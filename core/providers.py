from typing import Any, Callable, List, NamedTuple


class Provider(NamedTuple):
    name: str
    model: str
    translate: Callable[[str, List[dict]], Any]
