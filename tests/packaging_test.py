import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_modules_on_service_sys_path_do_not_shadow_stdlib():
    names = {path.stem for folder in ("api", "core") for path in (ROOT / folder).glob("*.py")}
    assert names.isdisjoint(sys.stdlib_module_names)
