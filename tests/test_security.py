import json
import re
from pathlib import Path


def test_no_google_keys_in_source_or_notebook():
    root = Path(__file__).parents[1]
    pattern = re.compile(r"AIza" + r"[A-Za-z0-9_-]{35}")
    files = list((root / "src").rglob("*.py")) + list((root / "examples").glob("*.py"))
    files += list((root / "notebooks").glob("*.ipynb"))
    assert not [str(p.relative_to(root)) for p in files if pattern.search(p.read_text())]
    notebook = json.loads((root / "notebooks/original_rag.ipynb").read_text())
    assert all(not c.get("outputs") for c in notebook["cells"])
    assert all(c.get("execution_count") is None for c in notebook["cells"])


def test_env_is_ignored():
    ignore = (Path(__file__).parents[1] / ".gitignore").read_text().splitlines()
    assert ".env" in ignore
    assert "!.env.example" in ignore
