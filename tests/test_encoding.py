"""Text files must be read and written as UTF-8, never the platform default.

On Windows the default is often cp1252, which fails on ordinary UTF-8 text such as curly quotes.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "oglodon"
BARE = re.compile(r"\.read_text\(\s*\)")
WRITE = re.compile(r"\.write_text\(((?:[^()]|\([^()]*\))*)\)")


def test_every_text_read_and_write_names_its_encoding():
    bad = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if BARE.search(text):
            bad.append(f"{path.name}: read_text() without encoding")
        for m in WRITE.finditer(text):
            if "encoding" not in m.group(1):
                bad.append(f"{path.name}: write_text without encoding")
    assert not bad, bad
