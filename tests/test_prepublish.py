import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prepublish_check.py"


def load():
    spec = importlib.util.spec_from_file_location("prepublish_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metadata_checks_catch_placeholder_author_and_pass_the_rest():
    mod = load()
    meta = {
        "project": {
            "version": "0.4.1",
            "authors": [{"name": "Oglodon contributors"}],
            "urls": {"Homepage": "https://github.com/Dhruvg0/Oglodon"},
        }
    }
    res = {name: status for name, status, _ in mod.check_metadata(meta)}
    assert res["authors are real"] == "FAIL"
    assert res["version matches __init__"] == "PASS"
    meta["project"]["authors"] = [{"name": "Real Person"}]
    assert {n: s for n, s, _ in mod.check_metadata(meta)}["authors are real"] == "PASS"
    meta["project"]["version"] = "9.9.9"
    out = {n: s for n, s, _ in mod.check_metadata(meta)}
    assert (
        out["version matches __init__"] == "FAIL" and out["changelog has an entry for this version"] == "FAIL"
    )


def test_unrequested_checks_are_skipped_not_passed(capsys):
    mod = load()
    mod.main([])
    text = capsys.readouterr().out
    assert "SKIP  build and twine check" in text and "SKIP  docs build" in text
