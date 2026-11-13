"""The documentation must not drift from the code: documented flags exist, snippets parse, configs load."""

import ast
import re
from pathlib import Path

import pytest

from oglodon import OglodonConfig
from oglodon.cli import main

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = sorted(DOCS.glob("*.md"))
FENCE = re.compile(r"```(\w*)\n(.*?)```", re.S)


def blocks(lang):
    for page in PAGES:
        for m in FENCE.finditer(page.read_text()):
            if m.group(1) == lang:
                yield page.name, m.group(2)


def help_text(cmd, capsys):
    with pytest.raises(SystemExit):
        main([cmd, "--help"])
    return capsys.readouterr().out


def test_every_documented_flag_exists_on_its_command(capsys):
    helps: dict[str, str] = {}
    checked = 0
    for name, code in blocks("bash"):
        joined = code.replace("\\\n", " ")
        for line in joined.splitlines():
            m = re.search(
                r"\boglodon(?: -m oglodon)?\s+(train|generate|eval|compare|verify|validate|info)\b(.*)", line
            )
            if (
                not m
                or "-m oglodon" not in line
                and not line.lstrip().startswith(("oglodon", "echo", "torchrun"))
            ):
                continue
            cmd, rest = m.group(1), m.group(2).split("#")[0]
            helps.setdefault(cmd, help_text(cmd, capsys))
            for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", rest):
                assert flag in helps[cmd], f"{name}: `oglodon {cmd}` has no {flag}"
                checked += 1
    assert checked > 25  # the scan really found the documented commands


def test_cli_reference_documents_every_flag_of_every_command(capsys):
    cli = (DOCS / "cli.md").read_text()
    for cmd in ("train", "generate", "eval", "quantize", "compare", "verify", "validate"):
        for flag in set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", help_text(cmd, capsys))) - {"--help"}:
            assert flag in cli, f"docs/cli.md does not mention {flag} (oglodon {cmd})"


def test_python_snippets_are_valid_syntax():
    n = 0
    for name, code in blocks("python"):
        try:
            ast.parse(code)
        except SyntaxError as e:  # pragma: no cover
            pytest.fail(f"{name}: {e}\n{code}")
        n += 1
    assert n >= 10


def test_every_page_is_in_the_navigation():
    nav = (DOCS.parent / "mkdocs.yml").read_text()
    for page in PAGES:
        assert page.name in nav, f"{page.name} is not in mkdocs.yml"


def test_documented_configs_load():
    yaml = pytest.importorskip("yaml")
    from oglodon.cli import model_config_from
    from oglodon.workflow import TrainSettings

    seen = set()
    for name, code in blocks("yaml"):
        data = yaml.safe_load(code)
        if name == "training.md":  # the oglodon train config
            flat = {**data.get("model", {}), **data.get("data", {}), **data.get("train", {})}
            s = TrainSettings.from_dict(flat)
            OglodonConfig(vocab_size=100, **{"hidden_size": s.hidden_size, "num_heads": s.num_heads, **s.model,
                                              "num_kv_heads": s.num_kv_heads, "max_seq_len": s.block_size})  # fmt: skip
            seen.add(name)
        if name == "experiments.md":  # the oglodon compare config
            base = model_config_from(data).to_dict()
            for v in data["variants"].values():
                OglodonConfig.from_dict({**base, **v})
            assert data["baseline"] in data["variants"]
            seen.add(name)
    assert seen == {"training.md", "experiments.md"}


def test_changelog_and_version_agree():
    import oglodon

    assert f"## {oglodon.__version__}" in (DOCS.parent / "CHANGELOG.md").read_text()
