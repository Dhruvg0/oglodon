#!/usr/bin/env python
"""Checks to run before publishing a release: `python scripts/prepublish_check.py [--build] [--docs] [--check-name]`.

Exit code 0 only when nothing blocks the release. Every check prints PASS, FAIL or SKIP; a check that was not
requested is SKIP and never counted as passed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_AUTHORS = {"oglodon contributors", "your name", ""}


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def check_metadata(meta: dict) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    proj = meta["project"]
    version = proj["version"]
    init = (ROOT / "src" / "oglodon" / "__init__.py").read_text()
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    out.append(
        (
            "version matches __init__",
            "PASS" if m and m.group(1) == version else "FAIL",
            f"pyproject {version}, __init__ {m.group(1) if m else 'missing'}",
        )
    )
    changelog = (ROOT / "CHANGELOG.md").read_text()
    ok = re.search(rf"^## {re.escape(version)}\b", changelog, re.M) is not None
    out.append(
        ("changelog has an entry for this version", "PASS" if ok else "FAIL", f"looking for '## {version}'")
    )
    authors = proj.get("authors", [])
    bad = [a for a in authors if a.get("name", "").strip().lower() in PLACEHOLDER_AUTHORS]
    out.append(
        (
            "authors are real",
            "FAIL" if bad or not authors else "PASS",
            "set your name in [project].authors"
            if bad or not authors
            else ", ".join(a["name"] for a in authors),
        )
    )
    urls = proj.get("urls", {})
    out.append(
        (
            "repository URL set",
            "PASS" if any("github.com" in u and "YOUR" not in u.upper() for u in urls.values()) else "FAIL",
            json.dumps(urls),
        )
    )
    out.append(("license file", "PASS" if (ROOT / "LICENSE").exists() else "FAIL", "LICENSE"))
    return out


def check_name(name: str) -> tuple[str, str, str]:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            info = json.load(r)["info"]
        return ("PyPI name is free", "FAIL", f"{name} already exists on PyPI (latest {info['version']})")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ("PyPI name is free", "PASS", f"{name} not found on PyPI (still search for trademarks)")
        return ("PyPI name is free", "FAIL", f"unexpected HTTP {e.code}")
    except OSError as e:
        return ("PyPI name is free", "FAIL", f"could not reach PyPI: {e}")


def check_build(version: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    dist = ROOT / "dist"
    for old in dist.glob("*") if dist.exists() else []:
        old.unlink()
    code, log = _run([sys.executable, "-m", "build"])
    out.append(("build sdist + wheel", "PASS" if code == 0 else "FAIL", log[-300:] if code else "ok"))
    if code:
        return out
    files = sorted(dist.glob("*"))
    code, log = _run([sys.executable, "-m", "twine", "check", *map(str, files)])
    out.append(("twine check", "PASS" if code == 0 else "FAIL", log[-300:]))
    wheel = next(dist.glob("*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    out.append(
        (
            "wheel has the package and no tests",
            "PASS"
            if any(n.endswith("oglodon/cli.py") for n in names) and not any("tests/" in n for n in names)
            else "FAIL",
            f"{len(names)} files",
        )
    )
    sdist = next(dist.glob("*.tar.gz"))
    with tarfile.open(sdist) as tf:
        snames = tf.getnames()
    junk = [n for n in snames if re.search(r"(__pycache__|\.pyc$|/site/|\.git/|\.env)", n)]
    out.append(
        ("sdist has no junk", "FAIL" if junk else "PASS", ", ".join(junk[:5]) or f"{len(snames)} files")
    )
    return out


def check_docs() -> tuple[str, str, str]:
    code, log = _run([sys.executable, "-m", "mkdocs", "build", "--strict", "-q", "-d", "/tmp/_oglodon_site"])
    return ("docs build (strict)", "PASS" if code == 0 else "FAIL", log[-300:] if code else "ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build and inspect sdist/wheel (needs build, twine)")
    ap.add_argument("--docs", action="store_true", help="strict mkdocs build (needs the docs extra)")
    ap.add_argument("--check-name", action="store_true", help="ask PyPI whether the project name is free")
    args = ap.parse_args(argv)
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    results = check_metadata(meta)
    results.append(
        check_name(meta["project"]["name"])
        if args.check_name
        else ("PyPI name is free", "SKIP", "pass --check-name")
    )
    if args.build:
        results += check_build(meta["project"]["version"])
    else:
        results.append(("build and twine check", "SKIP", "pass --build"))
    results.append(check_docs() if args.docs else ("docs build (strict)", "SKIP", "pass --docs"))
    for name, status, detail in results:
        print(f"{status:5} {name}: {detail}")
    failed = [n for n, s, _ in results if s == "FAIL"]
    print(
        f"\n{len(failed)} blocking problem(s)"
        if failed
        else "\nnothing blocks the release (SKIP lines were not checked)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
