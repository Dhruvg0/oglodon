# Publishing checklist (manual)

Oglodon never publishes by itself. A release happens only when you push a version tag (or start the workflow by hand).

## One-time setup

1. **Check the name is free**: `https://pypi.org/project/oglodon-transformer/` should return 404 (the import and command
   name `oglodon` can stay even if you change the distribution name in `pyproject.toml`). Also search GitHub, a
   trademark database and domain names for "Oglodon" before you promote it.
2. In `pyproject.toml` set your real name and e-mail in `authors`, check the repository URLs, and keep `version` in sync
   with `src/oglodon/__init__.py`.
3. **Create a PyPI account and a pending trusted publisher** (PyPI: *Your account, Publishing, Add a new pending
   publisher*): project name `oglodon-transformer`, owner/repository of your GitHub repo, workflow file `release.yml`,
   environment `pypi`. Do the same on TestPyPI with environment `testpypi`. Trusted publishing means no API token is
   stored anywhere.
4. In the GitHub repository create the two environments `pypi` and `testpypi` (Settings, Environments); you can require
   a manual approval on `pypi`.

## Every release

1. Update `CHANGELOG.md`, bump `version` in `pyproject.toml` and `src/oglodon/__init__.py`.
2. `ruff check . && ruff format --check . && mypy src && pytest` must pass. Run `pytest -m slow` once and, if you
   have a GPU, `bash scripts/gpu_check.sh` ([Performance and scaling](performance-and-scaling.md)).
3. Run `python scripts/prepublish_check.py --build --docs --check-name`. It checks version agreement, a changelog entry,
   that `authors` is not the placeholder, the repository URL, the build, `twine check`, wheel/sdist contents, the strict
   docs build and PyPI name availability. It exits non-zero while anything blocks the release; `SKIP` means not checked.
4. **Dry run on TestPyPI**: GitHub, Actions, *release*, Run workflow, target `testpypi`. Then in a clean virtualenv:
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple oglodon-transformer` and run
   [Getting started](getting-started.md).
5. **Release**: `git tag v0.4.0 && git push origin v0.4.0`. The workflow checks that the tag equals the package version,
   builds, runs `twine check`, and publishes to PyPI through trusted publishing.
6. Create a GitHub release from the tag with the changelog entry.

The workflow file is `.github/workflows/release.yml`. It has not been executed by the author, so treat the first
TestPyPI run as its test.

## Docs site

`mkdocs.yml` is included. `pip install -e ".[docs]" && mkdocs serve` previews it; `mkdocs gh-deploy` publishes to GitHub
Pages.
