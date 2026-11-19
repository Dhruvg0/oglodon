# Security policy

Oglodon is an alpha library for learning and small-scale research. Please still report security problems privately.

## Supported versions

Only the latest release on PyPI receives fixes.

## Reporting a vulnerability

Use GitHub's private reporting: go to the **Security** tab of this repository and choose **Report a vulnerability**.
Please do not open a public issue for a security problem.

Include what you found, the version of `oglodon-transformer`, and steps to reproduce. You can expect a first reply
within about a week. This is a one-person project, so fixes may take longer.

## Scope notes

* Loading a checkpoint reads `model.safetensors` (no pickle) plus JSON files. Do not load files from sources you do
  not trust all the same.
* `oglodon verify --hf <name>` downloads a model through the Hugging Face libraries; their own security notes apply.
