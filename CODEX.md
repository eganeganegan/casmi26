# Codex guidance

`AGENTS.md` is the authoritative repository instruction file and must be read first. Also follow
`CONTRIBUTING.md` for branches, commits, reviews, and artifact handling.

When using Codex in this repository:

- inspect the relevant code, configuration, tests, and current working-tree state before editing;
- use `rg` for search and make narrow patches that preserve unrelated work;
- prefer completing and validating an implementation over returning an untested sketch;
- start with focused tests, then run `make check` before handoff;
- run `make offline-inference` when the production path or packaged feature schema changes;
- never expose credentials or force-add ignored datasets and artifacts;
- do not push, merge, publish datasets, or submit to Kaggle unless the user explicitly asks;
- summarize changed files, validation evidence, and any remaining risks in the final handoff.

If an instruction in this file conflicts with `AGENTS.md`, follow `AGENTS.md`.
