# mjlab Agent Guide

This repository uses a project-local Codex harness.

- Codex config: `.codex/config.toml`
- Codex role configs: `.codex/agents/*.toml`
- Project-local ECC skills: `.agents/skills/`

## Workflow

- Use `uv run`, not bare `python`.
- Prefer targeted iteration commands:
  - `uv run pytest tests/<file>.py`
  - `uv run ty check`
  - `uv run pyright`
- Before commit, run `make check`.
- Before PR or merge, run `make test`.
- When changing user-facing behavior, update `docs/source/changelog.rst`.

## Collaboration

- Prefer the `explorer` role for read-only evidence gathering.
- Prefer the `reviewer` role after non-trivial changes.
- Prefer the `docs_researcher` role for API and release-note verification.
- Keep changes focused and avoid broad repo churn outside the task.

## Notes

- `CLAUDE.md` contains the concrete local development commands and commit/PR conventions.
- Add new project-specific Codex skills under `.agents/skills/`.
- Prefer skills over new legacy command surfaces when adding workflow helpers.
