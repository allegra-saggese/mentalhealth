---
paths:
  - "script0*.py"
  - "fsis-scripts/**"
  - "qa-scripts/**"
  - "packages.py"
  - "functions.py"
  - "**/raw/**"
---

# Data and Secrets (public repository)

The repository `allegra-saggese/mentalhealth` is **public**. The data are public federal sources plus one FOIA file; the working copy lives on Dropbox (`db_base` in `packages.py`).

## Hard rules

1. **No data in git.** No `*.csv`, `*.xlsx`, `*.dta`, `*.parquet`, `*.rds`, `*.zip`, nothing under `Data/`. Commit code and small clearly-labelled test extracts only if explicitly approved. The pre-commit hook blocks the rest.
2. **FSIS file is not redistributable** (FOIA response). It, and anything that reproduces its rows (establishment names, addresses), never enters git, an exhibit, an issue, or a PR.
3. **Credentials come from the environment only** (`USDA_NASS_API_KEY`, `HUD_API_TOKEN`). Never in code, notebooks, logs, commit messages or plans. The pre-commit hook scans staged changes.
4. **Known hazard:** an API key was committed earlier (git history, commits 6bdb09f and c1fa465). It must be revoked by its owner; removing it from history is a separate, user-approved operation. Until revoked, treat it as public.
5. **Nothing personal in committed files:** no machine paths, usernames, emails or session transcripts. Machine-local facts go in native auto memory (`~/.claude/projects/.../memory/`), not in `MEMORY.md`.
6. **Plans, specs and session logs stay local** (`quality_reports/` is git-ignored). Decisions belong in `code-audit/plans/` and are public.

## Working with the data

- Claude writes code that runs on the data and reads printed summaries; it does not need raw files in context. Read heads, shapes and counts, not full tables.
- Raw files in `Data/raw/` are never modified. Cleaning writes new dated files to `Data/clean/`.
- Output archiving is by moving to `Data/output/_archive/<date>_<reason>/`, never deleting.
- CDC counts below 10 are suppressed at source. Do not try to back them out, and do not publish cells that could re-identify a small county-year.

## If a secret or data file is staged or pushed

Stop. Do not commit. Tell the user which file and which pattern matched. Rewriting history or revoking keys is the user's action.
