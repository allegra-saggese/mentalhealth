# CLAUDE.md -- Large-scale livestock, slaughterhouse work, and mental health in the US

**Project:** `mentalhealth` (GitHub: allegra-saggese/mentalhealth, **public**) | data root `~/Dropbox/Mental`
**Institution:** UC Santa Cruz (Economics) <!-- assumed from account email; confirm -->
**Target:** top field journal, Nature or Science. Everything reported must survive a hostile referee.
**Branch:** main

County-year panel (rural US counties, about 2000-2023; analysis window 2010-2015) linking large livestock operations (CAFOs) and slaughterhouses (FSIS) to mental-health, mortality and crime outcomes. Causal identification is the goal, not description. Pipeline detail: [README.md](README.md). Estimator equations and definitions: [METHODS.md](METHODS.md).

**Phases:** (1) QA and verify data cleaning, (2) summary visualizations, (3) ML framework for variable selection. The work spans five tasks: variable selection and causal structure; model selection against theory and data; graphics and tables; interpretation of regressions; robustness.

---

## Scope Discipline

**Do exactly what was asked, nothing adjacent.** No extra README files, helper scripts, `.gitignore` edits or tooling that were not requested. If an addition looks valuable, list it at the end as a suggestion. Before adding anything not named in the request, ask: one line is cheaper than a revert.

---

## Working Agreement

- **Plan first.** Any task that touches more than 3 files, a shared module, estimation, or a deliverable's substance gets a plan saved to `quality_reports/plans/` (local and git-ignored; survives compaction). Decisions that shape a specification are recorded in the decision register in `code-audit/plans/` (committed). Small, well-specified edits go straight through.
- **After approval: contractor mode.** Execute, verify by running, report. Come back only for: a design fork, any choice touching an estimand, sample, control set, inference or interpretive language, a change to a shared module, anything destructive, anything outward-facing (push, PR, email).
- **Onboarding mode (first sessions, until the user says to stop).** Also check in at plan approval, after each stage runs, and before any re-estimation, with a short "what the workflow just did and why" note.
- **Learn from corrections.** Every correction becomes a `[LEARN:category] wrong -> right` line in [MEMORY.md](MEMORY.md) in the same turn. Machine-local detail goes to native auto memory, never into this public repo.
- **Plain language.** Define terms before using them (first expansion, first entry, the control set, leads). No private shorthand. Show the equation when asking for a method choice.
- **Results before interpretation.** Show results first; flag interpretation separately and only when asked.
- **Never delete old work.** Narrowing is reporting-level (`REPORT_*` in `script4_treatment.py`); keep estimating and archive by moving.
- **Never push.** No commit, push, PR or merge without an explicit request. Show diffs before re-running estimation. Ask before changing shared files.
- **Verify before claiming.** Run it; test syntax that could silently do nothing (e.g. a pyfixest term that is ignored); rebuild checks that reproduce an earlier result exactly.
- **No fabricated citations.** Search and verify; state what could not be verified.
- **Write the answer in text before the wrap-up box.** The box ends the turn.

---

## Core Principles

- **Single source of truth.** Treatments, sample, outcomes and controls are defined once, in `script4_treatment.py`; estimator math in `METHODS.md`; shared code in `did_tools.py`, `designs.py`, `did_cs.py`, `functions.py`, `packages.py`; figure style in `figstyle.py`. Exhibit scripts read result files; they do not re-estimate.
- **Pre-commit before looking.** Specification choices are written down (decision register, D-numbered) before results are read. A choice made after seeing results is labelled as such.
- **Quality gates.** Nothing is committed below 80/100, nothing is cited below 90. See [`quality-gates.md`](.claude/rules/quality-gates.md).
- **Two documentation tracks.** Internal: data provenance, cleaning decisions, methodological choices (`code-audit/`). External: clean, commented, reviewer-ready scripts that pass replication checks.

**Verification and writing discipline** (copied from the upstream workflow, retargeted to Python): [`verification-ladder.md`](.claude/references/verification-ladder.md), [`research-agent-laws.md`](.claude/references/research-agent-laws.md), [`orchestrator-protocol.md`](.claude/rules/orchestrator-protocol.md), [`writing-with-ai.md`](.claude/rules/writing-with-ai.md).

---

## Folder Structure

```
mentalhealth/                      # code only; data lives on Dropbox (db_base in packages.py)
├── CLAUDE.md, MEMORY.md           # this file; lessons learned (committed)
├── .claude/                       # rules, skills, agents, hooks, settings
├── .githooks/                     # pre-commit gate (install: ./scripts/install-hooks.sh)
├── script0a ... script0f-*.py     # Stage 0: data in (ingestion, cleaning, the panel build in script0f)
├── script1a, script2a ... script2f-*.py   # QA, descriptive figures, maps (being consolidated into script1a-1d)
├── script4_treatment.py, script4*-*.py   # Stage 4: design-based estimates and exhibits
├── packages.py, functions.py, did_tools.py, designs.py, did_cs.py, estimators.py
├── METHODS.md, README.md          # estimator definitions; run order and data availability
├── tests/                         # compare_outputs.py, test_precommit.sh
├── code-audit/plans/              # decision register, run-state notes, QA notes (internal)
├── quality_reports/               # plans, specs, session logs (local only, git-ignored)
├── templates/                     # decision record, session log, passport, journal profile
├── first-stage-testing/, z-archive/   # superseded work, kept, never run
└── qa-scripts/, fsis-scripts/, test-dfs/
```

Naming: scripts `script{stage}{letter}-{description}.py`, no gaps in letters; outputs `YYYY-MM-DD_{descriptor}.csv`; latest file via `latest_file_glob()` in `functions.py`.

---

## Commands

```bash
export PYTHONHASHSEED=0          # required: pyfixest SEs differ ~2e-5 across hash seeds
# Reported results (run order and timings: README "Running the reported results")
python3 script4a-design-estimates.py        # TWFE, event study, Callaway-Sant'Anna
python3 script4n-regression-tables.py && python3 script4p-full-appendix.py
# Prove a refactor changed no result
python3 tests/compare_outputs.py <baseline_root> <live_root>
# Install the commit gate (once per clone)
./scripts/install-hooks.sh
```

Credentials come from the environment only: `USDA_NASS_API_KEY`, `HUD_API_TOKEN`. Data are never committed.

---

## Non-negotiables

- **Specification (reported set):** county + year fixed effects; standard errors clustered by state; treatments T1, T3, T4, first expansion (E2); estimators TWFE and Callaway-Sant'Anna covariate-adjusted. Everything else is still estimated and appears in robustness exhibits.
- **Control set:** the 8 full-coverage controls, pre-committed (decisions D-16, D-31). Called "the control set".
- **Zero rule (D-34):** missing `cafo_*` counts are 0 from 2002; 2000-01 stay missing.
- **Paths:** only through `packages.py`. **Seeds:** `PYTHONHASHSEED=0`; `numpy.random.default_rng(seed)` once at the top of stochastic code.
- **Tolerances:** outputs equal to rtol 1e-6, atol 1e-8; across hash seeds, SEs to 1e-4.
- **Before citing any output,** check `code-audit/plans/2026-10-17_MODEL-RUN-STATE.md` for which script ran on which panel.
- **Figures and tables:** [`figure-and-table-standards.md`](.claude/rules/figure-and-table-standards.md).

---

## Quality Thresholds

| Score | Checkpoint | Meaning |
|-------|------------|---------|
| 80 | Commit | Good enough to save |
| 90 | PR / cite | Ready to share or quote in the paper |
| 95 | Excellence | Aspirational |

Enforced by `/commit` and by `.githooks/pre-commit` (bypass only with `SKIP_QUALITY_GATE=1` or `--no-verify`, reason in the commit message).

---

## Skills Quick Reference

- **Data and estimation:** `/data-analysis` `/diagnose` `/power-analysis` `/challenge` (specification grid) `/differential-audit` `/blast-radius` `/review-py`
- **Project-specific:** `/robustness-suite` `/spec-register` `/figure-check` `/variable-selection-protocol`
- **Reproducibility:** `/audit-reproducibility` `/replication-package` `/capture-environment` `/verify-artifact`
- **Paper and review:** `/review-paper` (`--peer`) `/seven-pass-review` `/respond-to-referees` `/verify-claims` `/credible-claims` `/submission-disclosures` `/humanize`
- **Research:** `/interview-me` `/research-ideation` `/lit-review` `/preregister`
- **Workflow:** `/commit` `/learn` `/checkpoint` `/context-status` `/issues` `/deep-audit` `/new-skill`

Full definitions are in `.claude/skills/*/SKILL.md`.

---

## Current Project State

| Item | State |
|------|-------|
| Stage 0-1 (ingestion, panel) | Run and verified; current panel `Data/merged/2026-10-12_panel.csv`; rebuilt from the merged stage-0 scripts 2026-10-24 in a scratch folder with 0 differing cells |
| Stage 4 (design-based estimates) | Reported set fixed 2026-10-15; run state in `code-audit/plans/2026-10-17_MODEL-RUN-STATE.md` |
| Refactor 2026-10-22 | Shared modules and `METHODS.md` added; output-equivalent (71 files identical under `PYTHONHASHSEED=0`) |
| Phase 3 (ML selection) | Specified, not built (B-series, `2026-09-27_MODEL-NAMING.md`) |
| Open blockers | Full-panel TWFE vs 2010-2020 window (A21); pre-trend failures for deaths of despair and violent crime (A22); API key in git history (6bdb09f, c1fa465) must be revoked; no LICENSE (owner decision) |
