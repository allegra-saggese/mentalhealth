---
paths:
  - "script*.py"
  - "*.py"
---

# Inference & Robustness (multiple testing + researcher degrees of freedom)

Advisory standards for the inference choices that decide whether a result survives a sharp referee — consolidated so they don't live only as a one-line objection. Applies to empirical analysis scripts; the manuscript-side check is in [`/review-paper`](../skills/review-paper/SKILL.md) (dimension 3) and the design-side commitment in [`/preregister`](../skills/preregister/SKILL.md).

## Project specifics (mentalhealth)

- **Reported inference:** county + year fixed effects, standard errors clustered by state. About 50 clusters: report a wild-cluster bootstrap p-value as a robustness check for the headline estimates.
- **Outcomes:** 7 (3 mental-health/mortality, 4 crime). Declare the family before looking and report unadjusted and adjusted values. Python tools: `statsmodels.stats.multitest.multipletests` (Holm, Benjamini-Hochberg); Romano-Wolf by resampling, implemented once in a shared helper.
- **Specification sweep:** the control set was pre-committed (D-16, D-31). The sweep over defensible control sets (`script4y`) documents sensitivity; it does not choose the set.
- **Leave-one-state-out** and drop-top-influential-county checks for each headline estimate.
- **Pre-trends:** report the pre-trend test and the Rambachan-Roth sensitivity (`script4w`) next to every event study. Standing blockers: A21 (full-panel TWFE null vs 2010-2020 window significant, window not yet justified) and A22 (pre-trends fail for deaths of despair and violent crime). No headline claim on those outcomes while A22 is open.
- **Post-selection inference (Phase 3, ML variable selection):** coefficients on variables chosen by lasso/ridge/tree methods are not valid with ordinary SEs. Use sample splitting (select on one fold, estimate on the other), double/debiased machine learning, or post-double-selection; pre-specify the candidate set and cross-validation scheme before running; report selection stability across folds and seeds.

## Multiple hypothesis testing

When a paper tests **many hypotheses** (several outcomes, subgroups, treatment arms, or specifications), unadjusted p-values overstate significance. Decide the correction **by what you control and pre-register the family**:

- **Family-wise error rate (FWER)** — control the probability of *any* false rejection. Use when even one false positive is costly (a headline claim).
  - **Romano–Wolf** stepdown (resampling, exploits cross-equation dependence — far less conservative than Bonferroni; no maintained pyfixest implementation, so write it once in a shared helper and test it against R `wildrwolf` on a toy case) is the modern default for a small family.
  - Holm–Bonferroni as a distribution-free fallback; plain Bonferroni only for a tiny family.
- **False discovery rate (FDR)** — control the *expected share* of false rejections among rejections. Use for **many** hypotheses where some false positives are acceptable (screening, heterogeneity scans).
  - Benjamini–Hochberg; **Anderson (2008) sharpened two-stage q-values** is the standard in applied micro (implement once in a shared helper).
- **Pre-register the family and the correction** (the unit of correction is a researcher degree of freedom). Report both unadjusted and adjusted; never pick the family that makes the result survive.

## Researcher degrees of freedom / specification robustness

A single specification is a point on a garden of forking paths. Make the robustness explicit:

- **Show the specification is not cherry-picked** — a specification/multiverse curve (sweep the defensible covariate sets, sample restrictions, functional forms; report the distribution of estimates, not one).
- **Leave-one-out / influential observations** — confirm the result isn't driven by a few units or one cluster.
- **Inference robustness** — alternative clustering levels, wild-cluster bootstrap with few clusters (`fwildclusterboot`/`boottest`), randomization inference where the design supports it.

## Reporting

- State the **family** and the **correction method** up front; report unadjusted *and* adjusted p-values/q-values.
- A robustness check that only ever confirms the headline is theatre — report the spec where the result *weakens*, and interpret it.

## Cross-references
- [`.claude/skills/review-paper/SKILL.md`](../skills/review-paper/SKILL.md) · [`.claude/skills/preregister/SKILL.md`](../skills/preregister/SKILL.md) · [`.claude/skills/power-analysis/SKILL.md`](../skills/power-analysis/SKILL.md).
