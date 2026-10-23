# Results Reporting

How results are shown, in conversation and in files. This is the standing contract for the paper's evidence.

1. **Results before interpretation.** Show the numbers (table, plot, or CSV path with a printed head) first. Interpretation, flags and caveats come as a separate, labelled block and only when asked, or when a flag blocks the next step.
2. **Plain language.** Define each term before its first use: "first expansion" (the first census wave at which a county's large-dairy count rises), "first entry" (0 to 1 or more large dairies), "the control set" (the 8 pre-committed full-coverage controls), "pre-treatment years". No private abbreviations. Decision and analysis IDs (D-34, A21) are cited with a plain phrase the first time.
3. **Show the equation** when asking the user to choose between estimators or specifications; put the estimand next to the estimator.
4. **Name the state you reached.** Say what was run, on which panel, with which seed, and what was not run or not verified. Never describe a check as passed that was not executed.
5. **Report against the headline.** Failed pre-trends, nulls, window sensitivity, and results that weaken the claim appear with equal prominence.
6. **Claims match evidence.** Do not upgrade "associated with" to "causes", or a pointwise result to a uniform one. Causal language requires the identification argument and the pre-trend/robustness evidence to be cited next to it.
7. **No fabricated citations.** Every reference is searched and checked against a source; unverifiable ones are labelled as such, never silently kept. Use `/verify-claims` before a draft leaves the repo.
8. **Cite outputs by file and run state.** Before quoting a number, confirm which script produced it and that it ran on the current panel (`code-audit/plans/*MODEL-RUN-STATE.md`).
9. **Multiple outcomes and specifications:** state the family and the correction used (see `inference-robustness.md`); show unadjusted and adjusted values.
