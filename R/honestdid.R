#!/usr/bin/env Rscript
# R/honestdid.R  --  Rambachan & Roth (2023) sensitivity analysis on the TWFE event studies
# (added 2026-10-17). Reads the inputs written by script4b-diagnostics.py --only honestdid.
#
# For each treatment definition x outcome, for two target parameters
#     first post-period effect (e = 0)
#     average of the first five post-period effects (e = 0 ... 4, or fewer if fewer exist)
# it reports
#     * the original (parallel-trends) confidence set,
#     * RELATIVE MAGNITUDES: post-treatment violations of parallel trends no larger than Mbar times the
#       largest pre-treatment violation, Mbar = 0, 0.5, 1, 1.5, 2   (method C-LF),
#     * SMOOTHNESS: the violation's slope can change by at most M between consecutive periods
#       (HonestDiD default grid of M, method FLCI),
#     * the BREAKDOWN value: the smallest Mbar (or M) at which the confidence set includes zero.
# DESCRIPTIVE: no verdicts are assigned here.
#
# Usage: Rscript R/honestdid.R      (after: python3 script4b-diagnostics.py --only honestdid)

suppressMessages(library(HonestDiD))

home <- Sys.getenv("HOME")
inp  <- file.path(home, "Dropbox/Mental/Data/output/tables/script4w/honestdid_inputs")
out  <- file.path(home, "Dropbox/Mental/Data/output/tables/script4w")
man  <- read.csv(file.path(inp, "manifest.csv"), stringsAsFactors = FALSE)
today <- format(Sys.Date(), "%Y-%m-%d")

basis <- function(n, idx) { v <- rep(0, n); v[idx] <- 1 / length(idx); v }
rows <- list()
add <- function(...) rows[[length(rows) + 1]] <<- data.frame(..., stringsAsFactors = FALSE)

for (i in seq_len(nrow(man))) {
  d <- man[i, ]
  b <- read.csv(file.path(inp, paste0(d$stem, "__beta.csv")))$beta
  S <- as.matrix(read.csv(file.path(inp, paste0(d$stem, "__sigma.csv")), header = FALSE))
  nPre <- d$num_pre; nPost <- d$num_post
  targets <- list(first_post = basis(nPost, 1), avg_first5 = basis(nPost, seq_len(min(5, nPost))))
  for (tn in names(targets)) {
    l <- targets[[tn]]
    base <- list(design = d$design, outcome = d$outcome, target = tn, num_pre = nPre, num_post = nPost)
    # original confidence set (exact parallel trends)
    oc <- tryCatch(constructOriginalCS(betahat = b, sigma = S, numPrePeriods = nPre, numPostPeriods = nPost, l_vec = l),
                   error = function(e) NULL)
    if (!is.null(oc)) do.call(add, c(base, list(restriction = "original (parallel trends)", M = NA, lb = oc$lb, ub = oc$ub)))
    # relative magnitudes
    rm <- tryCatch(createSensitivityResults_relativeMagnitudes(betahat = b, sigma = S, numPrePeriods = nPre,
            numPostPeriods = nPost, l_vec = l, Mbarvec = seq(0, 2, by = 0.5), gridPoints = 500),
            error = function(e) { message("RM failed: ", d$design, "|", d$outcome, "|", tn, ": ", conditionMessage(e)); NULL })
    if (!is.null(rm)) for (k in seq_len(nrow(rm))) do.call(add, c(base, list(restriction = "relative magnitudes (Mbar)",
                          M = rm$Mbar[k], lb = rm$lb[k], ub = rm$ub[k])))
    # smoothness
    sm <- tryCatch(createSensitivityResults(betahat = b, sigma = S, numPrePeriods = nPre, numPostPeriods = nPost,
            l_vec = l), error = function(e) { message("SD failed: ", d$design, "|", d$outcome, "|", tn, ": ", conditionMessage(e)); NULL })
    if (!is.null(sm)) for (k in seq_len(nrow(sm))) do.call(add, c(base, list(restriction = "smoothness (M)",
                          M = sm$M[k], lb = sm$lb[k], ub = sm$ub[k])))
  }
  cat(sprintf("done %s | %s\n", d$design, d$outcome))
}
res <- do.call(rbind, rows)
write.csv(res, file.path(out, paste0(today, "_honestdid_results.csv")), row.names = FALSE)

# breakdown values: smallest M at which the interval contains zero
bk <- do.call(rbind, lapply(split(res, list(res$design, res$outcome, res$target, res$restriction), drop = TRUE), function(x) {
  if (x$restriction[1] == "original (parallel trends)") return(NULL)
  x <- x[order(x$M), ]
  inc <- x$lb <= 0 & x$ub >= 0
  data.frame(design = x$design[1], outcome = x$outcome[1], target = x$target[1], restriction = x$restriction[1],
             breakdown_M = if (any(inc)) x$M[which(inc)[1]] else NA_real_,
             grid_max = max(x$M), zero_in_set_at_M0 = inc[1])
}))
write.csv(bk, file.path(out, paste0(today, "_honestdid_breakdown.csv")), row.names = FALSE)
cat("\nwrote results and breakdown to", out, "\n")
