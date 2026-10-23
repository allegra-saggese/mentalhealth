# qa_cs_crosscheck.R -- R `did` (Callaway & Sant'Anna) on the balanced extract written by qa_cs_crosscheck.py.
# Usage: Rscript qa_cs_crosscheck.R <extract.csv> <out_cells.csv> <out_overall.csv>
# Settings mirror did_cs.py: outcome regression, not-yet-treated comparison, base year g-1 (universal), no bootstrap.
suppressMessages(library(did))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
xvars <- grep("^x[0-9]+$", names(d), value = TRUE)
fit <- att_gt(yname = "y", tname = "year", idname = "fips", gname = "g", data = d,
              xformla = as.formula(paste("~", paste(xvars, collapse = " + "))),
              panel = TRUE, allow_unbalanced_panel = FALSE, control_group = "notyettreated",
              est_method = "reg", base_period = "universal", bstrap = FALSE, cband = FALSE)
write.csv(data.frame(cohort = fit$group, year = fit$t, att = fit$att, se = fit$se), args[2], row.names = FALSE)
s <- aggte(fit, type = "simple", na.rm = TRUE)
write.csv(data.frame(overall_att = s$overall.att, overall_se = s$overall.se), args[3], row.names = FALSE)
