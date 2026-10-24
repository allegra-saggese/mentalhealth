# script4za-dcdh.R -- de Chaisemartin & D'Haultfoeuille DID_l (R package DIDmultiplegtDYN), called by estimators.dcdh_estimates().
# Usage: Rscript script4za-dcdh.R <dir with in.csv> <unit> <time> <outcome> <treatment> <cluster> <effects> <placebos> <controls, comma separated or empty> [normalized: 1 or 0]
# Writes effects.csv, placebos.csv and ate.csv into <dir>. polars must be loaded before the package (its own instruction).
suppressMessages({library(polars); library(DIDmultiplegtDYN)})
a <- commandArgs(trailingOnly = TRUE)
dir <- a[1]
d <- read.csv(file.path(dir, "in.csv"))
ctrl <- if (nchar(a[9]) > 0) strsplit(a[9], ",")[[1]] else NULL
args <- list(df = d, outcome = a[4], group = a[2], time = a[3], treatment = a[5], effects = as.integer(a[7]),
             placebo = as.integer(a[8]), cluster = a[6], graph_off = TRUE)
if (!is.null(ctrl)) args$controls <- ctrl
if (length(a) >= 10 && a[10] == "1") args$normalized <- TRUE   # effects per one unit of treatment
r <- do.call(did_multiplegt_dyn, args)$results
write.csv(as.data.frame(r$Effects), file.path(dir, "effects.csv"), row.names = FALSE)
pl <- if (is.null(r$Placebos)) data.frame() else as.data.frame(r$Placebos)
write.csv(pl, file.path(dir, "placebos.csv"), row.names = FALSE)
write.csv(as.data.frame(r$ATE), file.path(dir, "ate.csv"), row.names = FALSE)
