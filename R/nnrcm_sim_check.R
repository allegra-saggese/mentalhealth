# nnrcm_sim_check.R -- does NNRCM recover a known spatial range on OUR county geometry? (script4h-gp.py, D-61)
# Simulates y = exponential-covariance Gaussian process (sd 1, range r km) + white noise (sd 0.5) at the county centroids, fits
# NNRCM.marginal.infer (smoothness 0.5), and writes the fitted (alpha, sigma, phi, xi) next to the truth. The calibration in
# script4h-gp.py turns the fitted phi into kilometres from this table; nothing is read from the vignette.
# Run:  NNRCM_LIB=<library with NNRCM> Rscript R/nnrcm_sim_check.R <gp_dir> <tables_dir> <YYYY-MM-DD>
# Reads <gp_dir>/gp_geometry.csv (x, y in units of 100 km); writes <tables_dir>/<date>_4h-gp_sim_check.csv
args <- commandArgs(trailingOnly = TRUE)
gp_dir <- args[1]; tables_dir <- args[2]; today <- args[3]
.libPaths(c(Sys.getenv("NNRCM_LIB"), .libPaths()))
suppressMessages({library(NNRCM); library(MASS)})
set.seed(20261122)

geo <- read.csv(file.path(gp_dir, "gp_geometry.csv"), colClasses = c(fips = "character"))
geo <- geo[order(geo$x), ]                                 # NNRCM's neighbour sets need the rows in x order (spNNGP orders by the first coordinate)
stopifnot(!anyDuplicated(geo$x))
L <- as.matrix(geo[, c("x", "y")]); n <- nrow(L)
cat("counties:", n, "\n")
Dm <- as.matrix(dist(L))

ranges_km <- c(50, 100, 250, 500, 1000); reps <- 3
rows <- list()
for (r in ranges_km) {
  Ch <- chol(exp(-Dm / (r / 100)) + diag(1e-8, n))
  for (k in seq_len(reps)) {
    y <- as.vector(t(Ch) %*% rnorm(n)) + rnorm(n, 0, 0.5)
    t0 <- Sys.time()
    sink("/dev/null")
    fit <- tryCatch(NNRCM.marginal.infer(Y = y, observed.locations = L, smoothness = 0.5), error = function(e) NULL)
    sink()
    if (is.null(fit)) next
    p <- fit$par
    rows[[length(rows) + 1]] <- data.frame(true_range_km = r, rep = k, n = n, alpha = p[1], sigma = p[2], phi = p[3], xi = p[4],
                                           convergence = fit$convergence, secs = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    cat(sprintf("range %4d km rep %d: phi %.3f sigma %.3f xi %.3f (%.0fs)\n", r, k, p[3], p[2], p[4], rows[[length(rows)]]$secs))
  }
}
out <- do.call(rbind, rows)
write.csv(out, file.path(tables_dir, paste0(today, "_4h-gp_sim_check.csv")), row.names = FALSE)
