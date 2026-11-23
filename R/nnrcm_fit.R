# nnrcm_fit.R -- NN-RCM fits to the A2 residuals and to the raw outcome, by outcome x series x year (x Census division). script4h-gp.py, D-61.
# For each cross-section: centre and scale the series (the model's prior on its variances has fixed scale), sort the counties by x (NNRCM's neighbour
# sets need the rows in the order spNNGP uses), fit NNRCM.marginal.infer with Matern smoothness 0.5 (exponential) -> (alpha = degrees of freedom,
# sigma = sd of the spatial part, phi = decay, xi = nugget variance; phi is converted to km in script4h-gp.py by the simulation calibration).
#   region "US"        all counties of the year;  region <division>  one fit per Census division with at least MIN_DIV counties that year
#   kappa 0.5          every cell;  kappa 1.5 (smoother) only for the national residual fit, as a sensitivity
# Run:  NNRCM_LIB=<library with NNRCM> Rscript R/nnrcm_fit.R <gp_dir> <tables_dir> <YYYY-MM-DD> "<outcome name>"
# Reads <gp_dir>/gp_input.csv; writes <tables_dir>/<date>_4h-gp_fits__<outcome slug>.csv (one row per fit; failures keep their error text).
args <- commandArgs(trailingOnly = TRUE)
gp_dir <- args[1]; tables_dir <- args[2]; today <- args[3]; outcome <- args[4]
.libPaths(c(Sys.getenv("NNRCM_LIB"), .libPaths()))
suppressMessages(library(NNRCM))
MIN_DIV <- 250
set.seed(20261122)

dat <- read.csv(file.path(gp_dir, "gp_input.csv"), colClasses = c(fips = "character", state_fips = "character"))
dat <- dat[dat$outcome == outcome, ]
stopifnot(nrow(dat) > 0)
cat(outcome, ":", nrow(dat), "county-years\n")

fit_one <- function(d, series, region, kappa, year) {
  d <- d[order(d$x), ]
  y <- d[[series]]; sd0 <- sd(y)
  y <- (y - mean(y)) / sd0
  t0 <- Sys.time()
  res <- tryCatch({
    sink("/dev/null"); on.exit(sink(), add = TRUE)
    f <- NNRCM.marginal.infer(Y = y, observed.locations = as.matrix(d[, c("x", "y")]), smoothness = kappa)
    sink(); on.exit()
    list(p = f$par, conv = f$convergence, err = "")
  }, error = function(e) { while (sink.number() > 0) sink(); list(p = rep(NA, 4), conv = NA, err = substr(conditionMessage(e), 1, 150)) })
  data.frame(outcome = outcome, series = ifelse(series == "resid", "A2 residual (E2)", "raw outcome"), region = region, kappa = kappa, year = year,
             n = nrow(d), sd_in = sd0, alpha = res$p[1], sigma = res$p[2], phi = res$p[3], xi = res$p[4], convergence = res$conv,
             secs = as.numeric(difftime(Sys.time(), t0, units = "secs")), error = res$err)
}

rows <- list()
for (yr in sort(unique(dat$year))) {
  dy <- dat[dat$year == yr, ]
  stopifnot(!anyDuplicated(dy$fips))
  for (series in c("resid", "raw")) {
    rows[[length(rows) + 1]] <- fit_one(dy, series, "US", 0.5, yr)
    if (series == "resid") rows[[length(rows) + 1]] <- fit_one(dy, series, "US", 1.5, yr)
    for (dv in sort(unique(dy$division))) {
      dd <- dy[dy$division == dv, ]
      if (nrow(dd) >= MIN_DIV) rows[[length(rows) + 1]] <- fit_one(dd, series, dv, 0.5, yr)
    }
  }
  cat("year", yr, "done;", length(rows), "fits so far\n")
  flush.console()
}
out <- do.call(rbind, rows)
slug <- gsub("[^a-z0-9]+", "_", tolower(outcome))
write.csv(out, file.path(tables_dir, paste0(today, "_4h-gp_fits__", slug, ".csv")), row.names = FALSE)
cat("wrote", nrow(out), "fits;", sum(out$convergence != 0 | is.na(out$convergence)), "not converged or failed\n")
