#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial.py -- county locations, spatial weights, Moran's I and the spatial Durbin panel model (numpy only; no spatial package).

    county_centroids(geojson)           centroid (lat, lon) of every county outline            (area-weighted polygon formula)
    haversine_km(lat, lon)              great-circle distance matrix in km
    distance_adjacency(lat, lon, km)    0/1 matrix: centroids within `km`, no self-links
    commuting_zone_adjacency(zone)      0/1 matrix: same commuting zone, no self-links
    row_standardise(A)                  W = A / row sums (a row of zeros stays zero)
    morans_i(x, W, n_perm, rng)         Moran's I with a permutation p-value
    sdm_fit(Y, D, X, W)                 spatial Durbin panel model, county and year effects, by maximum likelihood
    sdm_effects(fit, ...)               direct, indirect (spillover) and total effect of the treatment, with simulated SEs
    sdm_wild_bootstrap(fit, groups, ...) state-block wild bootstrap of the whole fit

The model (N counties, T years, balanced panel; W row-standardised, symmetric before standardising, no isolated rows):

    y_it = rho * sum_j W_ij y_jt  +  beta * D_it  +  theta * sum_j W_ij D_jt  +  x_it' gamma  +  a_i + d_t + e_it

Durbin terms (W*x) are added for the treatment only (a "restricted" Durbin model; the controls enter directly). The county
effects a_i and year effects d_t are removed by the two-way within transformation; the likelihood then has
n_eff = (N-1)(T-1) observations and the Jacobian (T-1) * sum_i ln(1 - rho * w_i) over the eigenvalues w_i of W except one unit
eigenvalue (the cross-sectional mean that the year effects remove). Standard errors come from the numerical Hessian of the
full log-likelihood. Effects (LeSage-Pace decomposition; to be verified against the source before it is cited):

    direct   = mean_i (beta + theta w_i) / (1 - rho w_i)      average own effect, including feedback through neighbours
    total    = (beta + theta) / (1 - rho)                     valid because rows of W sum to 1
    indirect = total - direct                                 the spillover

Everything here is deterministic given the rng passed in (default_rng); the caller sets the seed.
"""
import numpy as np

EARTH_KM = 6371.0


# ---------------------------------------------------------------------------------------------- locations
def haversine_km(lat, lon):
    """Great-circle distances (km) between all pairs of points; lat, lon in degrees."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _ring_area_centroid(ring, lat_ref):
    """Signed area and centroid (x east, y north) of one closed ring of [lon, lat] points, after a local equirectangular projection."""
    p = np.asarray(ring, float)
    x, y = p[:, 0] * np.cos(np.radians(lat_ref)), p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2
    if abs(area) < 1e-15:
        return 0.0, x.mean(), y.mean()
    return area, ((x + x1) * cross).sum() / (6 * area), ((y + y1) * cross).sum() / (6 * area)


def county_centroids(geojson):
    """Centroid of every feature of a county GeoJSON (feature id = 5-digit FIPS). Returns {fips: (lat, lon)}.

    Polygons and MultiPolygons; the outer ring adds its area and holes subtract theirs. The projection is local to each county, so the
    result is the planar centroid of the outline, accurate to a few km, which is all the 50-500 km cut-offs need."""
    out = {}
    for ft in geojson["features"]:
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        lat_ref = float(np.mean([pt[1] for pt in polys[0][0]]))
        wsum = cx = cy = 0.0
        for poly in polys:
            for k, ring in enumerate(poly):
                area, x, y = _ring_area_centroid(ring, lat_ref)
                w = abs(area) * (1 if k == 0 else -1)
                wsum, cx, cy = wsum + w, cx + w * x, cy + w * y
        if wsum <= 0:                                           # degenerate outline: fall back to the mean of the vertices
            pts = np.asarray([pt for poly in polys for pt in poly[0]], float)
            out[str(ft["id"]).zfill(5)] = (float(pts[:, 1].mean()), float(pts[:, 0].mean()))
        else:
            out[str(ft["id"]).zfill(5)] = (cy / wsum, cx / wsum / np.cos(np.radians(lat_ref)))
    return out


# ---------------------------------------------------------------------------------------------- weights
def distance_adjacency(lat, lon, km):
    """0/1 matrix, 1 where two different counties' centroids are within `km` kilometres."""
    A = (haversine_km(lat, lon) <= km).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def commuting_zone_adjacency(zone):
    """0/1 matrix, 1 where two different counties are in the same commuting zone; counties with a missing zone have no links."""
    z = np.asarray(zone, object)
    ok = np.array([v is not None and v == v for v in z])
    A = ((z[:, None] == z[None, :]) & ok[:, None] & ok[None, :]).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def row_standardise(A):
    """W = A / row sums; a row with no neighbour stays all zero (an 'isolate': drop it before fitting a Durbin model)."""
    s = A.sum(axis=1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A), where=s > 0)


# ---------------------------------------------------------------------------------------------- Moran's I
def morans_i(x, W, n_perm=999, rng=None):
    """Moran's I of x with weights W (row-standardised), the value expected under no spatial pattern, and a two-sided permutation p-value.

    I = (n / S0) * z'Wz / z'z with z = x - mean(x), S0 = sum of all weights. The null distribution reshuffles x over the counties."""
    x = np.asarray(x, float)
    z = x - x.mean()
    n, s0 = len(z), W.sum()
    stat = n / s0 * (z @ W @ z) / (z @ z)
    p = np.nan
    if n_perm and rng is not None:
        P = np.column_stack([rng.permutation(z) for _ in range(n_perm)])
        sim = n / s0 * np.einsum("ij,ij->j", P, W @ P) / (z @ z)
        p = (1 + np.sum(np.abs(sim) >= abs(stat))) / (n_perm + 1)
    return {"I": float(stat), "expected": -1.0 / (n - 1), "p_perm": float(p), "n": int(n)}


# ---------------------------------------------------------------------------------------------- Durbin panel model
def _two_way_demean(M):
    """Remove county (row) and year (column) means from an N x T array."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def _eigs(W):
    """Eigenvalues of W (real, because W is a row-standardised symmetric 0/1 matrix). General eigen-solver: it takes a few seconds
    for 3,000 counties and is done once per W; pass the result to sdm_fit(eig=...) to reuse it."""
    return np.sort(np.linalg.eigvals(W).real)


def sdm_fit(Y, D, X, W, eig=None):
    """Spatial Durbin panel model by maximum likelihood. Y, D: N x T arrays; X: list of N x T control arrays (may be empty);
    W: N x N row-standardised, no zero rows. Returns a dict with rho, beta, theta, gamma, their standard errors, the log-likelihood
    and the pieces `sdm_effects` and `sdm_wild_bootstrap` need."""
    if np.any(W.sum(axis=1) <= 0):
        raise ValueError("W has an isolated row; drop isolates before fitting")
    cols = [D, W @ D] + list(X)
    names = ["beta", "theta"] + [f"gamma{k + 1}" for k in range(len(X))]
    Z = np.column_stack([_two_way_demean(c).ravel() for c in cols])
    w_eig = _eigs(W) if eig is None else eig
    fit = _fit_core(_two_way_demean(Y), Z, W, w_eig, names, hessian=True)
    return fit


def _fit_core(Yd, Z, W, w_eig, names, hessian=False):
    """Maximum likelihood for rho and the coefficients, given the two-way demeaned outcome Yd (N x T) and regressors Z (NT x k)."""
    N, T = Yd.shape
    jac_eig = np.sort(w_eig)[:-1]                            # drop the unit eigenvalue the year effects remove
    n_eff = (N - 1) * (T - 1)
    y0 = Yd.ravel()
    WY = (W @ Yd).ravel()
    b0, *_ = np.linalg.lstsq(Z, y0, rcond=None)
    bw, *_ = np.linalg.lstsq(Z, WY, rcond=None)
    e0, ew = y0 - Z @ b0, WY - Z @ bw
    lo, hi = 1.0 / w_eig.min() + 1e-6, 1.0 - 1e-6

    def conc(rho):
        e = e0 - rho * ew
        return -0.5 * n_eff * np.log(e @ e / n_eff) + (T - 1) * np.sum(np.log1p(-rho * jac_eig))

    grid = np.linspace(lo, hi, 61)
    k = int(np.argmax([conc(r) for r in grid]))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    gr = (np.sqrt(5) - 1) / 2                                # golden-section refinement inside the best grid bracket
    c, d_ = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if conc(c) > conc(d_):
            b = d_
        else:
            a = c
        c, d_ = b - gr * (b - a), a + gr * (b - a)
    rho = float((a + b) / 2)
    coef = b0 - rho * bw
    e = e0 - rho * ew
    sig2 = float(e @ e / n_eff)
    out = {"rho": rho, "beta": float(coef[0]), "theta": float(coef[1]), "sigma2": sig2, "n_units": N, "n_years": T, "names": names,
           "coef": coef, "eig": w_eig, "W": W, "Z": Z, "n_eff": n_eff, "resid": e}
    if hessian:
        def loglik(par):                                     # full log-likelihood: par = [rho, coef..., ln sigma2]
            r, cf, ls2 = par[0], par[1:-1], par[-1]
            res = y0 - r * WY - Z @ cf
            return (-0.5 * n_eff * (np.log(2 * np.pi) + ls2) - res @ res / (2 * np.exp(ls2))
                    + (T - 1) * np.sum(np.log1p(-r * jac_eig)))

        par = np.concatenate([[rho], coef, [np.log(sig2)]])
        cov = np.linalg.pinv(-_num_hessian(loglik, par))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        out.update({"loglik": float(loglik(par)), "cov": cov, "se_rho": float(se[0]), "se": se[1:-1],
                    "se_beta": float(se[1]), "se_theta": float(se[2])})
    return out


def _num_hessian(f, x, rel=1e-4):
    """Central-difference Hessian of f at x."""
    n = len(x)
    h = rel * np.maximum(np.abs(x), 1.0)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def _effects(rho, beta, theta, w_eig):
    direct = np.mean((beta + theta * w_eig) / (1 - rho * w_eig))
    total = (beta + theta) / (1 - rho)
    return direct, total - direct, total


def sdm_effects(fit, n_draw=2000, rng=None):
    """Direct, indirect (spillover) and total effect of the treatment at the estimates, with simulation standard errors: draws of
    (rho, beta, theta) from the normal approximation to the maximum-likelihood estimate; draws with rho outside its admissible range are dropped."""
    rho, beta, theta = fit["rho"], fit["beta"], fit["theta"]
    est = _effects(rho, beta, theta, fit["eig"])
    out = {"direct": est[0], "indirect": est[1], "total": est[2]}
    if rng is not None:
        idx = [0, 1, 2]                                      # rho, beta, theta positions in the parameter vector
        cov = fit["cov"][np.ix_(idx, idx)]
        mean = np.array([rho, beta, theta])
        dr = rng.multivariate_normal(mean, cov, size=n_draw)
        lo, hi = 1.0 / fit["eig"].min(), 1.0
        dr = dr[(dr[:, 0] > lo) & (dr[:, 0] < hi)]
        sim = np.array([_effects(*r, fit["eig"]) for r in dr])
        for k, nm in enumerate(("direct", "indirect", "total")):
            out["se_" + nm] = float(sim[:, k].std(ddof=1))
    return out


def sdm_wild_bootstrap(fit, groups, n_boot=499, rng=None):
    """Wild cluster bootstrap of the whole fit, resampling by `groups` (a length-N array, e.g. the state of each county).

    Draws one Webb six-point weight per group, rescales every residual of that group by it, rebuilds y from the fitted model
    y = (I - rho W)^(-1) (Z coef + e*) (this keeps the spatial structure and the within-group dependence), and re-estimates with the
    regressors held fixed. Returns the draws of rho, beta, theta, direct, indirect and total as a dict of arrays."""
    N, T = fit["n_units"], fit["n_years"]
    labels, inv = np.unique(np.asarray(groups), return_inverse=True)
    webb = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    A = np.linalg.inv(np.eye(N) - fit["rho"] * fit["W"])
    Zb = (fit["Z"] @ fit["coef"]).reshape(N, T)
    E = fit["resid"].reshape(N, T)
    names = ("rho", "beta", "theta", "direct", "indirect", "total")
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        w = webb[rng.integers(0, 6, size=len(labels))][inv][:, None]
        f = _fit_core(_two_way_demean(A @ (Zb + w * E)), fit["Z"], fit["W"], fit["eig"], fit["names"])
        for k, v in zip(names, (f["rho"], f["beta"], f["theta"], *_effects(f["rho"], f["beta"], f["theta"], fit["eig"]))):
            draws[k].append(v)
    return {k: np.array(v) for k, v in draws.items()}
