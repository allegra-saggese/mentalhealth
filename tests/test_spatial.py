"""Tests for spatial.py (run: python3 tests/test_spatial.py).

Known-answer checks for centroids and distances, invariants of the weights, Moran's I on noise and on a smooth field, and
recovery of a known rho, beta, theta by the spatial Durbin panel model on simulated data. Exit code 0 if everything passes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spatial as sp  # noqa: E402

SEED = 20261029
failures = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


# ---- centroids: a square, a square with a hole off-centre, and a two-part county
sq = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
hole = [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0], [0.5, 0.5]]
gj = {"features": [
    {"id": "00001", "geometry": {"type": "Polygon", "coordinates": [sq]}},
    {"id": "00002", "geometry": {"type": "Polygon", "coordinates": [sq, hole]}},
    {"id": "00003", "geometry": {"type": "MultiPolygon", "coordinates": [[sq], [[[10, 0], [12, 0], [12, 2], [10, 2], [10, 0]]]]}},
]}
c = sp.county_centroids(gj)
check("square centroid is (1, 1)", abs(c["00001"][0] - 1) < 1e-3 and abs(c["00001"][1] - 1) < 1e-3, str(c["00001"]))
# the hole (area 0.25, centre 0.75, 0.75) pulls the centroid away from the hole: (4*1 - 0.25*0.75)/3.75
want = (4 * 1 - 0.25 * 0.75) / 3.75
check("a hole shifts the centroid away from it", abs(c["00002"][0] - want) < 2e-3 and abs(c["00002"][1] - want) < 2e-3, str(c["00002"]))
check("a two-part county sits between its parts", abs(c["00003"][1] - 6) < 0.2 and abs(c["00003"][0] - 1) < 1e-3, str(c["00003"]))

# ---- distances
d = sp.haversine_km([0.0, 1.0, 0.0], [0.0, 0.0, 90.0])
check("one degree of latitude is 111.19 km", abs(d[0, 1] - 111.19) < 0.05, f"{d[0, 1]:.2f}")
check("a quarter of the equator is 10,007.5 km", abs(d[0, 2] - 10007.5) < 1.0, f"{d[0, 2]:.1f}")
check("distance matrix symmetric with zero diagonal", np.allclose(d, d.T) and np.allclose(np.diag(d), 0))

# ---- weights
rng = np.random.default_rng(SEED)
N = 150
lat, lon = rng.uniform(35, 45, N), rng.uniform(-100, -85, N)
A = sp.distance_adjacency(lat, lon, 250.0)
W = sp.row_standardise(A)
check("adjacency symmetric, no self-links", np.array_equal(A, A.T) and np.all(np.diag(A) == 0))
check("row-standardised rows sum to 1", np.allclose(W.sum(axis=1), 1.0), f"isolates {int((A.sum(axis=1) == 0).sum())}")
zone = np.array([0, 0, 1, 1, 1, None], object)
Az = sp.commuting_zone_adjacency(zone)
check("commuting-zone links: same zone only, none for a missing zone", Az[2, 3] == 1 and Az[0, 2] == 0 and Az[5].sum() == 0 and Az[2, 2] == 0)
check("eigenvalues of W are real, at most 1, and include 1", abs(sp._eigs(W).max() - 1) < 1e-9)

# ---- Moran's I
noise = rng.normal(size=N)
m = sp.morans_i(noise, W, n_perm=499, rng=rng)
check("Moran's I of white noise is near its expectation", abs(m["I"] - m["expected"]) < 0.1 and m["p_perm"] > 0.01, f"I {m['I']:.3f}, p {m['p_perm']:.3f}")
smooth = np.sin(np.radians(lat) * 20) + np.cos(np.radians(lon) * 20)
m = sp.morans_i(smooth, W, n_perm=499, rng=rng)
check("Moran's I of a smooth field is clearly positive", m["I"] > 0.3 and m["p_perm"] < 0.01, f"I {m['I']:.3f}, p {m['p_perm']:.3f}")

# ---- the Durbin panel model recovers a known rho, beta, theta
RHO, BETA, THETA, GAMMA, T = 0.4, 1.0, 0.5, 0.3, 30
Wd = sp.row_standardise(sp.distance_adjacency(lat, lon, 400.0))
A_inv = np.linalg.inv(np.eye(N) - RHO * Wd)
est = {"rho": [], "beta": [], "theta": [], "direct": [], "indirect": []}
for s in range(8):
    r = np.random.default_rng(SEED + s)
    Dm = (r.random((N, T)) < 0.3).astype(float) + 0.5 * r.normal(size=(N, T))
    x = r.normal(size=(N, T))
    mu = BETA * Dm + THETA * (Wd @ Dm) + GAMMA * x + r.normal(size=(N, 1)) + r.normal(size=(1, T))
    Y = A_inv @ (mu + r.normal(size=(N, T)))
    f = sp.sdm_fit(Y, Dm, [x], Wd)
    e = sp.sdm_effects(f, n_draw=500, rng=r)
    for k in ("rho", "beta", "theta"):
        est[k].append(f[k])
    est["direct"].append(e["direct"]); est["indirect"].append(e["indirect"])
mean = {k: float(np.mean(v)) for k, v in est.items()}
check("rho recovered (sampling sd of one fit is about 0.07 at this size)", abs(mean["rho"] - RHO) < 0.06, f"mean {mean['rho']:.3f} vs {RHO}")
check("beta recovered", abs(mean["beta"] - BETA) < 0.05, f"mean {mean['beta']:.3f} vs {BETA}")
check("theta recovered", abs(mean["theta"] - THETA) < 0.1, f"mean {mean['theta']:.3f} vs {THETA}")
# effects against the explicit matrix S = (I - rho W)^(-1) (beta I + theta W)
S = A_inv @ (BETA * np.eye(N) + THETA * Wd)
direct_true, total_true = np.trace(S) / N, S.sum(axis=1).mean()
d_, i_, t_ = sp._effects(RHO, BETA, THETA, sp._eigs(Wd))
check("eigenvalue formula equals the explicit effects matrix", abs(d_ - direct_true) < 1e-9 and abs(t_ - total_true) < 1e-9,
      f"direct {d_:.6f} vs {direct_true:.6f}, total {t_:.6f} vs {total_true:.6f}")
check("indirect = total - direct", abs(i_ - (t_ - d_)) < 1e-12)
check("mean estimated indirect effect near the true one", abs(mean["indirect"] - (total_true - direct_true)) < 0.2,
      f"{mean['indirect']:.3f} vs {total_true - direct_true:.3f}")

# ---- the ML standard error is of the right size, and the wild bootstrap agrees with it in order of magnitude
sdb = []
for s in range(8):
    r = np.random.default_rng(SEED + 100 + s)
    Dm = (r.random((N, T)) < 0.3).astype(float)
    Y = A_inv @ (BETA * Dm + THETA * (Wd @ Dm) + r.normal(size=(N, 1)) + r.normal(size=(1, T)) + r.normal(size=(N, T)))
    f = sp.sdm_fit(Y, Dm, [], Wd)
    sdb.append((f["beta"], f["se_beta"]))
b, se = np.array(sdb).T
check("ML standard error close to the sampling spread of beta", 0.4 < np.mean(se) / max(b.std(ddof=1), 1e-9) < 2.5,
      f"mean se {np.mean(se):.3f}, spread {b.std(ddof=1):.3f}")
groups = (np.floor((lon + 100) / 5)).astype(int)
boot = sp.sdm_wild_bootstrap(f, groups, n_boot=40, rng=np.random.default_rng(SEED))
ratio = boot["beta"].std(ddof=1) / f["se_beta"]
check("wild-bootstrap spread of beta is within a factor of 3 of the ML standard error", 0.33 < ratio < 3, f"ratio {ratio:.2f}")

print("\nFAILED: " + ", ".join(failures) if failures else "\nall spatial tests passed")
sys.exit(1 if failures else 0)
