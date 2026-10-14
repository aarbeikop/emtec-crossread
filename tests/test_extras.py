"""Divergence test on simulated data with a known shape:
human RT ~ surprisal flattens above 7 bits, machine RT stays linear. Plus a
null case where both sources share one curve (but sit on different surprisal
ranges), which must NOT come out as a source difference."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread import extras  # noqa: E402


def sim(n_texts, source, rng, mean_s, high_slope):
    rows = []
    for t in range(n_texts):
        n = 80
        length = rng.integers(2, 12, n).astype(float)
        zipf = 7 - 0.3 * length + rng.normal(0, 0.7, n)
        surp = np.clip(rng.gamma(2, mean_s / 2, n) + 0.3 * (5 - zipf), 0, 25)
        f = 2.0 * np.minimum(surp, 7) + high_slope * np.clip(surp - 7, 0, None)
        gd = 200 + 8 * length - 10 * zipf + f + rng.normal(0, 25, n)
        d = pd.DataFrame(dict(text_id=f"{source}{t}", corpus=source, source=source, length=length,
                              zipf=zipf, surprisal=surp, GD=gd))
        for c in ["length", "zipf", "surprisal"]:
            d[f"prev_{c}"] = d[c].shift(1)
        rows.append(d.iloc[1:])
    return pd.concat(rows, ignore_index=True)


def test_divergence_detects_shape_difference():
    rng = np.random.default_rng(0)
    d = pd.concat([sim(60, "human", rng, 6, high_slope=0.3),
                   sim(200, "machine", rng, 4.5, high_slope=2.0)], ignore_index=True)
    o = extras.divergence(d, "GD", knot=7)
    t = o["tests"].set_index("test")["p"]
    pw = o["piecewise"].set_index(["source", "quantity"])["estimate"]
    print(o["tests"].round(4).to_string(index=False))
    print(o["piecewise"].round(2).to_string(index=False))
    assert t["spline x source (curve differs)"] < 0.01
    assert abs(pw[("human", "slope_below_knot")] - 2) < 0.6
    assert abs(pw[("human", "slope_above_knot")] - 0.3) < 0.8
    assert abs(pw[("machine", "slope_above_knot")] - 2) < 0.8


def test_divergence_null_with_range_difference():
    rng = np.random.default_rng(1)
    d = pd.concat([sim(60, "human", rng, 7, high_slope=0.5),
                   sim(200, "machine", rng, 4, high_slope=0.5)], ignore_index=True)
    o = extras.divergence(d, "GD", knot=7)
    p = o["tests"].set_index("test")["p"]["spline x source (curve differs)"]
    print("null case spline p =", round(p, 3))
    assert p > 0.05


def test_wild_bootstrap_few_clusters():
    """12 'human' clusters vs many 'machine' ones: wild p must reject a real slope
    difference and not reject when there is none."""
    rng = np.random.default_rng(2)
    real = pd.concat([sim(12, "human", rng, 6, high_slope=2.0),
                      sim(80, "machine", rng, 5, high_slope=2.0)], ignore_index=True)
    real.loc[real.source == "machine", "GD"] += 2.5 * np.minimum(real.loc[real.source == "machine", "surprisal"], 7)
    null = pd.concat([sim(12, "human", rng, 6, high_slope=2.0),
                      sim(80, "machine", rng, 5, high_slope=2.0)], ignore_index=True)
    out = {}
    for name, d in [("real", real), ("null", null)]:
        pw = extras.divergence(d, "GD", knot=7, wild=399)["piecewise"]
        out[name] = pw.set_index("quantity").loc["diff_slope_below_knot", "p_wild"]
    print("wild p:", out)
    assert out["real"] < 0.05 and out["null"] > 0.05


if __name__ == "__main__":
    test_wild_bootstrap_few_clusters()
    test_divergence_detects_shape_difference()
    test_divergence_null_with_range_difference()
    print("ok")
