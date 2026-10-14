"""Recover known surprisal slopes (human 6 ms/bit, machine 3 ms/bit) and a zero effect."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread import analysis  # noqa: E402


def sim(n_texts, slope, source, rng, surp_mean=7.0):
    rows = []
    for t in range(n_texts):
        n = 80
        length = rng.integers(2, 12, n).astype(float)
        zipf = 7 - 0.3 * length + rng.normal(0, 0.7, n)
        surp = np.clip(surp_mean - 0.8 * (zipf - 5) + rng.normal(0, 3, n), 0, None)
        gd = 200 + 8 * length - 10 * zipf + slope * surp + rng.normal(0, 25, n)
        d = pd.DataFrame(dict(text_id=f"{source}{t}", corpus=source, source=source, length=length,
                              zipf=zipf, surprisal=surp, GD=gd))
        for c in ["length", "zipf", "surprisal"]:
            d[f"prev_{c}"] = d[c].shift(1)
        rows.append(d.iloc[1:])
    return pd.concat(rows, ignore_index=True)


def test_recovers_slopes():
    rng = np.random.default_rng(0)
    d = pd.concat([sim(40, 6, "human", rng), sim(40, 3, "machine", rng, surp_mean=5),
                   sim(40, 0, "null", rng)], ignore_index=True)
    s = analysis.slopes(d, "GD", ["corpus"], n_boot=200).set_index("corpus")
    assert abs(s.loc["human", "slope_ms_per_bit"] - 6) < 0.6
    assert abs(s.loc["machine", "slope_ms_per_bit"] - 3) < 0.6
    assert s.loc["null", "ci_lo"] < 0 < s.loc["null", "ci_hi"]
    dll = analysis.delta_ll(d, "GD", ["corpus"], n_perm=500).set_index("corpus")
    assert dll.loc["human", "dLL"] > dll.loc["machine", "dLL"] > 0
    assert dll.loc["human", "p_perm"] < 0.01
    assert dll.loc["null", "p_perm"] > 0.05
    b = analysis.binned(d, "GD", ["corpus"])
    assert set(b.corpus) == {"human", "machine", "null"}
    print(s[["slope_ms_per_bit", "ci_lo", "ci_hi"]].round(2))
    print(dll[["dLL", "p_perm"]].round(4))


if __name__ == "__main__":
    test_recovers_slopes()
    print("ok")
