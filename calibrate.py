#!/usr/bin/env python
"""Over-confidence test (see src/crossread/calibration.py). Needs surprisal at several temperatures:

  python run.py surprisal --lm gpt2 --temps 0.75 1.25 1.5 2 2.5 3 4
  python calibrate.py --lm gpt2
  python calibrate.py --all          # every LM in out/surprisal/ that has temperature columns

Writes out/calibration/<lm>/ and, with several LMs, out/calibration/scaling_T1_vs_Tstar.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import calibration, scaling  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--lm", default="gpt2")
p.add_argument("--all", action="store_true")
p.add_argument("--measures", nargs="+", default=["skip_rate", "FFD", "GD", "reg_rate", "reread", "TFT"])
p.add_argument("--hotspot-bits", type=float, default=1.0, help="'near-certain' threshold for hotspots")
p.add_argument("--n-boot", type=int, default=1000)
a = p.parse_args()

OUT = Path("out")
words = pd.read_parquet(OUT / "words.parquet")
texts = pd.read_parquet(OUT / "texts.parquet")
files = sorted((OUT / "surprisal").glob("*.parquet")) if a.all else [OUT / "surprisal" / f"{a.lm.replace('/', '__')}.parquet"]

allest = []
for f in files:
    surp = pd.read_parquet(f)
    if len(calibration.temp_columns(surp)) < 3:
        print(f"[{f.stem}] no temperature columns, skipping (rerun surprisal with --temps)")
        continue
    res = OUT / "calibration" / f.stem
    res.mkdir(parents=True, exist_ok=True)
    print(f"\n######## {f.stem}  temps={list(calibration.temp_columns(surp))}")
    est, con, curves, tabs, common = calibration.tstar(words, texts, surp, a.measures, a.n_boot)
    sd = calibration.per_sd_slopes(tabs, common, est, a.measures)
    for name, df in [("tstar", est), ("contrasts", con), ("curves", curves), ("slopes_per_sd", sd)]:
        df.to_csv(res / f"{name}.csv", index=False)
    calibration.plot_curves(curves, res / "dll_vs_temperature.png")
    hs, ex = calibration.hotspots(tabs[1.0].loc[common], texts, a.hotspot_bits)
    hs.to_csv(res / "hotspots_summary.csv", index=False)
    ex.to_csv(res / "hotspots_examples.csv", index=False)
    print("\n== T* (best-fitting temperature; >1 = LM over-confident) ==")
    print(est.round(3).to_string(index=False))
    print("\n== contrasts on log T* (>0 = first group needs more flattening) ==")
    print(con.round(3).to_string(index=False))
    print("\n== surprisal slopes: ms per bit vs ms per within-corpus SD, at T=1 and T* ==")
    print(sd.pivot_table(index=["measure", "temperature", "unit"], columns="corpus", values="slope").round(2).to_string())
    print(f"\n== hotspots: words with surprisal < {a.hotspot_bits} bit; excess regression / rereading "
          "beyond length + frequency + LM surprisal (resid > 0 = readers struggle more than the LM expects) ==")
    cols = ["corpus", "subset", "n_low", "share_low", "reg_rate_low", "resid_reg_rate_low",
            "resid_reg_rate_low_lo", "resid_reg_rate_low_hi", "reread_low", "resid_reread_low",
            "resid_reread_low_lo", "resid_reread_low_hi"]
    print(hs.reindex(columns=cols).round(4).to_string(index=False))
    for c, g in ex.groupby("corpus"):
        print(f"\n-- {c}: near-certain words with the most excess rereading --")
        print(g.head(8)[["surprisal", "resid_reread", "context"]].round(2).to_string(index=False))
    allest.append(est.assign(lm=f.stem))

if len(allest) >= 3:
    E = pd.concat(allest)
    E = E[E.subset == "all"]
    rows = []
    for (c, m), g in E.groupby(["corpus", "measure"]):
        info = [scaling.lm_info(l) for l in g.lm]
        keep = [i is not None for i in info]
        g, info = g[keep], [i for i in info if i]
        x, fam = np.log10([i[1] for i in info]), [i[0] for i in info]
        for col in ["dLL_T1", "dLL_Tstar"]:
            y = g[col].to_numpy()
            rows.append(dict(corpus=c, measure=m, at=col, rel_scaling_slope=scaling._fit_slope(x, y / y.mean(), fam)))
        rows.append(dict(corpus=c, measure=m, at="log_Tstar_vs_size",
                         rel_scaling_slope=scaling._fit_slope(x, np.log(g.T_star.to_numpy()), fam)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "calibration" / "scaling_T1_vs_Tstar.csv", index=False)
    print("\n== inverse scaling before vs after calibration (rel. dLL change per 10x params), "
          "and how T* grows with size ==")
    print(S.pivot_table(index=["measure", "at"], columns="corpus", values="rel_scaling_slope").round(3).to_string())
