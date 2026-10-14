#!/usr/bin/env python
"""Inverse scaling on LLM-written vs human text. Run after `run.py surprisal` for several LMs.

  python scaling.py                    # all LMs in out/surprisal/ with a recognisable size
  python scaling.py --n-rep 5 --n-boot 200   # quick look

Writes out/scaling/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import scaling  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--n-readers", type=int, default=7, help="readers per text after subsampling (0 = all)")
p.add_argument("--n-rep", type=int, default=10)
p.add_argument("--n-boot", type=int, default=500)
a = p.parse_args()

OUT = Path("out")
res = OUT / "scaling"
res.mkdir(parents=True, exist_ok=True)
surps = {f.stem: pd.read_parquet(f) for f in sorted((OUT / "surprisal").glob("*.parquet"))}
used = {k: scaling.lm_info(k) for k in surps}
print("LMs:", {k: (v[0], f"{v[1] / 1e6:.0f}M") for k, v in used.items() if v})
skipped = [k for k, v in used.items() if not v]
if skipped:
    print("skipped (unknown size):", skipped)
if sum(bool(v) for v in used.values()) < 3:
    sys.exit("need >= 3 LMs with known sizes in out/surprisal/")

words = pd.read_parquet(OUT / "words.parquet")
texts = pd.read_parquet(OUT / "texts.parquet")
pm, summ, diffs = scaling.run(words, texts, surps, a.measures, a.n_readers, a.n_rep, a.n_boot)
pm.to_csv(res / "scaling_points.csv", index=False)
summ.to_csv(res / "scaling_slopes.csv", index=False)
diffs.to_csv(res / "scaling_diffs.csv", index=False)
scaling.plot(pm, res / "scaling.png")

print("\n== Delta-LL x1000 per LM (reader-matched) ==")
print((1000 * pm.pivot_table(index=["measure", "lm"], columns="corpus", values="dLL")).round(2).to_string())
print("\n== scaling slope per corpus (per 10x params; rel = proportion of corpus mean) ==")
print(summ.round(4).to_string(index=False))
print("\n== EMTeC minus human corpus (negative = steeper decline on LLM text) ==")
print(diffs.round(4).to_string(index=False))
print(f"\nwrote {res}/")
