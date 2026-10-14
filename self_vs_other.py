#!/usr/bin/env python
"""Self vs other surprisal on EMTeC (see src/crossread/selfother.py for the design).

  python self_vs_other.py                       # prompt-conditioned surprisal, 2000 item bootstraps
  python self_vs_other.py --variant text        # surprisal on the text alone
  python self_vs_other.py --n-boot 200          # quick look

Reads data/emtec/reading_measures_corrected.csv directly; writes out/self_vs_other/<variant>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import loaders, selfother  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/emtec/reading_measures_corrected.csv")
p.add_argument("--variant", choices=["prompt", "text"], default="prompt")
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--no-decoding", action="store_true", help="skip the per-decoding-strategy split")
a = p.parse_args()

out = Path("out/self_vs_other") / a.variant
out.mkdir(parents=True, exist_ok=True)

rm = loaders.read_table(a.data)
w, scols = selfother.word_table(rm)
print(f"{len(w):,} words | generators: {sorted(w.model.unique())} | "
      f"decoding: {sorted(w.decoding_strategy.unique())} | {w.item_id.nunique()} items")

ms = selfother.mean_surprisal_matrix(w, scols, a.variant)
ms.to_csv(out / "mean_surprisal_matrix.csv", index=False)
print("\n== mean surprisal (bits) of each scorer on each generator's texts ==")
print(ms.pivot(index="generator", columns="scorer", values="mean_surprisal").round(2).to_string())

print(f"\n== crossed dLL + self advantage ({a.variant}, {a.n_boot} item bootstraps) ==")
dd, mat, dd_dec, mat_dec = selfother.run(w, scols, a.measures, a.variant, a.n_boot,
                                         by_decoding=not a.no_decoding)
mat.to_csv(out / "dll_matrix.csv", index=False)
dd.to_csv(out / "self_advantage.csv", index=False)
if len(dd_dec):
    dd_dec.to_csv(out / "self_advantage_by_decoding.csv", index=False)
    mat_dec.to_csv(out / "dll_matrix_by_decoding.csv", index=False)
for m in a.measures:
    selfother.plot_matrix(mat, out / f"dll_matrix_{m}.png", m)
    print(f"\n-- {m}: dLL x1000 (rows = whose texts, cols = scorer) --")
    print((1000 * mat[mat.measure == m].pivot(index="generator", columns="scorer", values="dLL")).round(2).to_string())

fmt = lambda x: x.assign(DD_x1000=1000 * x.DD, lo=1000 * x.ci_lo, hi=1000 * x.ci_hi)[
    ["subset", "measure", "generator", "scorer", "relation", "DD_x1000", "lo", "hi", "p_boot"]].round(3)
print("\n== self advantage DD (x1000; >0 = scorer gains more on its own/family texts) ==")
print(fmt(dd).to_string(index=False))
if len(dd_dec):
    print("\n== by decoding strategy ==")
    print(fmt(dd_dec[dd_dec.relation == "self"]).to_string(index=False))
print(f"\nwrote {out}/")
