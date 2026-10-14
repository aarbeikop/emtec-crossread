#!/usr/bin/env python
"""Joint bootstrap: does temperature calibration remove inverse scaling on human but not on LLM text?
Needs surprisal at >= 3 temperatures for >= 3 LMs of known size.

  python -W ignore calib_scaling.py                 # 1000 item bootstraps
  python -W ignore calib_scaling.py --n-boot 200    # quick look
Writes out/calibration/joint/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import calibration  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT", "reg_rate", "reread"])
p.add_argument("--n-boot", type=int, default=1000)
p.add_argument("--emtec-genres", nargs="+", default=None,
               help="restrict EMTeC to these genres, e.g. non-fiction article_synopsis summarization")
p.add_argument("--split", choices=["decoding", "typicality", "generator"], default=None,
               help="decoding: EMTeC by decoding strategy; typicality: every corpus by text-level "
                    "LM-typicality tertiles (human corpora = control)")
p.add_argument("--typ-lm", default="gpt2", help="LM (label in out/surprisal) defining typicality")
p.add_argument("--words", choices=["all", "single", "multi"], default="all",
               help="single: only words that are ONE token under every LM (Liu et al. 2024 attribute "
                    "temperature gains to multi-token words); multi: words split by at least one LM")
p.add_argument("--link", choices=["linear", "quadratic"], default="linear",
               help="quadratic: add squared surprisal terms (super/sub-linear effects, Shain et al. 2024)")
a = p.parse_args()

from crossread import analysis  # noqa: E402
if a.link == "quadratic":
    analysis.SURP = ["surprisal", "prev_surprisal", "surprisal:surprisal", "prev_surprisal:prev_surprisal"]

OUT = Path("out")
tag = ("_genres-" + "-".join(sorted(a.emtec_genres)) if a.emtec_genres else "") + \
      (f"_split-{a.split}" if a.split else "") + \
      (f"_words-{a.words}" if a.words != "all" else "") + (f"_link-{a.link}" if a.link != "linear" else "")
res = OUT / "calibration" / ("joint" + tag)
res.mkdir(parents=True, exist_ok=True)
surps = {f.stem: pd.read_parquet(f) for f in sorted((OUT / "surprisal").glob("*.parquet"))}
words = pd.read_parquet(OUT / "words.parquet")
texts = pd.read_parquet(OUT / "texts.parquet")
word_keys = None
if a.words != "all":
    K = ["corpus", "text_id", "word_idx"]
    nt = pd.concat([s.set_index(K)["n_tokens"].rename(l) for l, s in surps.items()], axis=1)
    keep = (nt == 1).all(axis=1) if a.words == "single" else (nt > 1).any(axis=1)
    word_keys = nt.index[keep]
    print(f"--words {a.words}: {keep.mean():.0%} of words kept")
if a.split or a.emtec_genres or word_keys is not None or a.link != "linear":
    summ, con = calibration.grouped_scaling(words, texts, surps, a.measures, a.n_boot,
                                            emtec_genres=a.emtec_genres, split=a.split, typ_lm=a.typ_lm,
                                            word_keys=word_keys)
    summ.to_csv(res / "scaling_summary.csv", index=False)
    con.to_csv(res / "scaling_contrasts.csv", index=False)
    cols = [c for c in ["group", "n_texts", "mean_surprisal", "measure", "stat", "estimate", "lo", "hi", "p_boot"]
            if c in summ.columns]
    print("\n== per group (rel_T1 = inverse scaling at T=1, rel_Tstar = after calibration; < 0 = bigger LMs fit worse) ==")
    print(summ[summ.stat.isin(["rel_T1", "rel_Tstar"])][cols].round(3).to_string(index=False))
    print("\n== contrasts (< 0 = more inverse scaling in the first group) ==")
    print(con.round(3).to_string(index=False))
    print(f"\nwrote {res}/")
    sys.exit(0)
summ, con, per_lm = calibration.joint_scaling(words, texts, surps, a.measures, a.n_boot)
summ.to_csv(res / "scaling_summary.csv", index=False)
con.to_csv(res / "scaling_contrasts.csv", index=False)
per_lm.to_csv(res / "per_lm.csv", index=False)

print("\n== T* per LM (point) ==")
print(per_lm.pivot_table(index=["measure", "lm"], columns="corpus", values="T_star").to_string())
print("\n== per corpus: rel_T1 = inverse scaling at T=1, rel_Tstar = left after calibration, "
      "calib = removed by calibration, logT_slope = T* growth per 10x params ==")
print(summ.round(3).to_string(index=False))
print("\n== EMTeC minus human corpus (rel_Tstar < 0: more inverse scaling survives calibration on LLM text) ==")
print(con.round(3).to_string(index=False))
print(f"\nwrote {res}/")
