#!/usr/bin/env python
"""Follow-up analyses (run after `run.py analyze`).

  python extras.py --lm gpt2                 # all three
  python extras.py --lm gpt2 --only words    # just the high-surprisal word dump
  python extras.py --lm gpt2 --knot 7 --n-rep 50 --n-readers 7

Outputs in out/results/<lm>/extras/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import extras  # noqa: E402

OUT = Path("out")

p = argparse.ArgumentParser()
p.add_argument("--lm", default="gpt2", help="label of out/surprisal/<lm>.parquet")
p.add_argument("--only", choices=["readers", "divergence", "words"], default=None)
p.add_argument("--knot", type=float, default=7.0, help="hinge for piecewise slopes (bits)")
p.add_argument("--n-readers", type=int, default=7)
p.add_argument("--n-rep", type=int, default=50)
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--by", choices=["source", "corpus"], default="source",
               help="source: machine vs pooled human corpora; corpus: each corpus vs emtec")
p.add_argument("--wild", type=int, default=0, metavar="B",
               help="wild cluster restricted bootstrap with B reps for the piecewise differences "
                    "(use 999; needed when a corpus has few texts, e.g. MECO-en)")
p.add_argument("--emtec-genres", nargs="+", default=None,
               help="keep only these EMTeC genres, e.g. non-fiction article_synopsis summarization")
p.add_argument("--log", action="store_true",
               help="divergence on log RT: a slope difference that survives this is not just "
                    "one corpus's readers being uniformly slower")
a = p.parse_args()

res = OUT / "results" / a.lm
ext = res / "extras"
ext.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(res / "analysis_table.parquet")
texts = pd.read_parquet(OUT / "texts.parquet")
tag = ""
if a.emtec_genres:
    bad = set(a.emtec_genres) - set(d.loc[d.corpus == "emtec", "genre"])
    if bad:
        sys.exit(f"unknown EMTeC genres {bad}; available: {sorted(d.loc[d.corpus == 'emtec', 'genre'].unique())}")
    d = d[(d.corpus != "emtec") | d.genre.isin(a.emtec_genres)]
    tag = "_genres-" + "-".join(sorted(a.emtec_genres))
    ext = res / ("extras" + tag)
    ext.mkdir(parents=True, exist_ok=True)
print(f"[{a.lm}] {len(d):,} words | corpora: {sorted(d.corpus.unique())}"
      + (f" | EMTeC genres: {a.emtec_genres}" if a.emtec_genres else ""))
if "item_id" not in d.columns:
    sys.exit("analysis_table has no item_id: rerun `python run.py harmonize` and `python run.py analyze` first")

if a.only in (None, "readers"):
    print("\n== 1. reader-matched dLL ==")
    words = pd.read_parquet(OUT / "words.parquet")
    if a.emtec_genres:
        words = words[(words.corpus != "emtec") | words.genre.isin(a.emtec_genres)]
    surp = pd.read_parquet(OUT / "surprisal" / f"{a.lm}.parquet")
    raw, summ = extras.reader_matched_dll(words, texts, surp, a.measures,
                                          n_readers=a.n_readers, n_rep=a.n_rep)
    raw.to_csv(ext / "reader_matched_raw.csv", index=False)
    summ.to_csv(ext / "reader_matched_dll.csv", index=False)
    print(summ.round(4).to_string(index=False))

if a.only in (None, "divergence"):
    if d["source" if a.by == "source" else "corpus"].nunique() < 2:
        print("\n(divergence needs >= 2 sources, skipping)")
    else:
        print(f"\n== 2. divergence tests (knot = {a.knot} bits{', log RT' if a.log else ''}) ==")
        dd = d.copy()
        ref = "human"
        if a.by == "corpus":
            dd["source"], ref = dd["corpus"], "emtec"
        if a.log:
            import numpy as np
            for m in a.measures:
                dd[m] = np.log(dd[m].where(dd[m] > 0))
        sfx = ("_log" if a.log else "") + ("_bycorpus" if a.by == "corpus" else "")
        tests, pw, curves = [], [], []
        for m in a.measures:
            o = extras.divergence(dd, m, knot=a.knot, ref=ref, wild=a.wild)
            tests.append(o["tests"])
            pw.append(o["piecewise"])
            curves.append(o["curves"])
        tests, pw, curves = (pd.concat(x, ignore_index=True) for x in (tests, pw, curves))
        tests.to_csv(ext / f"divergence_tests{sfx}.csv", index=False)
        pw.to_csv(ext / f"piecewise_slopes{sfx}.csv", index=False)
        curves.to_csv(ext / f"spline_curves{sfx}.csv", index=False)
        extras.plot_curves(curves, ext / f"spline_curves{sfx}.png", knot=a.knot)
        print(tests.round(4).to_string(index=False))
        print()
        print(pw.round(3).to_string(index=False))

if a.only in (None, "words"):
    print("\n== 3. high-surprisal words ==")
    top, prof = extras.high_surprisal_words(d, texts, knot=a.knot)
    top.to_csv(ext / "top_surprisal_words.csv", index=False)
    prof.to_csv(ext / "high_surprisal_profile.csv", index=False)
    print(prof.round(3).to_string(index=False))
    for c, g in top.groupby("corpus"):
        print(f"\n-- {c}: 15 most surprising --")
        print(g.head(15)[["word", "surprisal", "resid", "context"]].round(1).to_string(index=False))

print(f"\nwrote {ext}/")
