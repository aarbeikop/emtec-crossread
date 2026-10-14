#!/usr/bin/env python
"""Export a trial-level table (one row per reader x word) for the mixed models in r/.

  python export_trials.py              # -> out/trials/trials.csv + lms.csv

Rows: words that pass the usual exclusions (first/last word, punctuation, surprisal <= 30)
under EVERY LM, and that the reader fixated (at least one duration measure present).
Columns:
  ids          corpus, source, text_id, item_id, subject_id, word_idx, decoding, genre, typ
  measures     FFD, GD, TFT (ms; NaN = not fixated in that measure)
  baseline     length, zipf, prev_length, prev_zipf
  per LM key   s__<key>, ps__<key>                       surprisal / previous-word surprisal at T=1
               s__<key>__<M>, ps__<key>__<M>             at that LM's T* for measure M, corpus-specific
                                                         (from out/calibration/<lm>/tstar.csv, if present)
typ = tertile of mean text surprisal under --typ-lm within corpus (high = most predictable),
the same split as calib_scaling.py --split typicality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import features, scaling  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--typ-lm", default="gpt2")
p.add_argument("--no-tstar", action="store_true")
a = p.parse_args()

OUT = Path("out")
res = OUT / "trials"
res.mkdir(parents=True, exist_ok=True)
words = pd.read_parquet(OUT / "words.parquet")
texts = pd.read_parquet(OUT / "texts.parquet")
surps = {f.stem: pd.read_parquet(f) for f in sorted((OUT / "surprisal").glob("*.parquet"))}
labels = [l for l in surps if scaling.lm_info(l)]
if not labels:
    sys.exit("no LMs with known size in out/surprisal/")
key = {l: l.split("__")[-1] for l in labels}
K = ["corpus", "text_id", "word_idx"]

# which words survive the exclusions under every LM (word-level, so use the aggregated table)
rt = features.aggregate_rts(words)
common, base = None, None
for l in labels:
    t = features.analysis_table(rt, features.word_predictors(texts, surps[l])).set_index(K)
    common = t.index if common is None else common.intersection(t.index)
    if base is None:
        base = t[["length", "zipf", "prev_length", "prev_zipf"]]
base = base.loc[common].reset_index()

# per-LM surprisal columns (T=1 and T*), with previous-word values shifted within text
tstar = {}
if not a.no_tstar:
    for l in labels:
        f = OUT / "calibration" / l / "tstar.csv"
        if f.exists():
            e = pd.read_csv(f)
            e = e[e.subset == "all"]
            tstar[l] = {(r.corpus, r.measure): r.T_star for r in e.itertuples()}
        else:
            print(f"[{l}] no {f}; T* columns skipped for this LM")

cols = base
for l in labels:
    s = surps[l].sort_values(K)
    g = s.groupby(["corpus", "text_id"])
    new = s[K].copy()
    new[f"s__{key[l]}"] = s["surprisal"].to_numpy()
    new[f"ps__{key[l]}"] = g["surprisal"].shift(1).to_numpy()
    if l in tstar:
        for m in a.measures:
            sc = np.full(len(s), np.nan)
            for c in s.corpus.unique():
                t = tstar[l].get((c, m))
                if t is None:
                    continue
                col = "surprisal" if t == 1 else f"surprisal_T{t:g}"
                if col not in s:
                    print(f"[{l}] column {col} missing, skipping T* for {c}/{m}")
                    continue
                sc[(s.corpus == c).to_numpy()] = s.loc[s.corpus == c, col].to_numpy()
            new[f"s__{key[l]}__{m}"] = sc
            new[f"ps__{key[l]}__{m}"] = pd.Series(sc, index=s.index).groupby(
                [s.corpus, s.text_id]).shift(1).to_numpy()
    cols = cols.merge(new, on=K, how="left")

# text typicality tertiles (within corpus)
ms = surps[a.typ_lm if a.typ_lm in surps else labels[0]].groupby("text_id")["surprisal"].mean()
tx = words[["corpus", "text_id"]].drop_duplicates()
tx["typ_val"] = tx.text_id.map(ms)
tx["typ"] = tx.groupby("corpus")["typ_val"].transform(
    lambda v: pd.qcut(v.rank(method="first"), 3, labels=["high", "mid", "low"]).astype(str))

keep = ["corpus", "source", "text_id", "subject_id", "word_idx", "decoding", "genre", "model"] + \
       (["item_id"] if "item_id" in words else []) + a.measures
w = words.reindex(columns=keep)
w = w[w[a.measures].notna().any(axis=1)]
d = w.merge(cols, on=K, how="inner").merge(tx[["text_id", "typ"]], on="text_id", how="left")
if "item_id" not in d:
    d["item_id"] = d["text_id"]
d["item_id"] = d["item_id"].fillna(d["text_id"]).astype(str)

num = d.select_dtypes("float").columns
d[num] = d[num].round(4)
print(f"writing {len(d):,} rows ...", flush=True)
import pyarrow as pa
import pyarrow.csv as pacsv
pacsv.write_csv(pa.Table.from_pandas(d, preserve_index=False), res / "trials.csv")  # ~50x faster than to_csv
pd.DataFrame([dict(key=key[l], label=l, family=scaling.lm_info(l)[0], params=scaling.lm_info(l)[1],
                   has_tstar=l in tstar) for l in labels]).to_csv(res / "lms.csv", index=False)
print(d.groupby("corpus").agg(rows=("word_idx", "size"), readers=("subject_id", "nunique"),
                              texts=("text_id", "nunique"), items=("item_id", "nunique")).to_string())
print(f"wrote {res/'trials.csv'} ({len(d):,} rows, {d.shape[1]} columns) and lms.csv")
