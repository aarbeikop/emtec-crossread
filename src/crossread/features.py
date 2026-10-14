"""Word-level predictors + word-level aggregated reading measures."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PUNCT = re.compile(r"[^\w]", re.UNICODE)


def word_predictors(texts: pd.DataFrame, surprisal: pd.DataFrame) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    w = texts.merge(surprisal, on=["corpus", "text_id", "word_idx"], how="left")
    clean = w["word"].str.replace(PUNCT, "", regex=True)
    w["length"] = clean.str.len()
    w["has_punct"] = (w["word"] != clean).astype(int)
    w["zipf"] = [zipf_frequency(c.lower(), lang) if c else 0.0 for c, lang in zip(clean, w["language"])]
    w = w.sort_values(["corpus", "text_id", "word_idx"])
    g = w.groupby(["corpus", "text_id"])
    for c in ["surprisal", "entropy", "length", "zipf"]:
        w[f"prev_{c}"] = g[c].shift(1)
    w["is_first"] = (g.cumcount() == 0).astype(int)
    w["is_last"] = (g.cumcount(ascending=False) == 0).astype(int)
    return w


def aggregate_rts(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over readers per word. Durations: mean over readers who fixated."""
    keys = ["corpus", "text_id", "word_idx"]
    meta_cols = ["source", "model", "decoding", "genre"] + (["item_id"] if "item_id" in df else [])
    meta = df.groupby(keys)[meta_cols].first()
    df = df.assign(
        # rereading = total time minus first pass, for readers who read the word in first pass
        reread=(df["TFT"] - df["GD"]).where(df["GD"].notna()),
        reg_out=df["reg_out"] if "reg_out" in df else np.nan)
    agg = df.groupby(keys).agg(FFD=("FFD", "mean"), GD=("GD", "mean"), TFT=("TFT", "mean"),
                               skip_rate=("skip", "mean"), reg_rate=("reg_out", "mean"),
                               reread=("reread", "mean"), n_readers=("subject_id", "nunique"))
    return agg.join(meta).reset_index()


def analysis_table(rt_words: pd.DataFrame, preds: pd.DataFrame, drop_punct: bool = True,
                   max_surprisal: float = 30.0) -> pd.DataFrame:
    """Standard exclusions (Wilcox et al. 2023 style): first/last word, punctuation, extreme surprisal."""
    d = rt_words.merge(preds.drop(columns=["word"]), on=["corpus", "text_id", "word_idx"], how="inner")
    keep = (d["is_first"] == 0) & (d["is_last"] == 0) & d["prev_surprisal"].notna() & d["surprisal"].notna()
    keep &= d["surprisal"] <= max_surprisal
    if drop_punct:
        keep &= (d["has_punct"] == 0)
    return d[keep].reset_index(drop=True)
