"""Config-driven loader: maps any word-level eye-tracking table into schema.py.

Every corpus has its own column names. Instead of one loader per corpus,
configs/corpora.yaml maps canonical names -> raw column names. Check them
against your actual files: the defaults are best guesses, marked VERIFY.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema


def read_table(path: str | Path, obj: str | None = None, sep: str | None = None,
               usecols=None) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in {".rda", ".rdata", ".rds"}:
        import rdata  # pure-python reader for R data files (data.frames only)

        res = rdata.read_rds(path) if suf == ".rds" else rdata.read_rda(path)
        if isinstance(res, dict):
            res = res[obj] if obj else next(iter(res.values()))
        return pd.DataFrame(res)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if sep is None:  # sniff tab vs comma from the header line
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.readline()
        sep = "\t" if head.count("\t") > head.count(",") else ","
    if usecols is not None:  # only read what we need (OneStop's IA report has 157 columns)
        head_cols = pd.read_csv(path, sep=sep, nrows=0).columns
        usecols = [c for c in usecols if c in head_cols]
    return pd.read_csv(path, sep=sep, low_memory=False, usecols=usecols)


def _map(raw: pd.DataFrame, cols: dict, name: str) -> pd.DataFrame:
    missing = {k: v for k, v in cols.items() if v not in raw.columns}
    if missing:
        raise KeyError(
            f"[{name}] columns not found: {missing}\n"
            f"available columns: {list(raw.columns)}\n"
            f"-> fix the mapping in configs/corpora.yaml"
        )
    return pd.DataFrame({k: raw[v].values for k, v in cols.items()})


def load_corpus(name: str, cfg: dict, root: Path = Path(".")) -> pd.DataFrame:
    need = None
    if Path(cfg["path"]).suffix.lower() in {".csv", ".tsv", ".txt"} and not cfg.get("join"):
        need = list(cfg["columns"].values()) + list((cfg.get("filter") or {}).keys())
    raw = read_table(root / cfg["path"], cfg.get("r_object"), cfg.get("sep"), usecols=need)

    for col, val in (cfg.get("filter") or {}).items():
        raw = raw[raw[col].isin(val if isinstance(val, list) else [val])]
    if raw.empty:
        raise ValueError(f"[{name}] no rows left after filter {cfg.get('filter')}")

    # optional join (e.g. EMTeC genre lives in stimuli.csv)
    if cfg.get("join"):
        j = cfg["join"]
        right = read_table(root / j["path"])
        keep = list(j["keys"].values()) + list(j["cols"].values())
        right = right[keep].drop_duplicates(list(j["keys"].values()))
        raw = raw.merge(right, left_on=list(j["keys"].keys()), right_on=list(j["keys"].values()),
                        how="left", suffixes=("", "_join"))
        for canon, rawcol in j["cols"].items():
            raw[f"__{canon}"] = raw[rawcol]
            cfg["columns"][canon] = f"__{canon}"

    df = _map(raw, cfg["columns"], name)

    # text id = concatenation of the configured key columns (EMTeC: item x model x decoding)
    keys = cfg.get("text_keys", ["text"])
    df["text_id"] = name + ":" + df[keys].astype(str).agg("|".join, axis=1)
    # item = the underlying prompt/passage. EMTeC shows each item in 15 model x decoding
    # versions with overlapping content, so resampling / clustering must happen at this level.
    item_keys = cfg.get("item_keys", keys[:1])
    df["item_id"] = name + ":" + df[item_keys].astype(str).agg("|".join, axis=1)
    df = df.drop(columns=[k for k in keys if k not in ("model", "decoding")])

    df["corpus"] = name
    df["language"] = cfg.get("language", "en")
    df["source"] = cfg.get("source", "human")
    default = "human" if cfg.get("source", "human") == "human" else "NA"
    for c in ["model", "decoding"]:
        if c not in df.columns:
            df[c] = cfg.get(c, default)
    if "genre" not in df.columns:
        df["genre"] = cfg.get("genre", "NA")
    for c in ["model", "decoding", "genre"]:
        df[c] = df[c].astype(str)

    for m in ["FFD", "GD", "TFT"]:
        df[m] = pd.to_numeric(df[m], errors="coerce")
        if cfg.get("zero_means_not_fixated", False):
            df.loc[df[m] <= 0, m] = np.nan

    if "skip" in df.columns:
        df["skip"] = pd.to_numeric(df["skip"], errors="coerce")
    else:  # derive: no first-pass reading time -> skipped
        df["skip"] = df["GD"].isna().astype(float)
    # a word with GD defined was not skipped; make the two consistent
    df.loc[df["skip"] == 1, ["FFD", "GD"]] = np.nan
    # optional: first-pass regression out of the word (0/1); undefined if skipped in first pass
    if "reg_out" in df.columns:
        df["reg_out"] = pd.to_numeric(df["reg_out"], errors="coerce")
        df.loc[(df["skip"] == 1) | df["GD"].isna(), "reg_out"] = np.nan
    else:
        df["reg_out"] = np.nan

    df["word_idx"] = pd.to_numeric(df["word_idx"], errors="coerce")
    n_nan = int(df["word_idx"].isna().sum())
    df = df.dropna(subset=["word_idx"])
    df["word_idx"] = (df["word_idx"] - df.groupby("text_id")["word_idx"].transform("min")).astype(int)
    df["word"] = df["word"].astype(str)
    if cfg.get("fix_excel_booleans", True):  # Excel turned "true"/"false" into TRUE/FALSE (Provo)
        df.loc[df["word"].isin(["TRUE", "FALSE"]), "word"] = df["word"].str.lower()
    _report_gaps(df, name, n_nan)
    df["subject_id"] = name + ":" + df["subject_id"].astype(str)

    df = df.drop_duplicates(["text_id", "subject_id", "word_idx"])
    return schema.check(df, name)


def _report_gaps(df, name, n_nan):
    """Words missing from the eye-tracking table are missing from the text the LM sees."""
    pos = df.groupby("text_id")["word_idx"].agg(["max", "nunique"])
    pos["missing"] = pos["max"] + 1 - pos["nunique"]
    bad = pos[pos["missing"] > 0]
    if n_nan or len(bad):
        print(f"  [{name}] WARNING: {n_nan} rows without word index dropped; "
              f"{len(bad)}/{len(pos)} texts have gaps ({int(bad['missing'].sum())} word positions "
              f"missing, e.g. {list(bad.index[:3])}). Surprisal near gaps is computed on an "
              f"incomplete text: see README 'Provo gaps'.")


def texts_from_words(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild each text as its ordered word list (modal word string per position)."""
    w = (df.assign(word=df["word"].fillna("").astype(str))
           .groupby(["corpus", "language", "text_id", "word_idx"])["word"]
           .agg(lambda s: s.mode().iat[0]).reset_index())
    return w.sort_values(["corpus", "text_id", "word_idx"])
