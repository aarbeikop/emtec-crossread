#!/usr/bin/env python
"""GREAT (He et al., ACL SRW 2026; github.com/qwurd231/GREAT) fixation file -> word-level reading measures.

  python tools/great_to_words.py data/great        # reads the repo's CSVs, writes data/great/great_words.csv

GREAT ships one row per fixation (Tobii Pro Spark, 60 Hz) with the word it landed on, so we compute
the standard measures per reading (participant x text) with pymovements' compute_reading_measures:
  FFD  first fixation on the word in first pass        GD   sum of the first-pass run on the word
  TFT  sum of all fixations on the word                skip 1 if the word got no first-pass fixation
  reg_out  1 if the first-pass run ended with a regression (next fixation on an earlier word)
First pass = before any fixation on a word further right. Words never fixated: 0 ms here, which
the config turns into NaN (zero_means_not_fixated: true).

Texts were reused across trial pairs, so many participants read the same text 2-5 times. Only the
FIRST exposure is kept (lowest trial number, which we take to be presentation order; verify with
the authors), because rereading a text changes eye movements.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/great")
fix = pd.read_csv(root / "Fixations_Para.csv", sep="\t")
words = pd.read_csv(root / "Word_Infor.csv", sep="\t")[["SN", "NW", "WORD"]]
texts = pd.read_csv(root / "Text_Data.csv")[["SENTENCEID", "MODEL"]]

fix["order"] = np.arange(len(fix))  # file order = temporal order within a trial
fix["trial_no"] = fix.TRIAL_ID.str.extract(r"^(\d+)")[0].astype(int)
first = fix.groupby(["PARTICIPANT_ID", "SENTENCEID"]).trial_no.transform("min")
n_before = fix[["PARTICIPANT_ID", "SENTENCEID", "TRIAL_ID"]].drop_duplicates().shape[0]
fix = fix[fix.trial_no == first]
n_after = fix[["PARTICIPANT_ID", "SENTENCEID", "TRIAL_ID"]].drop_duplicates().shape[0]
print(f"kept first exposure: {n_after} of {n_before} readings")

# reading measures with pymovements (same implementation the lab uses for EMTeC):
# one reading sequence per participant x text; AOIs = that text's words
import polars as pl
from pymovements.measure.reading import compute_reading_measures

fix["reading"] = fix.PARTICIPANT_ID.astype(str) + "_" + fix.SENTENCEID.astype(str)
fx = pl.from_pandas(fix.sort_values("order").rename(columns={
    "CURRENT_FIX_INTEREST_AREA_ID": "word_idx", "CURRENT_FIX_DURATION": "duration", "order": "onset"})
    [["reading", "word_idx", "duration", "onset"]])
aois = {}
for r, sid in fix[["reading", "SENTENCEID"]].drop_duplicates().itertuples(index=False):
    wl = words[words.SN == sid].sort_values("NW")
    aois[r] = pl.DataFrame({"word_idx": wl.NW.to_numpy(), "word": wl.WORD.astype(str).tolist()})
rm = compute_reading_measures(fx, aois, group_columns=["reading"]).to_pandas()
key = fix[["reading", "PARTICIPANT_ID", "SENTENCEID"]].drop_duplicates()
rm = rm.merge(key, on="reading")
# harmonized schema: GD = first-pass reading time; skip = no first-pass fixation (as Provo/OneStop
# IA_SKIP); reg_out = first-pass regression (FPReg, as EMTeC). 0 ms = not fixated (config sets NaN).
rows = pd.DataFrame(dict(subject_id=rm.PARTICIPANT_ID, text_id=rm.SENTENCEID, word_id=rm.word_index,
                         word=rm.word, FFD=rm.FFD, GD=rm.FPRT, TFT=rm.TFT, skip=1 - rm.FPF,
                         reg_out=rm.FPReg.where(rm.FPF == 1)))
out = rows.merge(texts.rename(columns={"SENTENCEID": "text_id", "MODEL": "model"}), on="text_id")
out.to_csv(root / "great_words.csv", index=False)
print(f"wrote {root / 'great_words.csv'}: {len(out):,} rows, {out.subject_id.nunique()} readers, "
      f"{out.text_id.nunique()} texts, skip rate {out.skip.mean():.2f}, mean GD {out.GD[out.GD > 0].mean():.0f} ms")
