#!/usr/bin/env python
"""OASST-ETC (Lopez-Cardona et al., ETRA 2025; github.com/Telefonica-Scientific-Research/oasstetc)
-> word-level reading measures, computed with pymovements.

  python tools/oasst_to_words.py data/oasstetc      # the cloned repo; writes data/oasstetc/oasst_words.csv

The texts are OpenAssistant (OASST1) assistant replies, which were WRITTEN BY HUMAN VOLUNTEERS, so
this is a human-text corpus in chat-assistant style: the genre-matched control for GREAT.
GP3 HD tracker at 60 Hz; words come from OCR of the screen and fixations were assigned to the
nearest word by the dataset authors (raw_data/<participant>/<session>/vertices/).

Quality filters:
  - trials the authors flagged as discarded (info.csv) are dropped
  - per text, the modal OCR word list defines the words; readings whose OCR differs are dropped
  - texts whose OCR words match the original reply text (from the PsychoPy log) at < --min-sim
    are dropped (OCR errors would corrupt surprisal)
Only ~2 readers per text: use the trial-level mixed models for this corpus (word-level averages
need --min-readers 1 and are very noisy).
"""
from __future__ import annotations

import argparse
import collections
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pymovements.measure.reading import compute_reading_measures

p = argparse.ArgumentParser()
p.add_argument("root", nargs="?", default="data/oasstetc")
p.add_argument("--min-sim", type=float, default=0.9)
a = p.parse_args()
root = Path(a.root)
raw = root / "oasstetc_data" / "raw_data" if (root / "oasstetc_data").exists() else root / "raw_data"

readings, texts_ref = [], {}
for sess in sorted(raw.glob("participant_*/session_*")):
    subj = sess.parent.name
    if not (sess / "info.csv").exists():
        continue
    info = pd.read_csv(sess / "info.csv", sep=";")
    bad = set(info.loc[info.discarted.astype(str) == "True", "trial"].map(lambda t: f"{t:g}" if isinstance(t, float) else str(t)))
    for f in sess.glob("*FINAL.csv"):  # original reply texts
        log = pd.read_csv(f, sep=";;;", engine="python", usecols=["n_resp", "resp_text"])
        for r in log.dropna().itertuples():
            texts_ref.setdefault(str(r.n_resp), str(r.resp_text).strip('"'))
    for fa in sorted((sess / "vertices").glob("fixations_assigned_*.csv")):
        trial = re.search(r"fixations_assigned_(.+)\.csv", fa.name).group(1)
        wc = sess / "vertices" / f"word_cor_image_fixations_{trial}.csv"
        if trial in bad or not wc.exists():
            continue
        try:
            fx = pd.read_csv(fa, sep=";")
        except pd.errors.EmptyDataError:
            continue
        if not {"ID", "word_number", "duration"} <= set(fx.columns):
            continue
        words = tuple(str(w) for w in pd.read_csv(wc, sep=";", dtype={"text": str}).text.fillna(""))
        readings.append(dict(subj=subj, sess=sess.name, trial=trial, words=words, fix=fx))

modal = {}
for t in {r["trial"] for r in readings}:
    modal[t] = collections.Counter(r["words"] for r in readings if r["trial"] == t).most_common(1)[0][0]
sim = {t: difflib.SequenceMatcher(None, " ".join(w).split(), texts_ref.get(t, "").split()).ratio()
       for t, w in modal.items()}
keep = [r for r in readings if r["words"] == modal[r["trial"]] and sim[r["trial"]] >= a.min_sim]
print(f"readings: {len(readings)} -> {len(keep)} kept "
      f"({sum(s < a.min_sim for s in sim.values())} of {len(sim)} texts dropped for OCR mismatch)")

fx_rows, aois = [], {}
for r in keep:
    rid = f"{r['subj']}|{r['sess']}|{r['trial']}"
    f = r["fix"].sort_values("ID")
    f = f[f.word_number.notna()]
    fx_rows.append(pd.DataFrame({"reading": rid, "word_idx": f.word_number.astype(int).to_numpy(),
                                 "duration": (f.duration * 1000).to_numpy(), "onset": f.ID.to_numpy()}))
    aois[rid] = pl.DataFrame({"word_idx": np.arange(len(r["words"])), "word": list(r["words"])})
fx_all = pl.from_pandas(pd.concat(fx_rows, ignore_index=True))
aois = {k: v for k, v in aois.items() if k in set(fx_all["reading"].to_list())}
rm = compute_reading_measures(fx_all, aois, group_columns=["reading"]).to_pandas()
parts = rm.reading.str.split("|", expand=True)
out = pd.DataFrame(dict(subject_id=parts[0], text_id=parts[2], word_id=rm.word_index, word=rm.word,
                        FFD=rm.FFD, GD=rm.FPRT, TFT=rm.TFT, skip=1 - rm.FPF,
                        reg_out=rm.FPReg.where(rm.FPF == 1)))
out.to_csv(root / "oasst_words.csv", index=False)
print(f"wrote {root / 'oasst_words.csv'}: {len(out):,} rows, {out.subject_id.nunique()} readers, "
      f"{out.text_id.nunique()} texts, {out.groupby('text_id').subject_id.nunique().mean():.1f} readers/text, "
      f"skip rate {out.skip.mean():.2f}, mean GD {out.GD[out.GD > 0].mean():.0f} ms")
