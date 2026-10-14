"""Fake EMTeC / Provo / MECO files with the configured raw column names.

RTs are generated from unigram surprisal with a known slope per corpus
(EMTeC 4 ms/bit, humans 6 ms/bit), so `--lm unigram` should recover a
smaller slope for EMTeC. Only for checking the plumbing.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from wordfreq import top_n_list, word_frequency

rng = np.random.default_rng(1)
VOCAB = top_n_list("en", 3000)
OUT = Path(__file__).parent / "synthetic_data"


def text(n):
    # zipfian-ish sampling, some punctuation
    w = [VOCAB[min(int(rng.pareto(1.1) * 20), len(VOCAB) - 1)] for _ in range(n)]
    w[0] = w[0].capitalize()
    w[-1] += "."
    return w


def readers(words, n_sub, slope, base=180):
    rows = []
    for s in range(n_sub):
        for i, w in enumerate(words):
            surp = -math.log2(max(word_frequency(w.strip(".").lower(), "en"), 1e-9))
            skip = rng.random() < (0.5 if len(w) < 4 else 0.15)
            gd = np.nan if skip else base + slope * surp + 8 * len(w) + rng.normal(0, 60)
            ffd = np.nan if skip else 0.8 * gd
            tft = (0 if skip else gd) + (rng.random() < 0.2) * rng.gamma(2, 80)
            rows.append((s, i, w, ffd, gd, tft if tft > 0 else np.nan, int(skip)))
    return rows


def main():
    for d in ["emtec", "provo", "meco"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    # EMTeC: 6 items x 3 models x 5 decodings, 8 readers each
    rm, stim = [], []
    for item in range(6):
        for model in ["phi2", "mistral", "wizardlm"]:
            for dec in ["greedy_search", "beam_search", "sampling", "topk", "topp"]:
                words = text(rng.integers(60, 90))
                for s, i, w, ffd, gd, tft, sk in readers(words, 8, slope=4):
                    rm.append(dict(subject_id=f"ET_{s:02d}", item_id=f"item{item:02d}", model=model,
                                   decoding_strategy=dec, word_id=i, word=w,
                                   FFD=0 if np.isnan(ffd) else ffd, FPRT=0 if np.isnan(gd) else gd,
                                   TFT=0 if np.isnan(tft) else tft))
                stim.append(dict(item_id=f"item{item:02d}", model=model, decoding_strategy=dec,
                                 task=["fiction", "poetry", "summarization"][item % 3]))
    pd.DataFrame(rm).to_csv(OUT / "emtec/reading_measures_corrected.csv", index=False)
    pd.DataFrame(stim).to_csv(OUT / "emtec/stimuli.csv", index=False)

    pv = []
    for t in range(20):
        for s, i, w, ffd, gd, tft, sk in readers(text(55), 10, slope=6):
            pv.append(dict(Participant_ID=f"P{s}", Text_ID=t + 1, Word_Number=i + 1, Word=w,
                           IA_FIRST_FIXATION_DURATION=ffd, IA_FIRST_RUN_DWELL_TIME=gd,
                           IA_DWELL_TIME=tft, IA_SKIP=sk))
    pd.DataFrame(pv).to_csv(OUT / "provo/Provo_Corpus-Eyetracking_Data.csv", index=False)

    mc = []
    for t in range(12):
        words = text(150)
        for lang in ["en", "de"]:
            for s, i, w, ffd, gd, tft, sk in readers(words, 8 if lang == "en" else 2, slope=6):
                mc.append({"lang": lang, "subid": f"{lang}_{s}", "trialid": t, "ianum": i + 1, "ia": w,
                           "firstfix.dur": ffd, "firstrun.dur": gd, "dur": tft, "skip": sk})
    pd.DataFrame(mc).to_csv(OUT / "meco/joint_data_trimmed.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
