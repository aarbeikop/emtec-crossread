"""Inverse scaling on simulated corpora. Readers respond to a latent 'human' surprisal h.
Fake LM k sees h + noise whose sd grows with model size (bigger = worse fit, as in Oh & Schuler).
planted: noise grows faster on the machine corpus -> relative slope more negative there.
null:    same growth everywhere, but corpora differ in reader count and RT noise
         -> relative slopes equal (the scale-free version must cancel those differences)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from wordfreq import top_n_list

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread import scaling  # noqa: E402

VOCAB = top_n_list("en", 5000)
LMS = {"EleutherAI__pythia-70m": 70e6, "EleutherAI__pythia-410m": 410e6,
       "EleutherAI__pythia-1.4b": 1.4e9, "gpt2": 124e6, "gpt2-large": 774e6}


def simulate(planted, seed=0):
    rng = np.random.default_rng(seed)
    rows, surp = [], {k: [] for k in LMS}
    for corpus, n_texts, n_sub, noise in [("emtec", 60, 8, 60), ("provo", 40, 20, 90)]:
        for t in range(n_texts):
            tid, n = f"{corpus}:{t}", 70
            words = [VOCAB[i] for i in rng.integers(0, len(VOCAB), n)]
            h = rng.gamma(2, 2.5, n)
            for k, size in LMS.items():
                grow = np.log10(size / 50e6) * (2.0 if (planted and corpus == "emtec") else 0.8)
                s = np.clip(h + rng.normal(0, 1 + grow, n), 0, None)
                surp[k].append(pd.DataFrame(dict(corpus=corpus, text_id=tid, word_idx=range(n), surprisal=s,
                                                 entropy=np.nan, n_tokens=1)))
            for subj in range(n_sub):
                gd = 200 + 6 * h + rng.normal(0, noise, n)
                rows.append(pd.DataFrame(dict(corpus=corpus, language="en", text_id=tid, item_id=tid,
                                              subject_id=f"{corpus}:{subj}", word_idx=range(n), word=words,
                                              FFD=gd * .8, GD=gd, TFT=gd * 1.2, skip=0.0,
                                              source="machine" if corpus == "emtec" else "human",
                                              model="m", decoding="d", genre="g")))
    words = pd.concat(rows, ignore_index=True)
    texts = words.drop_duplicates(["text_id", "word_idx"])[["corpus", "language", "text_id", "word_idx", "word"]]
    return words, texts, {k: pd.concat(v, ignore_index=True) for k, v in surp.items()}


def test_scaling():
    out = {}
    for name, planted in [("planted", True), ("null", False)]:
        w, t, s = simulate(planted)
        pm, summ, diffs = scaling.run(w, t, s, ["GD"], n_readers=7, n_rep=2, n_boot=200)
        out[name] = diffs.query("kind == 'rel'").iloc[0]
        print(name, "\n", summ.round(4), "\n", diffs.round(4))
    assert out["planted"].hi < 0
    assert out["null"].lo < 0 < out["null"].hi


if __name__ == "__main__":
    test_scaling()
    print("ok")
