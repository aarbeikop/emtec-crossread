"""Self-vs-other on simulated EMTeC-shaped data.

Reading times depend on a latent 'human' surprisal h. Every scorer sees h + noise.
planted: the generator's own scorer is less noisy on its own texts -> DD > 0 for 'self'.
null:    same noise everywhere, but scorers differ in quality and text sets in difficulty
         -> DD ~ 0 (the double difference must cancel both main effects).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread import selfother  # noqa: E402

SCORERS = ["gpt2", "opt-350m", "pythia-6.9b", "phi2", "mistral-instruct", "mistral-base", "llama2-13b"]
QUALITY = {"gpt2": 1.6, "opt-350m": 2.0, "pythia-6.9b": 1.4, "phi2": 1.8, "mistral-instruct": 1.5,
           "mistral-base": 1.5, "llama2-13b": 1.7}
DIFFICULTY = {"phi2": 0.0, "mistral": 40.0, "wizardlm": 15.0}


def simulate(planted, seed=0, n_items=40):
    rng = np.random.default_rng(seed)
    rows = []
    for it in range(n_items):
        for g in selfother.GENERATORS:
            for dec in ["greedy_search", "beam_search", "sampling", "topk", "topp"]:
                if g == "phi2" and dec == "beam_search":
                    continue
                n = 60
                h = rng.gamma(2, 2.2, n)
                length = rng.integers(1, 12, n)
                zipf = 7 - 0.3 * length + rng.normal(0, .6, n)
                surp = {}
                for s in SCORERS:
                    sd = QUALITY[s]
                    if planted and selfother.RELATION.get((g, s)) == "self":
                        sd = 0.6
                    surp[s] = np.clip(h + rng.normal(0, sd, n), 0, None)
                for subj in range(8):
                    rt = 180 + DIFFICULTY[g] + 6 * length - 8 * zipf + 4 * h + rng.normal(0, 60, n)
                    rt = np.where(rng.random(n) < .25, 0, np.clip(rt, 60, None))
                    for i in range(n):
                        r = dict(subject_id=f"s{subj}", item_id=f"item{it:02d}", model=g,
                                 decoding_strategy=dec, word_id=i, word="w", FFD=0.8 * rt[i],
                                 FPRT=rt[i], TFT=rt[i] * 1.2, word_length_without_punct=length[i],
                                 zipf_freq=zipf[i], task="fiction")
                        for s in SCORERS:
                            r[f"surprisal_{s}"] = surp[s][i]
                            r[f"surprisal_p_{s}"] = surp[s][i]
                        rows.append(r)
    return pd.DataFrame(rows)


def test_self_advantage():
    res = {}
    for name, planted in [("planted", True), ("null", False)]:
        w, scols = selfother.word_table(simulate(planted))
        dd, mat, _, _ = selfother.run(w, scols, ["GD"], "prompt", n_boot=300, by_decoding=False)
        res[name] = dd.set_index(["generator", "scorer"])
        print(name, "\n", dd[["generator", "scorer", "relation", "DD", "ci_lo", "ci_hi", "p_boot"]].round(4))
    for g, s in [("phi2", "phi2"), ("mistral", "mistral-instruct")]:
        assert res["planted"].loc[(g, s), "ci_lo"] > 0
        assert res["null"].loc[(g, s), "ci_lo"] < 0 < res["null"].loc[(g, s), "ci_hi"]


if __name__ == "__main__":
    test_self_advantage()
    print("ok")
