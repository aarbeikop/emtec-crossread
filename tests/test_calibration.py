"""Toy world with real softmax temperatures. Each word position has 'human' logits z.
Readers pay b * (-log softmax(z)[w]). The LM's logits are kappa * z (kappa > 1 = over-confident).
Surprisal at temperature T uses softmax(kappa z / T), so the best fit sits near T = kappa.
machine corpus: kappa = 2.5;  human corpus: kappa = 1.3  ->  T*(emtec) > T*(provo)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from wordfreq import top_n_list

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread import calibration  # noqa: E402

VOCAB = top_n_list("en", 5000)
TEMPS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    rows, surp = [], []
    for corpus, kappa, n_texts, n_sub in [("emtec", 2.5, 50, 8), ("provo", 1.3, 40, 12)]:
        for t in range(n_texts):
            tid, n = f"{corpus}:{t}", 80
            z = rng.normal(0, 2.0, (n, 40))
            w = np.array([rng.choice(40, p=np.exp(lsm(zz))) for zz in z])
            h = -lsm(z)[np.arange(n), w] / np.log(2)
            s = {f"surprisal_T{T:g}" if T != 1 else "surprisal": -lsm(kappa * z / T)[np.arange(n), w] / np.log(2)
                 for T in TEMPS}
            surp.append(pd.DataFrame(dict(corpus=corpus, text_id=tid, word_idx=range(n), entropy=np.nan, **s)))
            words = [VOCAB[i] for i in rng.integers(0, len(VOCAB), n)]
            for subj in range(n_sub):
                gd = 200 + 8 * h + rng.normal(0, 40, n)
                # late measures: on the machine corpus readers ALSO pay for what the LM misses
                late = h if corpus == "provo" else h + 0.8 * np.maximum(h - (-lsm(kappa * z)[np.arange(n), w] / np.log(2)), 0)
                reg = (rng.random(n) < 1 / (1 + np.exp(-(-2.5 + 0.25 * late)))).astype(float)
                tft = gd + reg * rng.gamma(2, 60, n) + 10 * late
                rows.append(pd.DataFrame(dict(corpus=corpus, language="en", text_id=tid, item_id=f"{corpus}:{t % 25}",
                                              subject_id=f"{corpus}:{subj}", word_idx=range(n), word=words,
                                              FFD=gd * .8, GD=gd, TFT=tft, skip=0.0, reg_out=reg,
                                              source="machine" if corpus == "emtec" else "human",
                                              model="m", decoding=["greedy_search", "sampling"][t % 2], genre="g")))
    w = pd.concat(rows, ignore_index=True)
    texts = w.drop_duplicates(["text_id", "word_idx"])[["corpus", "language", "text_id", "word_idx", "word"]]
    return w, texts, pd.concat(surp, ignore_index=True)


def test_tstar():
    w, t, s = simulate()
    est, con, curves, tabs, common = calibration.tstar(w, t, s, ["GD"], n_boot=200)
    print(est[["corpus", "subset", "T_star", "T_lo", "T_hi", "gain_pct"]])
    print(con)
    e = est.set_index(["corpus", "subset"])
    assert 2.0 <= e.loc[("emtec", "all"), "T_star"] <= 3.0
    assert 1.0 <= e.loc[("provo", "all"), "T_star"] <= 1.5
    c = con.set_index("contrast").loc["emtec - provo"]
    assert c.lo > 0
    sd = calibration.per_sd_slopes(tabs, common, est, ["GD"], n_boot=50)
    print(sd.round(2))
    # full measure set incl. stage contrasts + hotspots
    est2, con2, _, tabs2, common2 = calibration.tstar(w, t, s, ["skip_rate", "FFD", "GD", "reg_rate", "reread"], n_boot=100)
    print(est2[est2.subset == "all"][["corpus", "measure", "T_star", "T_lo", "T_hi"]])
    print(con2[con2.measure == "late-early"].round(3).to_string(index=False))
    hs, ex = calibration.hotspots(tabs2[1.0].loc[common2], t, thresh=1.0, n_boot=100)
    print(hs[["corpus", "subset", "n_low", "resid_reread_low", "resid_reread_low_lo", "resid_reread_low_hi"]].round(2))
    assert {"emtec", "provo"} <= set(hs.corpus) and len(ex)




def simulate_scaling(seed=1, by_decoding=False):
    """Several 'LMs' of growing size. Human corpus: LM logits = kappa(size) * z, so temperature
    can undo it. Machine corpus: additionally a size-growing distortion that is NOT a temperature
    (random logit noise), so inverse scaling should survive calibration only there."""
    rng = np.random.default_rng(seed)
    sizes = {"EleutherAI__pythia-70m": 70e6, "EleutherAI__pythia-160m": 160e6,
             "EleutherAI__pythia-410m": 410e6, "EleutherAI__pythia-1b": 1e9, "EleutherAI__pythia-1.4b": 1.4e9}
    rows, surp = [], {k: [] for k in sizes}
    for corpus, n_texts, n_sub in [("emtec", 50, 10), ("provo", 45, 12)]:
        for t in range(n_texts):
            tid, n = f"{corpus}:{t}", 80
            z = rng.normal(0, 2.0, (n, 40))
            w = np.array([rng.choice(40, p=np.exp(lsm(zz))) for zz in z])
            h = -lsm(z)[np.arange(n), w] / np.log(2)
            for k, size in sizes.items():
                g = np.log10(size / 50e6)
                logits = (1 + 0.8 * g) * z
                dec = ["greedy_search", "sampling"][t % 2] if corpus == "emtec" else "human"
                if corpus == "emtec" and (not by_decoding or dec == "greedy_search"):
                    logits = logits + 1.2 * g * rng.normal(0, 1, z.shape)
                s = {f"surprisal_T{T:g}" if T != 1 else "surprisal": -lsm(logits / T)[np.arange(n), w] / np.log(2)
                     for T in TEMPS}
                surp[k].append(pd.DataFrame(dict(corpus=corpus, text_id=tid, word_idx=range(n), entropy=np.nan, **s)))
            words = [VOCAB[i] for i in rng.integers(0, len(VOCAB), n)]
            for subj in range(n_sub):
                gd = 200 + 8 * h + rng.normal(0, 40, n)
                rows.append(pd.DataFrame(dict(corpus=corpus, language="en", text_id=tid, item_id=tid,
                                              subject_id=f"{corpus}:{subj}", word_idx=range(n), word=words,
                                              FFD=gd * .8, GD=gd, TFT=gd * 1.2, skip=0.0, reg_out=np.nan,
                                              source="x", model="m", genre=["news", "poetry"][t % 3 == 0],
                                              decoding=["greedy_search", "sampling"][t % 2] if corpus == "emtec" else "human")))
    wdf = pd.concat(rows, ignore_index=True)
    texts = wdf.drop_duplicates(["text_id", "word_idx"])[["corpus", "language", "text_id", "word_idx", "word"]]
    return wdf, texts, {k: pd.concat(v, ignore_index=True) for k, v in surp.items()}


def test_joint_scaling():
    w, t, s = simulate_scaling()
    summ, con, per_lm = calibration.joint_scaling(w, t, s, ["GD"], n_boot=200)
    print(summ.round(3).to_string(index=False))
    print(con.round(3).to_string(index=False))
    S = summ.set_index(["corpus", "stat"])["estimate"]
    assert S[("provo", "rel_T1")] < 0 and S[("emtec", "rel_T1")] < 0          # inverse scaling in both
    assert abs(S[("provo", "rel_Tstar")]) < abs(S[("provo", "rel_T1")]) / 2   # calibration fixes human
    c = con.set_index("stat").loc["rel_Tstar"]
    assert c.hi < 0                                                           # but not LLM text


def test_grouped():
    w, t, s = simulate_scaling(by_decoding=True)
    summ, con = calibration.grouped_scaling(w, t, s, ["GD"], n_boot=200, split="decoding")
    print(con.round(3).to_string(index=False))
    c = con.set_index(["contrast", "stat"]).loc[("emtec: deterministic - stochastic", "rel_T1")]
    assert c.hi < 0                       # planted: only greedy texts carry the distortion
    summ2, con2 = calibration.grouped_scaling(w, t, s, ["GD"], n_boot=100, split="typicality",
                                              typ_lm="EleutherAI__pythia-70m")
    assert {"emtec/typ-high", "provo/typ-low"} <= set(summ2.group)
    assert con2.contrast.str.contains("typicality effect").any()
    summ3, con3 = calibration.grouped_scaling(w, t, s, ["GD"], n_boot=100, emtec_genres=["news"])
    assert summ3[summ3.group == "emtec"].n_texts.iat[0] < 50
    print(summ3[summ3.stat == "rel_T1"][["group", "n_texts", "estimate", "lo", "hi"]].round(3))


if __name__ == "__main__":
    test_tstar()
    test_joint_scaling()
    test_grouped()
    print("ok")
