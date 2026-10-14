"""Does the 'bigger LM, worse reading-time fit' effect differ between LLM-written and human text?

For every corpus x LM: per-word cross-validated Delta-LL of surprisal over the
length/frequency baseline, with every text subsampled to the same number of readers
(the SAME reader subsets for all LMs, so LM comparisons are paired).
Then per corpus: regress mean Delta-LL on log10(#params) (+ LM-family intercepts).

Two versions of the scaling slope:
  absolute : d(Delta-LL) / d(log10 params)
  relative : the same after dividing by the corpus's mean Delta-LL across LMs, i.e. the
             proportional loss per 10x params. Scale-free, so differences in reader count,
             lab noise or RT scale between corpora cancel. This is the one to compare.
Uncertainty: cluster (item) bootstrap within corpus, pooled over reader subsamples.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from . import analysis, extras, features

warnings.filterwarnings("ignore", message=".*encountered in matmul")

KNOWN_PARAMS = {"gpt2": 124e6, "gpt2-medium": 355e6, "gpt2-large": 774e6, "gpt2-xl": 1.56e9,
                "distilgpt2": 82e6}


def lm_info(label: str):
    """label like 'gpt2-large' or 'EleutherAI__pythia-410m' -> (family, n_params) or None."""
    name = label.split("__")[-1].lower()
    if name in KNOWN_PARAMS:
        return "gpt2", KNOWN_PARAMS[name]
    m = re.match(r"(pythia|opt|qwen[\d.]*|smollm2?|llama-?\d*|gpt-neo)[-_]?([\d.]+)([mb])", name)
    if m:
        return m.group(1), float(m.group(2)) * (1e6 if m.group(3) == "m" else 1e9)
    return None


def per_word_dll(d, measure, k=10, seed=0):
    cl = analysis.cluster_ids(d)
    folds = analysis._fold_ids(cl, min(k, len(np.unique(cl))), np.random.default_rng(seed))
    kk = folds.max() + 1
    y = d[measure].to_numpy(float)
    full = analysis._cv_loglik(d, y, analysis.BASE + analysis.SURP, folds, kk)
    base = analysis._cv_loglik(d, y, analysis.BASE, folds, kk)
    return full - base, cl


def _fit_slope(x, y, fam):
    fams = sorted(set(fam))
    X = np.column_stack([x] + [(np.asarray(fam) == f).astype(float) for f in fams])
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


def run(words, texts, surps: dict, measures=("FFD", "GD", "TFT"), n_readers=7, n_rep=10,
        n_boot=500, k=10, seed=0, min_readers=3):
    rng = np.random.default_rng(seed)
    labels = [l for l in surps if lm_info(l)]
    info = {l: lm_info(l) for l in labels}
    preds = {l: features.word_predictors(texts, surps[l]) for l in labels}
    keys = ["corpus", "text_id", "word_idx"]
    points, boot_rows = [], []
    for rep in range(n_rep):
        sub = extras._subsample_readers(words, n_readers, rng) if n_readers else words
        rt = features.aggregate_rts(sub)
        rt = rt[rt["n_readers"] >= min(min_readers, n_readers or min_readers)]
        tabs = {l: features.analysis_table(rt, preds[l]).set_index(keys) for l in labels}
        common = None
        for t in tabs.values():
            common = t.index if common is None else common.intersection(t.index)
        for corpus in sorted(set(common.get_level_values(0))):
            idx = common[common.get_level_values(0) == corpus].sort_values()
            for m in measures:
                cols, mask = [], np.ones(len(idx), bool)
                for l in labels:
                    d = tabs[l].loc[idx].reset_index()
                    mask &= d[m].notna().to_numpy()
                    cols.append(d)
                cols = [c[mask].reset_index(drop=True) for c in cols]
                dll = []
                for l, d in zip(labels, cols):
                    v, cl = per_word_dll(d, m, k, seed=rep)
                    dll.append(v)
                dll = np.array(dll)  # LMs x words
                uc, inv = np.unique(cl, return_inverse=True)
                S = np.zeros((len(labels), len(uc)))
                np.add.at(S, (slice(None), inv), dll)
                n = np.bincount(inv, minlength=len(uc)).astype(float)
                x = np.log10([info[l][1] for l in labels])
                fam = [info[l][0] for l in labels]
                mean = S.sum(1) / n.sum()
                for l, v in zip(labels, mean):
                    points.append(dict(rep=rep, corpus=corpus, measure=m, lm=l, family=info[l][0],
                                       params=info[l][1], dLL=v, n_words=int(n.sum())))
                for b in range(n_boot):
                    w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
                    mb = (S @ w) / (n @ w)
                    boot_rows.append(dict(rep=rep, b=b, corpus=corpus, measure=m,
                                          abs_slope=_fit_slope(x, mb, fam),
                                          rel_slope=_fit_slope(x, mb / mb.mean(), fam)))
        print(f"  scaling: rep {rep + 1}/{n_rep}")
    points = pd.DataFrame(points)
    boots = pd.DataFrame(boot_rows)

    # point estimates from the rep-averaged curves
    pm = points.groupby(["corpus", "measure", "lm", "family", "params"], as_index=False)["dLL"].mean()
    summ = []
    for (c, m), g in pm.groupby(["corpus", "measure"]):
        x, y, fam = np.log10(g.params), g.dLL.to_numpy(), list(g.family)
        bb = boots[(boots.corpus == c) & (boots.measure == m)]
        summ.append(dict(corpus=c, measure=m, n_lms=len(g),
                         abs_slope=_fit_slope(x, y, fam),
                         abs_lo=bb.abs_slope.quantile(.025), abs_hi=bb.abs_slope.quantile(.975),
                         rel_slope=_fit_slope(x, y / y.mean(), fam),
                         rel_lo=bb.rel_slope.quantile(.025), rel_hi=bb.rel_slope.quantile(.975)))
    summ = pd.DataFrame(summ)

    # EMTeC minus each human corpus, draw by draw (corpora are independent samples)
    diffs = []
    if "emtec" in set(boots.corpus):
        for c in sorted(set(boots.corpus) - {"emtec"}):
            for m in measures:
                e = boots[(boots.corpus == "emtec") & (boots.measure == m)].reset_index(drop=True)
                h = boots[(boots.corpus == c) & (boots.measure == m)].reset_index(drop=True)
                n_ = min(len(e), len(h))
                for kind in ["abs", "rel"]:
                    dd = e[f"{kind}_slope"][:n_].to_numpy() - h[f"{kind}_slope"][:n_].to_numpy()
                    pt = (summ.query("corpus == 'emtec' and measure == @m")[f"{kind}_slope"].iat[0]
                          - summ.query("corpus == @c and measure == @m")[f"{kind}_slope"].iat[0])
                    diffs.append(dict(contrast=f"emtec-{c}", measure=m, kind=kind, diff=pt,
                                      lo=np.quantile(dd, .025), hi=np.quantile(dd, .975),
                                      p_boot=min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
    return pm, summ, pd.DataFrame(diffs)


def plot(pm, path):
    import matplotlib.pyplot as plt

    ms = list(pm.measure.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.4 * len(ms), 3.6), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for (c, fam), g in pm[pm.measure == m].groupby(["corpus", "family"]):
            g = g.sort_values("params")
            rel = g.dLL / pm[(pm.measure == m) & (pm.corpus == c)].dLL.mean()
            ax.plot(g.params, rel, marker="o" if fam.startswith("pythia") else "s",
                    ls="-" if fam.startswith("pythia") else "--", label=f"{c} / {fam}")
        ax.set_xscale("log")
        ax.axhline(1, color="grey", lw=.5)
        ax.set_xlabel("parameters")
        ax.set_ylabel("Delta-LL / corpus mean (reader-matched)")
        ax.set_title(m)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
