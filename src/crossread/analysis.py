"""Three analyses, all run per group (corpus, or EMTeC condition):

1. delta_ll   : predictive power of surprisal over a baseline (held-out
                per-word Delta log-likelihood, text-grouped K-fold CV,
                cluster sign-flip permutation test). Goodkind & Bicknell 2018;
                Wilcox et al. 2023.
2. slopes     : surprisal effect size in ms/bit with text-cluster bootstrap CIs.
                This is the number to compare human vs machine text.
3. binned     : residual RT by surprisal bin on a shared grid, to see (a) where
                each corpus sits on the surprisal axis and (b) whether the
                RT-surprisal link stays linear in that region.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = ["length", "zipf", "length:zipf", "prev_length", "prev_zipf", "prev_length:prev_zipf"]
SURP = ["surprisal", "prev_surprisal"]


def cluster_ids(d: pd.DataFrame) -> np.ndarray:
    """Resampling / CV unit: the item (EMTeC prompt across its 15 versions), else the text."""
    return (d["item_id"] if "item_id" in d.columns else d["text_id"]).to_numpy()


def design(d: pd.DataFrame, terms: list[str]) -> np.ndarray:
    cols = [np.ones(len(d))]
    for t in terms:
        if ":" in t:
            a, b = t.split(":")
            cols.append(d[a].to_numpy(float) * d[b].to_numpy(float))
        else:
            cols.append(d[t].to_numpy(float))
    return np.column_stack(cols)


def _fold_ids(texts: np.ndarray, k: int, rng) -> np.ndarray:
    uniq = rng.permutation(np.unique(texts))
    fold_of = {t: i % k for i, t in enumerate(uniq)}
    return np.array([fold_of[t] for t in texts])


def _cv_loglik(d, y, terms, folds, k):
    ll = np.full(len(d), np.nan)
    X = design(d, terms)
    for f in range(k):
        tr, te = folds != f, folds == f
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        sigma = np.std(y[tr] - X[tr] @ beta, ddof=X.shape[1])
        r = y[te] - X[te] @ beta
        ll[te] = -0.5 * np.log(2 * np.pi * sigma**2) - r**2 / (2 * sigma**2)
    return ll


def delta_ll(d: pd.DataFrame, measure: str, by: list[str], base=BASE, target=SURP,
             k: int = 10, n_perm: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in d.groupby(by):
        g = g.dropna(subset=[measure] + [t for t in base + target if ":" not in t])
        n_txt = g["text_id"].nunique()
        cl = cluster_ids(g)
        kk = min(k, len(np.unique(cl)))
        if kk < 3 or len(g) < 50:
            continue
        y = g[measure].to_numpy(float)
        folds = _fold_ids(cl, kk, rng)
        dll = _cv_loglik(g, y, base + target, folds, kk) - _cv_loglik(g, y, base, folds, kk)
        # cluster (text-level) sign-flip permutation test of mean dLL > 0
        per_text = pd.Series(dll).groupby(cl).sum().to_numpy()
        obs = per_text.sum() / len(dll)
        flips = rng.choice([-1, 1], size=(n_perm, len(per_text)))
        null = (flips * per_text).sum(1) / len(dll)
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                         measure=measure, n_words=len(g), n_texts=n_txt, n_clusters=len(per_text),
                         dLL=obs, dLL_se=dll.std(ddof=1) / np.sqrt(len(dll)),
                         p_perm=(np.sum(null >= obs) + 1) / (n_perm + 1)))
    return pd.DataFrame(rows)


def slopes(d: pd.DataFrame, measure: str, by: list[str], base=BASE, n_boot: int = 1000,
           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    terms = base + SURP
    rows = []
    for key, g in d.groupby(by):
        g = g.dropna(subset=[measure, "surprisal", "prev_surprisal"]).reset_index(drop=True)
        if len(g) < 50:
            continue
        X, y = design(g, terms), g[measure].to_numpy(float)
        b = np.linalg.lstsq(X, y, rcond=None)[0][1 + terms.index("surprisal")]
        idx_by_text = list(pd.Series(np.arange(len(g))).groupby(cluster_ids(g)).indices.values())
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(idx_by_text), len(idx_by_text))
            ix = np.concatenate([idx_by_text[i] for i in pick])
            boots.append(np.linalg.lstsq(X[ix], y[ix], rcond=None)[0][1 + terms.index("surprisal")])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), measure=measure,
                         slope_ms_per_bit=b, ci_lo=lo, ci_hi=hi,
                         mean_surprisal=g["surprisal"].mean(), sd_surprisal=g["surprisal"].std(),
                         n_words=len(g)))
    return pd.DataFrame(rows)


def binned(d: pd.DataFrame, measure: str, by: list[str], n_bins: int = 12) -> pd.DataFrame:
    """Residualize on BASE within group, then mean residual per shared surprisal bin."""
    edges = np.unique(np.quantile(d["surprisal"].dropna(), np.linspace(0, 1, n_bins + 1)))
    out = []
    for key, g in d.groupby(by):
        g = g.dropna(subset=[measure]).copy()
        if len(g) < 50:
            continue
        X = design(g, BASE)
        y = g[measure].to_numpy(float)
        g["resid"] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        g["bin"] = pd.cut(g["surprisal"], edges, include_lowest=True)
        s = g.groupby("bin", observed=True).agg(surprisal_mid=("surprisal", "mean"),
                                                resid_mean=("resid", "mean"),
                                                resid_se=("resid", lambda r: r.std() / np.sqrt(len(r))),
                                                share_of_words=("resid", lambda r: len(r) / len(g)))
        for k_, v in zip(by, key if isinstance(key, tuple) else (key,)):
            s[k_] = v
        s["measure"] = measure
        out.append(s.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)
