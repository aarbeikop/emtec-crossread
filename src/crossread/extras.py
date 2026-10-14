"""Follow-up analyses for the human vs machine contrast.

1. reader_matched_dll : Delta-LL is inflated by the number of readers averaged per
                        word (less noise in y). Subsample every text to the same
                        number of readers, repeat, and compare corpora on equal footing.
2. divergence         : does the RT ~ surprisal curve differ by source?
                        (a) natural-spline x source, joint Wald test of the interaction
                            (no knot choice, so use this as the primary test);
                        (b) piecewise-linear with a hinge at `knot` bits, giving
                            interpretable low/high slopes per source.
                        Word-level OLS with text-clustered SEs. The trial-level GAM in
                        r/divergence_gam.R is the stronger version for the paper.
3. high_surprisal_words : top-N surprisal words per corpus with context, plus a
                        profile of everything above `knot` bits, to see what the
                        surprising words in machine vs human text actually are.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import analysis, features

warnings.filterwarnings("ignore", message=".*encountered in matmul")

CONTROLS = "length * zipf + prev_length * prev_zipf + prev_surprisal"


# ---------------------------------------------------------------- 1. reader-matched dLL
def _subsample_readers(words: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    pairs = words[["text_id", "subject_id"]].drop_duplicates()
    pairs = pairs.assign(_r=rng.random(len(pairs)))
    pairs["_rank"] = pairs.groupby("text_id")["_r"].rank(method="first")
    keep = pairs[pairs["_rank"] <= n].set_index(["text_id", "subject_id"]).index
    idx = pd.MultiIndex.from_frame(words[["text_id", "subject_id"]])
    return words[idx.isin(keep)]


def _slope(g, measure):
    terms = analysis.BASE + analysis.SURP
    g = g.dropna(subset=[measure, "surprisal", "prev_surprisal"])
    X = analysis.design(g, terms)
    return np.linalg.lstsq(X, g[measure].to_numpy(float), rcond=None)[0][1 + terms.index("surprisal")]


def reader_matched_dll(words, texts, surp, measures, n_readers=7, n_rep=50, k=10, seed=0):
    rng = np.random.default_rng(seed)
    preds = features.word_predictors(texts, surp)
    rows = []
    for rep in range(n_rep):
        rt = features.aggregate_rts(_subsample_readers(words, n_readers, rng))
        rt = rt[rt["n_readers"] >= min(3, n_readers)]
        d = features.analysis_table(rt, preds)
        for m in measures:
            r = analysis.delta_ll(d, m, ["corpus"], k=k, n_perm=0, seed=rep)
            for _, row in r.iterrows():
                g = d[d.corpus == row["corpus"]]
                rows.append(dict(rep=rep, corpus=row["corpus"], measure=m, dLL=row["dLL"],
                                 slope=_slope(g, m), n_words=row["n_words"]))
        if (rep + 1) % 10 == 0:
            print(f"  reader-matched: {rep + 1}/{n_rep}")
    raw = pd.DataFrame(rows)
    summ = (raw.groupby(["corpus", "measure"])
               .agg(dLL_mean=("dLL", "mean"), dLL_lo=("dLL", lambda x: x.quantile(.025)),
                    dLL_hi=("dLL", lambda x: x.quantile(.975)), slope_mean=("slope", "mean"),
                    slope_lo=("slope", lambda x: x.quantile(.025)),
                    slope_hi=("slope", lambda x: x.quantile(.975)), n_words=("n_words", "mean"))
               .reset_index())
    summ["n_readers_per_text"] = n_readers
    return raw, summ


# ---------------------------------------------------------------- 2. divergence tests
def _fit(formula, d):
    import statsmodels.formula.api as smf

    groups = pd.factorize(analysis.cluster_ids(d))[0]
    return smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": groups})


def _wald_terms(res, must_contain):
    names = res.params.index
    sel = [i for i, n in enumerate(names) if all(s in n for s in must_contain)]
    R = np.zeros((len(sel), len(names)))
    for r, i in enumerate(sel):
        R[r, i] = 1
    w = res.wald_test(R, scalar=True)
    return float(w.statistic), float(w.pvalue), len(sel)


def wild_cluster_p(X, y, groups, j, B=999, seed=0):
    """Wild cluster *restricted* bootstrap p-value for H0: beta_j = 0 (Cameron, Gelbach &
    Miller 2008), Webb 6-point weights, which stay valid with few clusters (MECO-en has 12)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    X, y, g = X[order], y[order], groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    G, (n, k) = len(starts), X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    P = XtX_inv @ X.T
    c = G / (G - 1) * (n - 1) / (n - k)

    def tstat(yv):
        b = P @ yv
        u = yv - X @ b
        S = np.add.reduceat(X * u[:, None], starts, axis=0)
        V = c * XtX_inv @ (S.T @ S) @ XtX_inv
        return b[j] / np.sqrt(V[j, j])

    t0 = tstat(y)
    Xr = np.delete(X, j, axis=1)
    fit_r = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    u_r = y - fit_r
    sizes = np.diff(np.r_[starts, n])
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    tb = np.empty(B)
    for i in range(B):
        w = np.repeat(rng.choice(webb, G), sizes)
        tb[i] = tstat(fit_r + u_r * w)
    return float((np.sum(np.abs(tb) >= abs(t0)) + 1) / (B + 1)), G


def divergence(d: pd.DataFrame, measure: str, knot: float = 7.0, df_spline: int = 4,
               ref: str = "human", wild: int = 0):
    d = d.dropna(subset=[measure, "surprisal", "prev_surprisal", "length", "zipf",
                         "prev_length", "prev_zipf"]).copy()
    d["source"] = pd.Categorical(d["source"], categories=[ref] + sorted(set(d.source) - {ref}))
    src = f"C(source, Treatment('{ref}'))"
    out = {}

    # (a) spline x source, shared knots (cr() places them on pooled data)
    res_s = _fit(f"{measure} ~ cr(surprisal, df={df_spline}, constraints='center') * {src} + {CONTROLS}", d)
    stat, p, q = _wald_terms(res_s, ["cr(surprisal", "C(source"])
    lin = _fit(f"{measure} ~ surprisal * {src} + {CONTROLS}", d)
    stat_l, p_l, q_l = _wald_terms(lin, ["surprisal:", "C(source"])
    out["tests"] = pd.DataFrame([
        dict(measure=measure, test="spline x source (curve differs)", stat=stat, df=q, p=p),
        dict(measure=measure, test="linear slope x source", stat=stat_l, df=q_l, p=p_l)])

    # (b) piecewise linear
    d["hinge"] = np.clip(d["surprisal"] - knot, 0, None)
    res_p = _fit(f"{measure} ~ (surprisal + hinge) * {src} + {CONTROLS}", d)
    names = list(res_p.params.index)
    rows = []
    for s in d["source"].cat.categories:
        tag = "" if s == ref else f":{src}[T.{s}]"
        for label, parts in [("slope_below_knot", ["surprisal"]),
                             ("slope_above_knot", ["surprisal", "hinge"])]:
            c = np.zeros(len(names))
            for p_ in parts:
                c[names.index(p_)] = 1
                if tag:
                    c[names.index(f"{p_}{tag}")] = 1
            t = res_p.t_test(c)
            lo, hi = np.asarray(t.conf_int()).ravel()
            rows.append(dict(measure=measure, source=s, quantity=label, knot=knot,
                             estimate=float(np.ravel(t.effect)[0]), ci_lo=lo, ci_hi=hi))
    for s in d["source"].cat.categories[1:]:
        for p_, lab in [("surprisal", "diff_slope_below_knot"), ("hinge", "diff_change_at_knot")]:
            n = f"{p_}:{src}[T.{s}]"
            row = dict(measure=measure, source=f"{s}-{ref}", quantity=lab, knot=knot,
                       estimate=res_p.params[n], ci_lo=res_p.conf_int().loc[n, 0],
                       ci_hi=res_p.conf_int().loc[n, 1], p=res_p.pvalues[n])
            if wild:
                row["p_wild"], row["n_clusters"] = wild_cluster_p(
                    res_p.model.exog, res_p.model.endog, pd.factorize(analysis.cluster_ids(d))[0],
                    names.index(n), B=wild)
            rows.append(row)
    out["piecewise"] = pd.DataFrame(rows)

    # prediction curves from the spline model over the shared surprisal range
    top = min(d.groupby("source", observed=True)["surprisal"].quantile(.99))
    grid = np.linspace(d["surprisal"].quantile(.01), top, 60)
    curves = []
    base = {c: d[c].mean() for c in ["length", "zipf", "prev_length", "prev_zipf", "prev_surprisal"]}
    for s in d["source"].cat.categories:
        g = pd.DataFrame(dict(surprisal=grid, source=pd.Categorical([s] * len(grid),
                                                                    categories=d["source"].cat.categories),
                              **{k: v for k, v in base.items()}))
        pr = res_s.get_prediction(g).summary_frame(alpha=.05)
        curves.append(pd.DataFrame(dict(measure=measure, source=s, surprisal=grid,
                                        fit=pr["mean"], lo=pr["mean_ci_lower"], hi=pr["mean_ci_upper"])))
    out["curves"] = pd.concat(curves, ignore_index=True)
    return out


def plot_curves(curves: pd.DataFrame, path, knot=None):
    import matplotlib.pyplot as plt

    ms = list(curves["measure"].unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 3.8), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for s, g in curves[curves.measure == m].groupby("source"):
            ax.plot(g.surprisal, g.fit, label=s)
            ax.fill_between(g.surprisal, g.lo, g.hi, alpha=.2)
        if knot is not None:
            ax.axvline(knot, color="grey", lw=.6, ls="--")
        ax.set_xlabel("surprisal (bits)")
        ax.set_ylabel(f"predicted {m} (ms), controls at mean")
        ax.set_title(m)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 3. what are the surprising words?
def high_surprisal_words(d: pd.DataFrame, texts: pd.DataFrame, measure="GD", top_n=200,
                         knot=7.0, window=6):
    d = d.copy()
    if "word" not in d.columns:
        d = d.merge(texts[["corpus", "text_id", "word_idx", "word"]],
                    on=["corpus", "text_id", "word_idx"], how="left")
    # residual RT after the length/frequency baseline, within corpus
    d["resid"] = np.nan
    for c, g in d.groupby("corpus"):
        g = g.dropna(subset=[measure])
        X = analysis.design(g, analysis.BASE)
        y = g[measure].to_numpy(float)
        d.loc[g.index, "resid"] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    seqs = {t: dict(zip(g["word_idx"].astype(int), g["word"])) for t, g in texts.groupby("text_id")}

    def ctx(r):
        w, i = seqs[r.text_id], int(r.word_idx)
        left = [w.get(j, "<gap>") for j in range(max(0, i - window), i)]
        right = [w.get(j, "<gap>") for j in range(i + 1, i + 1 + window) if j <= max(w)]
        return " ".join(left + [f"[{w[i]}]"] + right)

    top = (d.sort_values("surprisal", ascending=False).groupby("corpus").head(top_n)
             .sort_values(["corpus", "surprisal"], ascending=[True, False]).copy())
    top["context"] = top.apply(ctx, axis=1)
    top = top[["corpus", "model", "decoding", "genre", "text_id", "word_idx", "word", "surprisal",
               "zipf", "length", measure, "resid", "context"]]

    hi = d[d.surprisal > knot].copy()
    hi["capitalized"] = hi["word"].str[:1].str.isupper()
    hi["has_digit"] = hi["word"].str.contains(r"\d")
    prof = []
    for by in (["corpus"], ["corpus", "decoding"]):
        p = (hi.groupby(by).agg(n_high=("surprisal", "size"), mean_surprisal=("surprisal", "mean"),
                                mean_zipf=("zipf", "mean"), mean_length=("length", "mean"),
                                share_capitalized=("capitalized", "mean"),
                                share_digit=("has_digit", "mean"), mean_resid=("resid", "mean"))
               .reset_index())
        tot = d.groupby(by).size().rename("n_all").reset_index()
        p = p.merge(tot, on=by)
        p["share_of_words"] = p["n_high"] / p["n_all"]
        prof.append(p)
    return top, pd.concat(prof, ignore_index=True)
