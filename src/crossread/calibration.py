"""Are language models over-confident readers of machine text?

Hypothesis: LLM-written text is in-distribution for LMs, so LM surprisal on it is *compressed*
(too many near-zero values) relative to what human readers experience. One mechanism would
explain the earlier findings:
  (a) steeper ms-per-bit slopes on LLM text   -> same cost spread over a smaller bit range
  (b) inverse scaling                           -> bigger LMs compress more
  (c) no self advantage / family disadvantage   -> a model compresses its own text most
Tests:
  T*      : temperature that maximises reading-time fit, surprisal under softmax(logits / T).
            Prediction: T* > 1, larger on EMTeC than on human corpora, and within EMTeC larger
            for deterministic decoding (greedy, beam) than for sampling.
  per-SD  : surprisal slopes in ms per within-corpus SD. If the ms/bit gap between corpora
            is compression, it should shrink per SD and after calibrating at T*.
  scaling : inverse-scaling slope across LMs at T = 1 vs at each LM's own T*.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from . import analysis, features, scaling

warnings.filterwarnings("ignore", message=".*encountered in matmul")

# corpora of LLM-written text; every contrast is <LLM corpus> - <human corpus>
LLM_CORPORA = ("emtec", "great")


def temp_columns(surp: pd.DataFrame) -> dict:
    cols = {1.0: "surprisal"}
    for c in surp.columns:
        m = re.fullmatch(r"surprisal_T([\d.]+)", c)
        if m:
            cols[float(m.group(1))] = c
    return dict(sorted(cols.items()))


def _tables(rt, texts, surp, tcols):
    keys = ["corpus", "text_id", "word_idx"]
    tabs = {}
    for t, c in tcols.items():
        s = surp[["corpus", "text_id", "word_idx", c]].rename(columns={c: "surprisal"})
        s["entropy"] = np.nan
        tabs[t] = features.analysis_table(rt, features.word_predictors(texts, s)).set_index(keys)
    common = None
    for tb in tabs.values():
        common = tb.index if common is None else common.intersection(tb.index)
    return tabs, common.sort_values()


def _cluster_sums(tabs, idx, measure, k=10):
    """S[T, cluster] = summed per-word CV dLL; n[cluster] = word counts (common rows across T)."""
    ts = list(tabs)
    frames = [tabs[t].loc[idx].reset_index() for t in ts]
    mask = np.ones(len(idx), bool)
    for f in frames:
        mask &= f[measure].notna().to_numpy()
    if mask.sum() < 200 or np.nanstd(frames[0].loc[mask, measure]) == 0:
        return None
    frames = [f[mask].reset_index(drop=True) for f in frames]
    rows = []
    for f in frames:
        v, cl = scaling.per_word_dll(f, measure, k)
        rows.append(v)
    uc, inv = np.unique(cl, return_inverse=True)
    S = np.zeros((len(ts), len(uc)))
    np.add.at(S, (slice(None), inv), np.array(rows))
    return S, np.bincount(inv, minlength=len(uc)).astype(float), uc


def tstar(words, texts, surp, measures=("FFD", "GD", "TFT"), n_boot=1000, seed=0,
          min_readers=3, k=10):
    """T* per corpus and per EMTeC decoding strategy, with item-bootstrap CIs.
    EMTeC decoding subsets are resampled jointly (same items) so their contrasts are paired."""
    rng = np.random.default_rng(seed)
    tcols = temp_columns(surp)
    ts = np.array(list(tcols))
    if len(ts) < 3:
        raise ValueError("need >= 3 temperatures: rerun `run.py surprisal --temps ...`")
    rt = features.aggregate_rts(words)
    rt = rt[rt["n_readers"] >= min_readers]
    tabs, common = _tables(rt, texts, surp, tcols)
    corp = common.get_level_values(0)
    dec = tabs[1.0].loc[common, "decoding"].to_numpy()

    subsets = [(c, "all", common[corp == c]) for c in sorted(set(corp))]
    if "emtec" in set(corp):
        subsets += [("emtec", d, common[(corp == "emtec") & (dec == d)])
                    for d in sorted(set(dec[corp == "emtec"]))]

    curves, est, draws = [], [], {}
    for m in measures:
        sums = {}
        for c, d, idx in subsets:
            if len(idx) < 200:
                continue
            r = _cluster_sums(tabs, idx, m, k)
            if r is None:
                continue
            S, n, uc = r
            sums[(c, d)] = (S, n, uc)
            mean = S.sum(1) / n.sum()
            for t, v in zip(ts, mean):
                curves.append(dict(corpus=c, subset=d, measure=m, T=t, dLL=v))
        # bootstrap: one item draw per corpus, shared by that corpus's subsets AND by all measures
        # (same generator state per corpus for every measure), so stage contrasts are paired
        for c in sorted({c for c, _ in sums}):
            keys_c = [kk for kk in sums if kk[0] == c]
            items = np.unique(analysis.cluster_ids(tabs[1.0].loc[common[corp == c]].reset_index()))
            rng = np.random.default_rng([seed, sorted(set(corp)).index(c)])
            for b in range(n_boot):
                cnt = np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items))
                wmap = dict(zip(items, cnt))
                for kk in keys_c:
                    S, n, uc = sums[kk]
                    w = np.array([wmap[u] for u in uc], float)
                    tot = n @ w
                    if tot == 0:
                        continue
                    mb = (S @ w) / tot
                    draws.setdefault((m,) + kk, []).append((ts[np.argmax(mb)], mb.max() - mb[ts == 1.0][0]))
        for (c, d), (S, n, uc) in sums.items():
            mean = S.sum(1) / n.sum()
            dd = np.array(draws[(m, c, d)])
            j = np.argmax(mean)
            est.append(dict(corpus=c, subset=d, measure=m, T_star=ts[j],
                            T_lo=np.quantile(dd[:, 0], .025), T_hi=np.quantile(dd[:, 0], .975),
                            p_T_gt_1=(dd[:, 0] > 1).mean(),
                            dLL_T1=mean[ts == 1.0][0], dLL_Tstar=mean[j],
                            gain_pct=100 * (mean[j] / mean[ts == 1.0][0] - 1),
                            n_words=int(n.sum()), n_items=len(uc)))
    est = pd.DataFrame(est)

    # contrasts on log T* (draw by draw)
    con = []
    stage = {"early": ["skip_rate", "FFD"], "late": ["reg_rate", "reread"]}
    for c in sorted({c for (mm, c, d) in draws}):
        for d in sorted({d for (mm, cc, d) in draws if cc == c}):
            g = {k_: [np.log(np.array(draws[(m, c, d)])[:, 0]) for m in v if (m, c, d) in draws]
                 for k_, v in stage.items()}
            if all(g.values()):
                n_ = min(map(len, g["early"] + g["late"]))
                diff = np.mean([x[:n_] for x in g["late"]], 0) - np.mean([x[:n_] for x in g["early"]], 0)
                draws[("late-early", c, d)] = diff
                con.append(_contrast("late-early", f"{c}/{d}: late (reg, reread) - early (skip, FFD)", diff))
    if ("late-early", "emtec", "all") in draws:
        for c in sorted({c for (mm, c, d) in draws if mm == "late-early"} - {"emtec"}):
            e, h = draws[("late-early", "emtec", "all")], draws.get(("late-early", c, "all"))
            if h is not None:
                n_ = min(len(e), len(h))
                con.append(_contrast("late-early", f"stage gap: emtec - {c}", e[:n_] - h[:n_]))
    for m in measures:
        get = lambda c, d: np.log(np.array(draws[(m, c, d)])[:, 0]) if (m, c, d) in draws else None
        e = get("emtec", "all")
        for c in sorted({c for (mm, c, d) in draws if mm == m} - set(LLM_CORPORA)):
            h = get(c, "all")
            if e is not None and h is not None:
                n_ = min(len(e), len(h))
                con.append(_contrast(m, f"emtec - {c}", e[:n_] - h[:n_]))
        det = [get("emtec", d) for d in ("greedy_search", "beam_search") if get("emtec", d) is not None]
        sto = [get("emtec", d) for d in ("sampling", "topk", "topp") if get("emtec", d) is not None]
        if det and sto:
            n_ = min(map(len, det + sto))
            diff = np.mean([x[:n_] for x in det], 0) - np.mean([x[:n_] for x in sto], 0)
            con.append(_contrast(m, "emtec: deterministic - stochastic decoding", diff))
    return est, pd.DataFrame(con), pd.DataFrame(curves), tabs, common


def _contrast(m, label, d):
    return dict(measure=m, contrast=label, log_ratio=np.median(d), lo=np.quantile(d, .025),
                hi=np.quantile(d, .975), p_boot=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def per_sd_slopes(tabs, common, est, measures=("FFD", "GD", "TFT"), n_boot=500):
    """ms per bit and ms per within-corpus SD, at T = 1 and at each corpus's own T*."""
    rows = []
    for m in measures:
        for c in sorted(set(common.get_level_values(0))):
            idx = common[common.get_level_values(0) == c]
            e = est[(est.corpus == c) & (est.subset == "all") & (est.measure == m)]
            if e.empty:
                continue
            for label, t in [("T=1", 1.0), ("T*", float(e.T_star.iat[0]))]:
                d = tabs[t].loc[idx].reset_index()
                sd = d["surprisal"].std()
                for unit in ["per_bit", "per_sd"]:
                    dd = d.copy()
                    if unit == "per_sd":
                        dd["surprisal"] = d["surprisal"] / sd
                        dd["prev_surprisal"] = d["prev_surprisal"] / sd
                    s = analysis.slopes(dd, m, ["corpus"], n_boot=n_boot).iloc[0]
                    rows.append(dict(corpus=c, measure=m, temperature=label, T=t, unit=unit,
                                     slope=s.slope_ms_per_bit, lo=s.ci_lo, hi=s.ci_hi, sd_bits=sd))
    return pd.DataFrame(rows)


def plot_curves(curves, path):
    import matplotlib.pyplot as plt

    ms = list(curves.measure.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.4 * len(ms), 3.6), squeeze=False)
    for ax, m in zip(axes[0], ms):
        for (c, d), g in curves[curves.measure == m].groupby(["corpus", "subset"]):
            ref = g.loc[g["T"] == 1.0, "dLL"].iat[0]
            ax.plot(g["T"], g.dLL / ref, marker="o", lw=2 if d == "all" else .8,
                    ls="-" if d == "all" else ":", label=c if d == "all" else f"{c}/{d}")
        ax.axvline(1, color="grey", lw=.5)
        ax.axhline(1, color="grey", lw=.5)
        ax.set_xlabel("temperature T")
        ax.set_ylabel("dLL(T) / dLL(1)")
        ax.set_title(m)
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- hotspots
def hotspots(tab1: pd.DataFrame, texts: pd.DataFrame, thresh: float = 1.0, n_boot: int = 1000,
             top_n: int = 100, window: int = 6, seed: int = 0):
    """Words the LM is near-certain about (surprisal < thresh bits at T = 1) that readers still
    regress from or reread. Residuals are from a within-corpus model with length, frequency AND
    the LM's surprisal (+ spillover), so resid > 0 means readers struggle more than the
    LM-based model expects for that word."""
    rng = np.random.default_rng(seed)
    d = tab1.reset_index().copy()
    for m in ["reg_rate", "reread"]:
        d[f"resid_{m}"] = np.nan
        for c, g in d.groupby("corpus"):
            terms = analysis.BASE + analysis.SURP
            g = g.dropna(subset=[m] + [t for t in terms if ":" not in t])
            if len(g) < 100:
                continue
            X = analysis.design(g, terms)
            y = g[m].to_numpy(float)
            d.loc[g.index, f"resid_{m}"] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    d["low"] = d["surprisal"] < thresh
    d["cluster"] = analysis.cluster_ids(d)

    rows = []
    groups = [(c, "all", g) for c, g in d.groupby("corpus")]
    groups += [("emtec", dec, g) for dec, g in d[d.corpus == "emtec"].groupby("decoding")]
    for c, dec, g in groups:
        lo = g[g.low]
        row = dict(corpus=c, subset=dec, n_words=len(g), n_low=len(lo), share_low=len(lo) / len(g))
        cl = lo["cluster"].to_numpy()
        uc, inv = np.unique(cl, return_inverse=True)
        for m in ["reg_rate", "reread", "resid_reg_rate", "resid_reread"]:
            v = lo[m].to_numpy(float)
            ok = ~np.isnan(v)
            row[f"{m}_low"] = np.nanmean(v) if ok.any() else np.nan
            if m.startswith("resid") and ok.sum() > 20:
                sums = np.bincount(inv[ok], v[ok], len(uc))
                cnts = np.bincount(inv[ok], minlength=len(uc)).astype(float)
                bs = []
                for _ in range(n_boot):
                    w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc))
                    bs.append((sums @ w) / max(cnts @ w, 1))
                row[f"{m}_low_lo"], row[f"{m}_low_hi"] = np.quantile(bs, [.025, .975])
        rows.append(row)
    summary = pd.DataFrame(rows)

    seqs = {t: dict(zip(g["word_idx"].astype(int), g["word"])) for t, g in texts.groupby("text_id")}

    def ctx(r):
        w, i = seqs[r.text_id], int(r.word_idx)
        left = [w.get(j, "<gap>") for j in range(max(0, i - window), i)]
        right = [w.get(j, "<gap>") for j in range(i + 1, i + 1 + window) if j <= max(w)]
        return " ".join(left + [f"[{w[i]}]"] + right)

    ex = (d[d.low].dropna(subset=["resid_reread"]).sort_values("resid_reread", ascending=False)
            .groupby("corpus").head(top_n).copy())
    ex["context"] = ex.apply(ctx, axis=1)
    ex = ex[["corpus", "decoding", "genre", "text_id", "word_idx", "surprisal", "reg_rate", "reread",
             "resid_reg_rate", "resid_reread", "context"]]
    return summary, ex


# --------------------------------------------------------------------------- joint bootstrap
def joint_scaling(words, texts, surps: dict, measures=("FFD", "GD", "TFT", "reg_rate", "reread"),
                  n_boot=1000, seed=0, min_readers=3, k=10):
    """Inverse scaling before vs after temperature calibration, with uncertainty.

    Per corpus, ONE item bootstrap draw is shared by all LMs, temperatures and measures; in each
    draw T* is re-chosen per LM, then across LMs (log10 params, family intercepts) we fit
      rel_T1     : relative dLL change per 10x params at T = 1 (inverse scaling, < 0)
      rel_Tstar  : the same at each LM's own T* (inverse scaling left after calibration)
      calib      : rel_Tstar - rel_T1 (how much calibration removes, > 0)
      logT_slope : growth of log T* per 10x params (bigger models need more flattening)
    Corpora are independent samples, so EMTeC - human contrasts are formed draw by draw."""
    rt = features.aggregate_rts(words)
    rt = rt[rt["n_readers"] >= min_readers]
    labels = [l for l in sorted(surps) if scaling.lm_info(l) and len(temp_columns(surps[l])) >= 3]
    if len(labels) < 3:
        raise ValueError("need >= 3 LMs with known size and >= 3 temperatures")
    store, per_lm = {}, []
    for l in labels:
        tcols = temp_columns(surps[l])
        ts = np.array(list(tcols))
        tabs, common = _tables(rt, texts, surps[l], tcols)
        corp = common.get_level_values(0)
        for c in sorted(set(corp)):
            for m in measures:
                if m not in tabs[1.0].columns:
                    continue
                r = _cluster_sums(tabs, common[corp == c], m, k)
                if r is not None:
                    store[(c, m, l)] = (*r, ts)
        print(f"  joint: scored {l}")
    corpora = sorted({c for c, _, _ in store})
    draws, point = {}, {}
    for ci, c in enumerate(corpora):
        items = np.unique(np.concatenate([v[2] for (cc, _, _), v in store.items() if cc == c]))
        pos = {kk: np.searchsorted(items, v[2]) for kk, v in store.items() if kk[0] == c}
        rng = np.random.default_rng([seed, ci])
        for b in range(n_boot + 1):
            wi = np.ones(len(items)) if b == 0 else \
                np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)).astype(float)
            for m in measures:
                lms = [l for l in labels if (c, m, l) in store]
                if len(lms) < 3:
                    continue
                d1, ds, tst = [], [], []
                for l in lms:
                    S, n, uc, ts = store[(c, m, l)]
                    w = wi[pos[(c, m, l)]]
                    mean = (S @ w) / max(n @ w, 1)
                    j = int(np.argmax(mean))
                    d1.append(mean[ts == 1.0][0])
                    ds.append(mean[j])
                    tst.append(ts[j])
                    if b == 0:
                        per_lm.append(dict(corpus=c, measure=m, lm=l, params=scaling.lm_info(l)[1],
                                           T_star=ts[j], dLL_T1=d1[-1], dLL_Tstar=ds[-1]))
                d1, ds, tst = map(np.array, (d1, ds, tst))
                x = np.log10([scaling.lm_info(l)[1] for l in lms])
                fam = [scaling.lm_info(l)[0] for l in lms]
                r1 = scaling._fit_slope(x, d1 / d1.mean(), fam)
                rs = scaling._fit_slope(x, ds / ds.mean(), fam)
                rec = dict(rel_T1=r1, rel_Tstar=rs, calib=rs - r1,
                           logT_slope=scaling._fit_slope(x, np.log(tst), fam))
                if b == 0:
                    point[(c, m)] = rec
                else:
                    draws.setdefault((c, m), []).append(rec)
    stats = ["rel_T1", "rel_Tstar", "calib", "logT_slope"]
    rows = []
    for (c, m), rec in point.items():
        D = pd.DataFrame(draws[(c, m)])
        for s in stats:
            v = D[s].to_numpy()
            rows.append(dict(corpus=c, measure=m, stat=s, estimate=rec[s], lo=np.quantile(v, .025),
                             hi=np.quantile(v, .975),
                             p_boot=min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    summ = pd.DataFrame(rows)
    con = []
    for m in measures:
        for L in [x for x in LLM_CORPORA if (x, m) in point]:
            E = pd.DataFrame(draws[(L, m)])
            for c in corpora:
                if c in LLM_CORPORA or (c, m) not in point:
                    continue
                H = pd.DataFrame(draws[(c, m)])
                n_ = min(len(E), len(H))
                for s in stats:
                    v = E[s].to_numpy()[:n_] - H[s].to_numpy()[:n_]
                    con.append(dict(measure=m, contrast=f"{L} - {c}", stat=s,
                                    diff=point[(L, m)][s] - point[(c, m)][s],
                                    lo=np.quantile(v, .025), hi=np.quantile(v, .975),
                                    p_boot=min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    return summ, pd.DataFrame(con), pd.DataFrame(per_lm)


# --------------------------------------------------------------------------- grouped version
DETERMINISTIC = ("greedy_search", "beam_search")
STOCHASTIC = ("sampling", "topk", "topp")


def grouped_scaling(words, texts, surps: dict, measures=("FFD", "GD", "TFT"), n_boot=1000, seed=0,
                    min_readers=3, k=10, emtec_genres=None, split=None, typ_lm="gpt2", word_keys=None):
    """joint_scaling on sub-groups of texts.

    split=None         : one group per corpus (same as joint_scaling, optionally genre-filtered)
    split="decoding"   : EMTeC split by decoding strategy; human corpora stay whole
    split="generator"  : EMTeC split by the LLM that wrote the text; human corpora stay whole
    word_keys          : optional MultiIndex (corpus, text_id, word_idx) of words to keep
                         (e.g. single-token words only); context features still use the full text
    split="typicality" : every corpus split into tertiles of text-level LM-typicality
                         (mean surprisal under `typ_lm`; 'high' = most predictable third).
                         Human corpora get the same split as a control: if inverse scaling also
                         grows with typicality in human text, it is about predictability, not
                         about who wrote the text.
    Bootstrap: one item draw per corpus, shared by all of that corpus's groups, so
    within-corpus contrasts (high vs low typicality, deterministic vs stochastic) are paired;
    contrasts across corpora are formed draw by draw from independent draws."""
    rt = features.aggregate_rts(words)
    rt = rt[rt["n_readers"] >= min_readers]
    if emtec_genres:
        rt = rt[(rt["corpus"] != "emtec") | rt["genre"].isin(emtec_genres)]
    if word_keys is not None:
        rt = rt[pd.MultiIndex.from_frame(rt[["corpus", "text_id", "word_idx"]]).isin(word_keys)]
    tx = rt[["corpus", "text_id", "decoding", "model"]].drop_duplicates("text_id").copy()
    if split == "generator":
        tx["group"] = np.where(tx.corpus == "emtec", "emtec/" + tx.model.astype(str), tx.corpus)
    elif split == "decoding":
        tx["group"] = np.where(tx.corpus == "emtec", "emtec/" + tx.decoding.astype(str), tx.corpus)
    elif split == "typicality":
        if typ_lm not in surps:
            raise ValueError(f"typicality LM {typ_lm} not in out/surprisal/")
        ms = surps[typ_lm].groupby("text_id")["surprisal"].mean()
        tx["typ"] = tx["text_id"].map(ms)
        tx["tert"] = tx.groupby("corpus")["typ"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["high", "mid", "low"]).astype(str))
        tx["group"] = tx["corpus"] + "/typ-" + tx["tert"]
    else:
        tx["group"] = tx["corpus"]
    gmap = dict(zip(tx.text_id, tx.group))
    group_info = (tx.groupby("group").agg(n_texts=("text_id", "size"),
                                          **({"mean_surprisal": ("typ", "mean")} if "typ" in tx else {}))
                    .reset_index())

    labels = [l for l in sorted(surps) if scaling.lm_info(l) and len(temp_columns(surps[l])) >= 3]
    store = {}
    for l in labels:
        tcols = temp_columns(surps[l])
        ts = np.array(list(tcols))
        tabs, common = _tables(rt, texts, surps[l], tcols)
        grp = np.array([gmap.get(t, "") for t in common.get_level_values(1)])
        for g in sorted(set(grp) - {""}):
            for m in measures:
                if m not in tabs[1.0].columns:
                    continue
                r = _cluster_sums(tabs, common[grp == g], m, k)
                if r is not None:
                    store[(g, m, l)] = (*r, ts)
        print(f"  grouped: scored {l}")

    corp = lambda g: g.split("/")[0]
    groups = sorted({g for g, _, _ in store})
    corpora = sorted({corp(g) for g in groups})
    draws, point = {}, {}
    for ci, c in enumerate(corpora):
        gs = [g for g in groups if corp(g) == c]
        items = np.unique(np.concatenate([v[2] for (g, _, _), v in store.items() if corp(g) == c]))
        pos = {kk: np.searchsorted(items, v[2]) for kk, v in store.items() if corp(kk[0]) == c}
        rng = np.random.default_rng([seed, ci])
        for b in range(n_boot + 1):
            wi = np.ones(len(items)) if b == 0 else \
                np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)).astype(float)
            for g in gs:
                for m in measures:
                    lms = [l for l in labels if (g, m, l) in store]
                    if len(lms) < 3:
                        continue
                    d1, ds, tst = [], [], []
                    for l in lms:
                        S, n, uc, ts = store[(g, m, l)]
                        w = wi[pos[(g, m, l)]]
                        mean = (S @ w) / max(n @ w, 1)
                        j = int(np.argmax(mean))
                        d1.append(mean[ts == 1.0][0]); ds.append(mean[j]); tst.append(ts[j])
                    d1, ds, tst = map(np.array, (d1, ds, tst))
                    x = np.log10([scaling.lm_info(l)[1] for l in lms])
                    fam = [scaling.lm_info(l)[0] for l in lms]
                    r1 = scaling._fit_slope(x, d1 / d1.mean(), fam)
                    rs = scaling._fit_slope(x, ds / ds.mean(), fam)
                    rec = dict(rel_T1=r1, rel_Tstar=rs, calib=rs - r1,
                               logT_slope=scaling._fit_slope(x, np.log(tst), fam))
                    if b == 0:
                        point[(g, m)] = rec
                    else:
                        draws.setdefault((g, m), []).append(rec)

    stats = ["rel_T1", "rel_Tstar", "calib", "logT_slope"]

    def ci(v):
        return np.quantile(v, .025), np.quantile(v, .975), min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))

    rows = []
    for (g, m), rec in point.items():
        D = pd.DataFrame(draws[(g, m)])
        for s in stats:
            lo, hi, p = ci(D[s].to_numpy())
            rows.append(dict(group=g, measure=m, stat=s, estimate=rec[s], lo=lo, hi=hi, p_boot=p))
    summ = pd.DataFrame(rows).merge(group_info, on="group", how="left")

    def contrast(label, m, a_groups, b_groups, paired):
        A = [pd.DataFrame(draws[(g, m)]) for g in a_groups if (g, m) in draws]
        B = [pd.DataFrame(draws[(g, m)]) for g in b_groups if (g, m) in draws]
        if not A or not B:
            return []
        n_ = min(map(len, A + B))
        out = []
        for s in ["rel_T1", "rel_Tstar"]:
            va = np.mean([x[s].to_numpy()[:n_] for x in A], 0)
            vb = np.mean([x[s].to_numpy()[:n_] for x in B], 0)
            pa = np.mean([point[(g, m)][s] for g in a_groups if (g, m) in point])
            pb = np.mean([point[(g, m)][s] for g in b_groups if (g, m) in point])
            lo, hi, p = ci(va - vb)
            out.append(dict(measure=m, contrast=label, paired=paired, stat=s, diff=pa - pb, lo=lo, hi=hi, p_boot=p))
        return out

    con = []
    for m in measures:
        if split == "generator":
            gens = sorted(g for g in groups if g.startswith("emtec/"))
            for i, g1 in enumerate(gens):
                for g2 in gens[i + 1:]:
                    con += contrast(f"{g1} - {g2}", m, [g1], [g2], True)
                for c in corpora:
                    if c not in LLM_CORPORA:
                        con += contrast(f"{g1} - {c}", m, [g1], [c], False)
        elif split == "decoding":
            con += contrast("emtec: deterministic - stochastic", m,
                            [f"emtec/{d}" for d in DETERMINISTIC], [f"emtec/{d}" for d in STOCHASTIC], True)
            for c in corpora:
                if c not in LLM_CORPORA:
                    con += contrast(f"emtec deterministic - {c}", m, [f"emtec/{d}" for d in DETERMINISTIC], [c], False)
                    con += contrast(f"emtec stochastic - {c}", m, [f"emtec/{d}" for d in STOCHASTIC], [c], False)
        elif split == "typicality":
            for c in corpora:
                con += contrast(f"{c}: high - low typicality", m, [f"{c}/typ-high"], [f"{c}/typ-low"], True)
            for c in corpora:
                if c not in LLM_CORPORA:
                    for t in ["high", "mid", "low"]:
                        con += contrast(f"emtec - {c} (typ-{t})", m, [f"emtec/typ-{t}"], [f"{c}/typ-{t}"], False)
                    # difference-in-differences: does typicality matter MORE in LLM text?
                    A = [pd.DataFrame(draws.get((f"emtec/typ-{t}", m), [])) for t in ("high", "low")]
                    B = [pd.DataFrame(draws.get((f"{c}/typ-{t}", m), [])) for t in ("high", "low")]
                    if all(len(x) for x in A + B):
                        n_ = min(map(len, A + B))
                        for s in ["rel_T1", "rel_Tstar"]:
                            v = (A[0][s][:n_].to_numpy() - A[1][s][:n_].to_numpy()) - \
                                (B[0][s][:n_].to_numpy() - B[1][s][:n_].to_numpy())
                            pt = (point[(f"emtec/typ-high", m)][s] - point[(f"emtec/typ-low", m)][s]) - \
                                 (point[(f"{c}/typ-high", m)][s] - point[(f"{c}/typ-low", m)][s])
                            lo, hi, p = ci(v)
                            con.append(dict(measure=m, contrast=f"typicality effect: emtec - {c}", paired=False,
                                            stat=s, diff=pt, lo=lo, hi=hi, p_boot=p))
        else:
            for L in [x for x in LLM_CORPORA if x in corpora]:
                for c in corpora:
                    if c not in LLM_CORPORA:
                        con += contrast(f"{L} - {c}", m, [L], [c], False)
    return summ, pd.DataFrame(con)
