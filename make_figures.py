#!/usr/bin/env python
"""Paper figures from the result CSVs (nothing is recomputed).

  python make_figures.py                 # word-level (cross-validated) results, the default
  python make_figures.py --source lmm    # Fig 3 from the trial-level LMM splits instead

Writes out/figures/*.pdf (+ .png previews). Figures whose inputs are missing are skipped.

  fig1_scaling     relative Delta-LL vs. #params per corpus; rows T=1 / T*, columns FFD GD TFT
  fig2_contrasts   EMTeC - human contrasts in relative scaling slope, word-level and LMM, T=1 and T*
  fig3_robustness  (a) EMTeC by decoding strategy, (b) by text-typicality tertile vs. OneStop
  fig4_units       surprisal effect per bit vs. per within-corpus SD
  figA_tstar       best-fitting temperature vs. #params
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--source", choices=["cv", "lmm"], default="cv")
p.add_argument("--lm", default="gpt2", help="LM for fig4 (out/calibration/<lm>/slopes_per_sd.csv)")
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--out", default="out")
p.add_argument("--corpora", nargs="+", default=["emtec", "onestop", "provo", "meco_en"],
               help="corpora shown in the main figures, e.g. --corpora emtec onestop provo (MECO to appendix)")
a = p.parse_args()
OUT = Path(a.out)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
# ACL: Times body text, 3.15in column, 6.3in text width. Figures match: serif, 8pt.
COL, WIDE = 3.15, 6.3
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})
INK, INK2, MUTED, ZERO = "#0b0b0b", "#52514e", "#8a8984", "#c3c2b7"
# Identity colors (validated, all-pairs, light): LLM text = orange; human corpora blue / aqua.
# MECO (12 texts, underpowered) is a de-emphasized neutral with its own marker + dashed line.
CORPUS = {
    "emtec":   dict(label="EMTeC (LLM)", color="#eb6834", marker="o", ls="-"),
    "great":   dict(label="GREAT (LLM)", color="#eb6834", marker="D", ls="--"),
    "onestop": dict(label="OneStop", color="#2a78d6", marker="s", ls="-"),
    "provo":   dict(label="Provo", color="#1baf7a", marker="^", ls="-"),
    "oasst":   dict(label="OASST-ETC (human chat)", color="#2a78d6", marker="P", ls="--"),
    "meco_en": dict(label="MECO-en", color=MUTED, marker="v", ls="--"),
}
ORDER = [c for c in ["emtec", "great", "onestop", "provo", "meco_en", "oasst"] if c in a.corpora]
MEAS = {"FFD": "First fixation", "GD": "Gaze duration", "TFT": "Total time",
        "reg_rate": "Regression rate", "reread": "Rereading"}


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {FIG / name}.pdf")


def read(path):
    path = OUT / path
    if not path.exists():
        print(f"  [skip] missing {path}")
        return None
    return pd.read_csv(path)


def family(lm):
    return "gpt2" if "gpt2" in lm else "pythia"


def tidy_summary(df, kind):
    """-> group, measure, variant (T1 / Tstar), est, lo, hi  (both CSV flavours)."""
    if df is None:
        return None
    if kind == "cv":
        g = "group" if "group" in df else "corpus"
        d = df[df.stat.isin(["rel_T1", "rel_Tstar"])]
        return pd.DataFrame(dict(group=d[g], measure=d.measure, variant=d.stat.str.replace("rel_", ""),
                                 est=d.estimate, lo=d.lo, hi=d.hi))
    return pd.DataFrame(dict(group=df.group, measure=df.measure, variant=df.variant,
                             est=df.rel_slope, lo=df.lo, hi=df.hi))


def tidy_contrasts(df, kind):
    if df is None:
        return None
    if kind == "cv":
        d = df[df.stat.isin(["rel_T1", "rel_Tstar"])]
        return pd.DataFrame(dict(contrast=d.contrast, measure=d.measure,
                                 variant=d.stat.str.replace("rel_", ""), est=d["diff"], lo=d.lo, hi=d.hi))
    return pd.DataFrame(dict(contrast=df.contrast, measure=df.measure, variant=df.variant,
                             est=df["diff"], lo=df.lo, hi=df.hi))


def zero_line(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=ZERO, lw=0.8, zorder=0)


# ---------------------------------------------------------------- fig 1
def fig1():
    per = read("calibration/joint/per_lm.csv")
    if per is None:
        return
    ms = [m for m in a.measures if m in set(per.measure)]
    fig, axes = plt.subplots(2, len(ms), figsize=(WIDE, 3.6), sharex=True, squeeze=False)
    for r, (col, vlabel) in enumerate([("dLL_T1", "$T=1$"), ("dLL_Tstar", "$T=T^*$")]):
        for ci, m in enumerate(ms):
            ax = axes[r][ci]
            ax.grid(axis="y")
            ax.axhline(1, color=ZERO, lw=0.8, zorder=0)
            for c in [c for c in ORDER if c in set(per.corpus)]:
                g = per[(per.corpus == c) & (per.measure == m)].sort_values("params")
                if len(g) < 3:
                    continue
                st = CORPUS[c]
                x = np.log10(g.params.to_numpy())
                y = (g[col] / g[col].mean()).to_numpy()
                fam = np.array([family(l) for l in g.lm])
                fams = sorted(set(fam))
                X = np.column_stack([x] + [(fam == f).astype(float) for f in fams])
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                off = dict(zip(fams, coef[1:]))
                base = np.mean(coef[1:])
                yadj = y - np.array([off[f] for f in fam]) + base   # remove family offsets
                xs = np.linspace(x.min(), x.max(), 10)
                ax.plot(10 ** xs, base + coef[0] * xs, color=st["color"], ls=st["ls"], lw=1.4, zorder=2)
                for f, face in [("pythia", st["color"]), ("gpt2", "white")]:
                    k = fam == f
                    ax.scatter(10 ** x[k], yadj[k], s=16, marker=st["marker"], facecolor=face,
                               edgecolor=st["color"], linewidth=0.9, zorder=3)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(MEAS.get(m, m))
            if r == 1:
                ax.set_xlabel("Parameters")
            if ci == 0:
                ax.set_ylabel(f"{vlabel}\nrel. $\\Delta$LL")
    handles = [mpl.lines.Line2D([], [], color=CORPUS[c]["color"], ls=CORPUS[c]["ls"],
                                marker=CORPUS[c]["marker"], ms=4, label=CORPUS[c]["label"])
               for c in ORDER if c in set(per.corpus)]
    handles += [mpl.lines.Line2D([], [], color=INK2, marker="o", ls="", ms=4, label="Pythia"),
                mpl.lines.Line2D([], [], color=INK2, marker="o", ls="", ms=4, mfc="white", label="GPT-2")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig1_scaling")


# ---------------------------------------------------------------- fig 2
def fig2():
    cv = tidy_contrasts(read("calibration/joint/scaling_contrasts.csv"), "cv")
    lm = tidy_contrasts(read("lmer/scaling/scaling_contrasts.csv"), "lmm")
    parts = [d.assign(method=k) for k, d in [("cv", cv), ("lmm", lm)] if d is not None]
    if not parts:
        return
    d = pd.concat(parts)
    ms = [m for m in a.measures if m in set(d.measure)]
    humans = [c for c in ["onestop", "provo", "meco_en", "oasst"] if c in ORDER and f"emtec - {c}" in set(d.contrast)]
    combos = [(meth, v) for meth in ["cv", "lmm"] if meth in set(d.method) for v in ["T1", "Tstar"]]
    fig, axes = plt.subplots(1, len(ms), figsize=(WIDE, 0.55 + 0.42 * len(humans) * len(combos) / 2),
                             sharey=True, squeeze=False)
    off = np.linspace(0.27, -0.27, len(combos)) if len(combos) > 1 else [0]
    XL = (-1.0, 0.6)
    for ax, m in zip(axes[0], ms):
        zero_line(ax)
        ax.grid(axis="x")
        for i, c in enumerate(humans):
            st = CORPUS[c]
            for (meth, v), o in zip(combos, off):
                r = d[(d.contrast == f"emtec - {c}") & (d.measure == m) & (d.method == meth) & (d.variant == v)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = -i + o
                lo, hi = max(r.lo, XL[0]), min(r.hi, XL[1])
                ax.plot([lo, hi], [y, y], color=st["color"], lw=1.1, solid_capstyle="round")
                for edge, cut, mk in [(XL[0], r.lo < XL[0], "<"), (XL[1], r.hi > XL[1], ">")]:
                    if cut:  # CI runs off the axis: arrowhead at the edge
                        ax.scatter([edge], [y], marker=mk, s=12, color=st["color"], clip_on=False)
                ax.scatter([r.est], [y], s=22, marker="o" if meth == "cv" else "D",
                           facecolor=st["color"] if v == "T1" else "white", edgecolor=st["color"],
                           linewidth=1, zorder=3)
        ax.set_xlim(*XL)
        ax.set_title(MEAS.get(m, m))
        ax.set_xlabel("EMTeC $-$ human, rel. slope")
    axes[0][0].set_yticks([-i for i in range(len(humans))])
    axes[0][0].set_yticklabels([f"vs. {CORPUS[c]['label']}" for c in humans])
    axes[0][0].tick_params(axis="y", length=0)
    lab = {"cv": "word-level CV", "lmm": "trial-level LMM"}
    h = [mpl.lines.Line2D([], [], color=INK2, marker="o" if meth == "cv" else "D", ls="", ms=4.5,
                          mfc=INK2 if v == "T1" else "white", label=f"{lab[meth]}, {'$T=1$' if v == 'T1' else '$T^*$'}")
         for meth, v in combos]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, -0.02, "< 0: stronger inverse scaling on LLM-written text. Bars: 95% item-bootstrap CIs; arrows: CI continues off axis.",
             fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_contrasts")


# ---------------------------------------------------------------- fig 3
DEC = [("beam_search", "beam"), ("greedy_search", "greedy"), ("topk", "top-$k$"), ("topp", "top-$p$"),
       ("sampling", "ancestral sampling")]


def fig3(measure="GD"):
    if a.source == "cv":
        dec = tidy_summary(read("calibration/joint_split-decoding/scaling_summary.csv"), "cv")
        typ_raw = read("calibration/joint_split-typicality/scaling_summary.csv")
        typ = tidy_summary(typ_raw, "cv")
    else:
        dec = tidy_summary(read("lmer/scaling_decoding/scaling_summary.csv"), "lmm")
        typ_raw = None
        typ = tidy_summary(read("lmer/scaling_typicality/scaling_summary.csv"), "lmm")
    if dec is None and typ is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
    # (a) decoding
    ax = axes[0]
    if dec is not None:
        rows = [(f"emtec/{k}", lab, "emtec") for k, lab in DEC] + \
               [(c, CORPUS[c]["label"], c) for c in ["onestop", "provo"] if c in ORDER]
        rows = [r for r in rows if r[0] in set(dec.group)]
        zero_line(ax)
        ax.grid(axis="x")
        for i, (g, lab, c) in enumerate(rows):
            st = CORPUS[c]
            for v, o in [("T1", 0.14), ("Tstar", -0.14)]:
                r = dec[(dec.group == g) & (dec.measure == measure) & (dec.variant == v)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = -i + o
                ax.plot([max(r.lo, -1.5), min(r.hi, 1.0)], [y, y], color=st["color"], lw=1.1)
                ax.scatter([r.est], [y], s=20, marker=st["marker"],
                           facecolor=st["color"] if v == "T1" else "white", edgecolor=st["color"], lw=1, zorder=3)
        ax.axhline(-len(DEC) + 0.5, color=ZERO, lw=0.6, ls=":")
        ax.set_yticks([-i for i in range(len(rows))])
        ax.set_yticklabels([r[1] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"Rel. scaling slope ({MEAS[measure].lower()})")
        ax.set_title("(a) EMTeC by decoding strategy", loc="left")
    # (b) typicality
    ax = axes[1]
    if typ is not None:
        zero_line(ax, "y")
        ax.grid(axis="y")
        tert = ["high", "mid", "low"]
        for j, c in enumerate([c for c in ["emtec", "onestop"] if f"{c}/typ-high" in set(typ.group)]):
            st = CORPUS[c]
            r = [typ[(typ.group == f"{c}/typ-{t}") & (typ.measure == measure) & (typ.variant == "T1")] for t in tert]
            if any(x.empty for x in r):
                continue
            est = [x.est.iat[0] for x in r]
            lo = [x.lo.iat[0] for x in r]
            hi = [x.hi.iat[0] for x in r]
            xs = np.arange(3) + (j - 0.5) * 0.12
            ax.plot(xs, est, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4.5, zorder=3)
            for xx, l, h in zip(xs, lo, hi):
                ax.plot([xx, xx], [max(l, -1.5), min(h, 1.0)], color=st["color"], lw=1.1)
            ax.annotate(st["label"], (xs[-1], est[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
        ax.set_xticks(range(3))
        labels = ["most\npredictable", "middle", "least\npredictable"]
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.3, 2.9)
        ax.set_ylabel("Rel. scaling slope")
        ax.set_title("(b) By text typicality (within-corpus tertiles)", loc="left")
    fig.text(0.01, -0.03, f"{MEAS[measure]}. Filled: $T=1$, hollow: $T^*$ (a only). < 0: bigger LMs fit worse. "
             "Bars: 95% item-bootstrap CIs (clipped to the axis).", fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, f"fig3_robustness{'_lmm' if a.source == 'lmm' else ''}")


# ---------------------------------------------------------------- fig 4
def fig4():
    sd = read(f"calibration/{a.lm}/slopes_per_sd.csv")
    if sd is None:
        return
    sd = sd[(sd.temperature == "T=1") & sd.measure.isin(a.measures)]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=False)
    cs = [c for c in ORDER if c in set(sd.corpus)]
    for ax, (unit, lab) in zip(axes, [("per_bit", "ms per bit"), ("per_sd", "ms per SD of surprisal")]):
        ax.grid(axis="y")
        for i, m in enumerate(a.measures):
            for j, c in enumerate(cs):
                r = sd[(sd.unit == unit) & (sd.measure == m) & (sd.corpus == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (j - (len(cs) - 1) / 2) * 0.17
                st = CORPUS[c]
                ax.plot([x, x], [r.lo, r.hi], color=st["color"], lw=1)
                ax.scatter([x], [r.slope], s=14, marker=st["marker"], color=st["color"], zorder=3)
        ax.set_xticks(range(len(a.measures)))
        ax.set_xticklabels(a.measures)
        ax.set_ylim(bottom=0)
        ax.set_title(lab)
    h = [mpl.lines.Line2D([], [], color=CORPUS[c]["color"], marker=CORPUS[c]["marker"], ls="", ms=4,
                          label=CORPUS[c]["label"]) for c in cs]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.17), columnspacing=0.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig4_units")


# ---------------------------------------------------------------- appendix: T*
def figA():
    per = read("calibration/joint/per_lm.csv")
    if per is None:
        return
    ms = [m for m in a.measures if m in set(per.measure)]
    fig, axes = plt.subplots(1, len(ms), figsize=(WIDE, 1.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        ax.grid(axis="y")
        ax.axhline(1, color=ZERO, lw=0.8, zorder=0)
        for c in [c for c in ORDER if c in set(per.corpus)]:
            st = CORPUS[c]
            g = per[(per.corpus == c) & (per.measure == m)]
            for f, face in [("pythia", st["color"]), ("gpt2", "white")]:
                k = g[[family(l) == f for l in g.lm]].sort_values("params")
                ax.plot(k.params, k.T_star, color=st["color"], ls=st["ls"], lw=1, marker=st["marker"],
                        ms=3.5, mfc=face, mec=st["color"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([0.75, 1, 1.5, 2, 3, 5, 8])
        ax.set_yticklabels(["0.75", "1", "1.5", "2", "3", "5", "8"])
        ax.minorticks_off()
        ax.set_title(MEAS.get(m, m))
        ax.set_xlabel("Parameters")
    axes[0][0].set_ylabel("$T^*$")
    h = [mpl.lines.Line2D([], [], color=CORPUS[c]["color"], ls=CORPUS[c]["ls"], marker=CORPUS[c]["marker"],
                          ms=4, label=CORPUS[c]["label"]) for c in ORDER if c in set(per.corpus)]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=0.8)
    save(fig, "figA_tstar")


# ---------------------------------------------------------------- teaser: one panel, two corpora
def fig_teaser(measure="GD", corpora=("emtec", "onestop")):
    """Column-width teaser: the Pythia suite only (same data, same order, only size varies), so the
    curve is not mixed with GPT-2-vs-Pythia family offsets. Delta-LL relative to the smallest model."""
    per = read("calibration/joint/per_lm.csv")
    if per is None:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.grid(axis="y")
    ax.axhline(1, color=ZERO, lw=0.8, zorder=0)
    for c in corpora:
        st = CORPUS[c]
        g = per[(per.corpus == c) & (per.measure == measure) & per.lm.str.contains("pythia")].sort_values("params")
        if len(g) < 3:
            continue
        for col, ls, face in [("dLL_T1", "-", st["color"]), ("dLL_Tstar", ":", "white")]:
            y = g[col] / g[col].iloc[0]
            ax.plot(g.params, y, color=st["color"], ls=ls, marker=st["marker"], ms=4.5, mfc=face,
                    mec=st["color"], lw=1.4)
        ax.annotate(st["label"], (g.params.iloc[-1], g.dLL_T1.iloc[-1] / g.dLL_T1.iloc[0]),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(right=per.params.max() * 3.2)
    ax.set_xlabel("Pythia parameters")
    ax.set_ylabel(f"$\\Delta$LL relative to 70M\n({MEAS[measure].lower()})")
    h = [mpl.lines.Line2D([], [], color=INK2, ls="-", marker="o", ms=4, label="$T=1$"),
         mpl.lines.Line2D([], [], color=INK2, ls=":", marker="o", ms=4, mfc="white", label="$T^*$")]
    ax.legend(handles=h, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, f"fig_teaser_{measure}")


# ---------------------------------------------------------------- per-corpus slopes (not differences)
def fig_slopes():
    """The scaling slope itself per corpus: below 0 = bigger LMs fit worse. Easier to read than
    the contrasts: EMTeC sits below zero, the human corpora around it."""
    cv = tidy_summary(read("calibration/joint/scaling_summary.csv"), "cv")
    lm = tidy_summary(read("lmer/scaling/scaling_summary.csv"), "lmm")
    panels = [(k, d) for k, d in [("Word-level (cross-validated)", cv), ("Trial-level mixed models", lm)] if d is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(COL * len(panels) * 0.95, 2.1), sharey=True, squeeze=False)
    ms = a.measures
    for ax, (title, d) in zip(axes[0], panels):
        zero_line(ax, "y")
        ax.grid(axis="y")
        cs = [c for c in ORDER if c in set(d.group)]
        for i, m in enumerate(ms):
            for j, c in enumerate(cs):
                st = CORPUS[c]
                r = d[(d.group == c) & (d.measure == m) & (d.variant == "T1")]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (j - (len(cs) - 1) / 2) * 0.18
                lo, hi = max(r.lo, -1.2), min(r.hi, 0.6)
                ax.plot([x, x], [lo, hi], color=st["color"], lw=1.1)
                ax.scatter([x], [r.est], s=20, marker=st["marker"], color=st["color"], zorder=3)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([MEAS[m] for m in ms])
        ax.set_ylim(-1.2, 0.6)
        ax.set_title(title)
    axes[0][0].set_ylabel("Rel. $\\Delta$LL change per 10$\\times$ params")
    h = [mpl.lines.Line2D([], [], color=CORPUS[c]["color"], marker=CORPUS[c]["marker"], ls="", ms=4,
                          label=CORPUS[c]["label"]) for c in ORDER]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.text(0.01, -0.03, "$T=1$. < 0: bigger LMs fit reading times worse. 95% item-bootstrap CIs (clipped).",
             fontsize=6.5, color=INK2)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_slopes")


for f in [fig1, fig2, fig3, fig4, figA, fig_teaser, fig_slopes]:
    f()
print(f"done -> {FIG}/")
