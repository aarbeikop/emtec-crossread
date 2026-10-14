#!/usr/bin/env python
"""LaTeX tables from the result CSVs (nothing is recomputed). Rerun after any new LM or split.

  python make_tables.py           # -> out/tables/*.tex, then \\input{tables/<name>} in the paper

  tab_slopes        relative scaling slope per corpus x measure, T=1 and T*, word-level and LMM
  tab_contrasts     EMTeC - human corpus, same layout, with 95% CIs
  tab_robustness    gaze duration: EMTeC by decoding strategy and by typicality tertile (+ OneStop)
  tab_tstar         best-fitting temperature per LM x corpus (gaze duration), appendix

Bold = 95% item-bootstrap CI excludes 0. Uses only \\hline (no extra packages) and graphicx's
\\resizebox (already loaded by the ACL template).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
p.add_argument("--measures", nargs="+", default=["FFD", "GD", "TFT"])
p.add_argument("--corpora", nargs="+", default=["emtec", "great", "onestop", "provo", "meco_en", "oasst"])
a = p.parse_args()
OUT = Path(a.out)
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

NAME = {"emtec": "EMTeC", "great": "GREAT", "onestop": "OneStop", "provo": "Provo", "meco_en": "MECO-en", "oasst": "OASST-ETC"}
MEAS = {"FFD": "First fixation", "GD": "Gaze duration", "TFT": "Total time"}
DEC = {"beam_search": "beam", "greedy_search": "greedy", "topk": "top-$k$", "topp": "top-$p$",
       "sampling": "ancestral sampling"}


def read(path):
    f = OUT / path
    if not f.exists():
        print(f"  [skip] missing {f}")
        return None
    return pd.read_csv(f)


def num(x):
    return "--" if pd.isna(x) else f"{x:.2f}".replace("-", "$-$")


def cell(r, ci=False):
    if r is None:
        return "--"
    s = num(r["est"])
    if r["lo"] > 0 or r["hi"] < 0:
        s = f"\\textbf{{{s}}}"
    if ci:
        s += f" {{\\scriptsize [{num(r['lo'])}, {num(r['hi'])}]}}"
    return s


def tidy(df, kind, key):
    """-> key, measure, variant (T1/Tstar), est, lo, hi"""
    if df is None:
        return None
    if kind == "cv":
        d = df[df.stat.isin(["rel_T1", "rel_Tstar"])]
        est = d["estimate"] if "estimate" in d else d["diff"]
        return pd.DataFrame({"key": d[key], "measure": d.measure, "variant": d.stat.str.replace("rel_", ""),
                             "est": est, "lo": d.lo, "hi": d.hi})
    est = df["rel_slope"] if "rel_slope" in df else df["diff"]
    return pd.DataFrame({"key": df[key], "measure": df.measure, "variant": df.variant,
                         "est": est, "lo": df.lo, "hi": df.hi})


def get(d, key, m, v):
    if d is None:
        return None
    r = d[(d.key == key) & (d.measure == m) & (d.variant == v)]
    return None if r.empty else r.iloc[0]


def grid(blocks, rows, ci, caption, label, row_head):
    box = "  \\resizebox{\\textwidth}{!}{%" if ci else "  {\\small%"
    """blocks: [(block title, tidy df)]; rows: [(key, row label)]."""
    ms = [m for m in a.measures if m in MEAS]
    cols = "l" + "cc" * len(ms)
    head1 = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{MEAS[m]}}}}}" for m in ms)
    head2 = " & ".join("$T{=}1$ & $T^*$" for _ in ms)
    lines = [f"\\begin{{table*}}[t]", "  \\centering", box,
             f"  \\begin{{tabular}}{{{cols}}}", "    \\hline",
             f"    \\textbf{{{row_head}}} & {head1} \\\\", f"     & {head2} \\\\", "    \\hline"]
    for title, d in blocks:
        if d is None:
            continue
        lines.append(f"    \\multicolumn{{{1 + 2 * len(ms)}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for key, lab in rows:
            if not any(get(d, key, m, v) is not None for m in ms for v in ["T1", "Tstar"]):
                continue
            cells = [cell(get(d, key, m, v), ci) for m in ms for v in ["T1", "Tstar"]]
            lines.append(f"    {lab} & " + " & ".join(cells) + " \\\\")
        lines.append("    \\hline")
    lines += ["  \\end{tabular}}", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", "\\end{table*}", ""]
    return "\n".join(lines)


def write(name, text):
    (TAB / f"{name}.tex").write_text(text)
    print(f"  wrote {TAB / name}.tex")


# ---------------------------------------------------------------- slopes per corpus
cv = tidy(read("calibration/joint/scaling_summary.csv"), "cv", "corpus")
lm = tidy(read("lmer/scaling/scaling_summary.csv"), "lmm", "group")
rows = [(c, NAME[c]) for c in a.corpora]
if cv is not None or lm is not None:
    write("tab_slopes", grid(
        [("Word-level, cross-validated", cv), ("Trial-level mixed models", lm)], rows, False,
        "Relative scaling slope: change in $\\Delta$LL per 10$\\times$ parameters, relative to the corpus "
        "mean over LMs (family intercepts for GPT-2 and Pythia). Negative values mean larger LMs fit "
        "reading times worse. Bold: 95\\% item-bootstrap CI excludes 0.",
        "tab:slopes", "Corpus"))

# ---------------------------------------------------------------- contrasts
cvc = tidy(read("calibration/joint/scaling_contrasts.csv"), "cv", "contrast")
lmc = tidy(read("lmer/scaling/scaling_contrasts.csv"), "lmm", "contrast")
LLM = [c for c in ["emtec", "great"] if c in a.corpora]
rows = [(f"{L} - {c}", f"{NAME[L]} $-$ {NAME[c]}") for L in LLM for c in a.corpora if c not in ("emtec", "great")]
if cvc is not None or lmc is not None:
    write("tab_contrasts", grid(
        [("Word-level, cross-validated", cvc), ("Trial-level mixed models", lmc)], rows, True,
        "Difference in relative scaling slope between LLM-written (EMTeC) and human-written text. "
        "Negative: stronger inverse scaling on LLM-written text. 95\\% item-bootstrap CIs in brackets; "
        "bold: CI excludes 0.",
        "tab:contrasts", "Contrast"))

# ---------------------------------------------------------------- robustness (one measure, CIs)
def robustness(measure="GD"):
    src = [("Word-level", "calibration/joint_split-decoding", "calibration/joint_split-typicality", "cv"),
           ("LMM", "lmer/scaling_decoding", "lmer/scaling_typicality", "lmm")]
    parts = []
    for lab, dpath, tpath, kind in src:
        dd = tidy(read(f"{dpath}/scaling_summary.csv"), kind, "group")
        tt = tidy(read(f"{tpath}/scaling_summary.csv"), kind, "group")
        parts.append((lab, pd.concat([x for x in (dd, tt) if x is not None]) if (dd is not None or tt is not None) else None))
    parts = [(l, d) for l, d in parts if d is not None]
    if not parts:
        return
    rows = [("EMTeC by decoding", None)] + [(f"emtec/{k}", f"\\quad {v}") for k, v in DEC.items()] + \
           [("EMTeC by typicality", None)] + \
           [(f"emtec/typ-{t}", f"\\quad {l}") for t, l in [("high", "most predictable third"), ("mid", "middle third"),
                                                          ("low", "least predictable third")]] + \
           [("OneStop by typicality", None)] + \
           [(f"onestop/typ-{t}", f"\\quad {l}") for t, l in [("high", "most predictable third"), ("mid", "middle third"),
                                                            ("low", "least predictable third")]]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{l}}}}}" for l, _ in parts)
    sub = " & ".join("$T{=}1$ & $T^*$" for _ in parts)
    L = ["\\begin{table}[t]", "  \\centering", "  \\resizebox{\\columnwidth}{!}{%",
         f"  \\begin{{tabular}}{{l{'cc' * len(parts)}}}", "    \\hline",
         f"    & {head} \\\\", f"    & {sub} \\\\", "    \\hline"]
    for key, lab in rows:
        if lab is None:
            L.append(f"    \\multicolumn{{{1 + 2 * len(parts)}}}{{l}}{{\\textit{{{key}}}}} \\\\")
            continue
        cells = [cell(get(d, key, measure, v)) for _, d in parts for v in ["T1", "Tstar"]]
        if all(c == "--" for c in cells):
            continue
        L.append(f"    {lab} & " + " & ".join(cells) + " \\\\")
    L += ["    \\hline", "  \\end{tabular}}",
          f"  \\caption{{Relative scaling slope for {MEAS[measure].lower()} within subsets. Typicality: "
          "tertiles of mean text surprisal (GPT-2) within each corpus. Bold: 95\\% CI excludes 0.}",
          "  \\label{tab:robustness}", "\\end{table}", ""]
    write("tab_robustness", "\n".join(L))


robustness()

# ---------------------------------------------------------------- T* per LM (appendix)
per = read("calibration/joint/per_lm.csv")
if per is not None:
    m = "GD"
    d = per[per.measure == m]
    lms = d[["lm", "params"]].drop_duplicates().sort_values(["params"])
    fam = lambda l: "GPT-2" if "gpt2" in l else "Pythia"
    nice = lambda l: l.split("__")[-1].replace("pythia-", "Pythia-").replace("gpt2-", "GPT-2 ").replace(
        "gpt2", "GPT-2 small") if l != "gpt2" else "GPT-2 small"
    cs = [c for c in a.corpora if c in set(d.corpus)]
    L = ["\\begin{table}[t]", "  \\centering", "  \\resizebox{\\columnwidth}{!}{%", f"  \\begin{{tabular}}{{lr{'c' * len(cs)}}}", "    \\hline",
         "    \\textbf{LM} & \\textbf{Params} & " + " & ".join(f"\\textbf{{{NAME[c]}}}" for c in cs) + " \\\\",
         "    \\hline"]
    for r in lms.itertuples():
        vals = [d[(d.corpus == c) & (d.lm == r.lm)].T_star for c in cs]
        cells = [f"{v.iat[0]:g}" if len(v) else "--" for v in vals]
        L.append(f"    {nice(r.lm)} & {r.params / 1e6:,.0f}M & " + " & ".join(cells) + " \\\\")
    L += ["    \\hline", "  \\end{tabular}}",
          f"  \\caption{{Best-fitting temperature $T^*$ for {MEAS[m].lower()} (point estimates; grid "
          "0.75--8). $T^* > 1$: the LM is over-confident relative to readers.}", "  \\label{tab:tstar}",
          "\\end{table}", ""]
    write("tab_tstar", "\n".join(L).replace("1,400M", "1.4B").replace("2,800M", "2.8B").replace("1,000M", "1B"))
print(f"done -> {TAB}/")
