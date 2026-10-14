#!/usr/bin/env python
"""Pipeline entry point.

  python run.py columns emtec            # print raw columns to fix the config mapping
  python run.py harmonize                # raw corpora -> out/words.parquet, out/texts.parquet
  python run.py surprisal [--lm gpt2]    # -> out/surprisal/<lm>.parquet (default: all lms in config)
  python run.py analyze                  # -> out/results/<lm>/*.csv + plots
  python run.py all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))
from crossread import analysis, features, loaders, surprisal  # noqa: E402

OUT = Path("out")


def cfg_load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_columns(cfg, name):
    c = cfg["corpora"][name]
    raw = loaders.read_table(Path(cfg["data_root"]) / c["path"], c.get("r_object"), c.get("sep"))
    print(f"{name}: {raw.shape}\n{list(raw.columns)}\n")
    print(raw.head(5).to_string())


def cmd_harmonize(cfg):
    frames = []
    for name in cfg["use"]:
        df = loaders.load_corpus(name, cfg["corpora"][name], Path(cfg["data_root"]))
        print(f"[{name}] {len(df):,} rows | {df.text_id.nunique()} texts | "
              f"{df.subject_id.nunique()} readers | skip rate {df.skip.mean():.2f} | "
              f"mean GD {df.GD.mean():.0f} ms")
        frames.append(df)
    words = pd.concat(frames, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    words.to_parquet(OUT / "words.parquet")
    loaders.texts_from_words(words).to_parquet(OUT / "texts.parquet")


def cmd_surprisal(cfg, lm=None):
    texts = pd.read_parquet(OUT / "texts.parquet")
    (OUT / "surprisal").mkdir(parents=True, exist_ok=True)
    for name in ([lm] if lm else cfg["lms"]):
        label = Path(name).name if Path(name).exists() else name.replace("/", "__")
        if name == "unigram":
            parts = [surprisal.score_texts(t, surprisal.UnigramScorer(lang), name)
                     for lang, t in texts.groupby("language")]
            s = pd.concat(parts, ignore_index=True)
        else:
            scorer = surprisal.LMScorer(name, device=cfg.get("device", "cpu"))
            print(f"[{name}] bow-correction={scorer.bow_correction} ctx={scorer.max_ctx}")
            s = surprisal.score_texts(texts, scorer, name)
        s.to_parquet(OUT / "surprisal" / f"{label}.parquet")
        print(f"[{name}] scored {len(s):,} words, mean surprisal {s.surprisal.mean():.2f} bits")


def cmd_analyze(cfg):
    a = cfg["analysis"]
    words = pd.read_parquet(OUT / "words.parquet")
    texts = pd.read_parquet(OUT / "texts.parquet")
    rt = features.aggregate_rts(words)
    rt = rt[rt["n_readers"] >= a.get("min_readers", 1)]
    for sfile in sorted((OUT / "surprisal").glob("*.parquet")):
        lm = sfile.stem
        res = OUT / "results" / lm
        res.mkdir(parents=True, exist_ok=True)
        d = features.analysis_table(rt, features.word_predictors(texts, pd.read_parquet(sfile)))
        d.to_parquet(res / "analysis_table.parquet")

        groupings = {"corpus": ["corpus"], "source": ["source"]}
        if "emtec" in d["corpus"].values:
            groupings |= {"emtec_decoding": ["corpus", "decoding"],
                          "emtec_genre": ["corpus", "genre"],
                          "emtec_model": ["corpus", "model"]}
        dll, slp, bins = [], [], []
        for gname, by in groupings.items():
            sub = d if not gname.startswith("emtec") else d[d.corpus == "emtec"]
            for m in a["measures"]:
                r = analysis.delta_ll(sub, m, by, k=a["k_folds"], n_perm=a["n_perm"])
                dll.append(r.assign(grouping=gname))
                slp.append(analysis.slopes(sub, m, by, n_boot=a["n_boot"]).assign(grouping=gname))
                if gname in {"corpus", "emtec_decoding"}:
                    bins.append(analysis.binned(sub, m, by).assign(grouping=gname))
        dll, slp, bins = (pd.concat(x, ignore_index=True) for x in (dll, slp, bins))
        dll.to_csv(res / "delta_ll.csv", index=False)
        slp.to_csv(res / "slopes.csv", index=False)
        bins.to_csv(res / "binned.csv", index=False)
        print(f"\n=== {lm} ===")
        cols = ["grouping", "corpus", "source", "decoding", "measure", "slope_ms_per_bit", "ci_lo",
                "ci_hi", "mean_surprisal", "n_words"]
        print(slp[slp.grouping.isin(["corpus", "emtec_decoding"])]
              .reindex(columns=cols).dropna(axis=1, how="all").round(2).to_string(index=False))
        try:
            plots(slp, bins, res)
        except ImportError:
            print("(matplotlib not installed, skipping plots)")


def plots(slp, bins, res):
    import matplotlib.pyplot as plt

    for m in bins["measure"].unique():
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        s = slp[(slp.grouping.isin(["corpus", "emtec_decoding"])) & (slp.measure == m)].copy()
        s["label"] = s["corpus"] + s.get("decoding", pd.Series("", index=s.index)).fillna("").map(
            lambda x: f" / {x}" if x else "")
        s = s.reset_index(drop=True)
        ax[0].errorbar(s.slope_ms_per_bit, range(len(s)),
                       xerr=[s.slope_ms_per_bit - s.ci_lo, s.ci_hi - s.slope_ms_per_bit], fmt="o")
        ax[0].set_yticks(range(len(s)), s.label)
        ax[0].axvline(0, color="grey", lw=0.5)
        ax[0].set_xlabel(f"surprisal slope ({m} per bit)")
        b = bins[(bins.grouping == "corpus") & (bins.measure == m)]
        for c, g in b.groupby("corpus"):
            ax[1].errorbar(g.surprisal_mid, g.resid_mean, yerr=g.resid_se, marker="o", label=c, capsize=2)
        ax[1].set_xlabel("surprisal (bits, shared bins)")
        ax[1].set_ylabel(f"{m} residual after length/freq baseline")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(res / f"slopes_and_bins_{m}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["columns", "harmonize", "surprisal", "analyze", "all"])
    p.add_argument("corpus", nargs="?")
    p.add_argument("--config", default="configs/corpora.yaml")
    p.add_argument("--lm", default=None, help="one HF id / local path, or 'unigram' (no weights)")
    args = p.parse_args()
    cfg = cfg_load(args.config)
    if args.stage == "columns":
        cmd_columns(cfg, args.corpus)
    if args.stage in {"harmonize", "all"}:
        cmd_harmonize(cfg)
    if args.stage in {"surprisal", "all"}:
        cmd_surprisal(cfg, args.lm)
    if args.stage in {"analyze", "all"}:
        cmd_analyze(cfg)
