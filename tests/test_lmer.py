"""End to end: toy world -> export_trials.py -> r/scaling_lmer.R (needs Rscript + lme4 + data.table).
Planted (see test_calibration.simulate_scaling): both corpora show inverse scaling at T=1, but
temperature only fixes it on the human corpus, so emtec - provo at T* must be clearly < 0."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from crossread import calibration  # noqa: E402
from test_calibration import simulate_scaling  # noqa: E402


def main():
    if not shutil.which("Rscript"):
        print("Rscript not found, skipping")
        return
    w, t, s = simulate_scaling()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        (out / "surprisal").mkdir(parents=True)
        w.to_parquet(out / "words.parquet")
        t.to_parquet(out / "texts.parquet")
        for k, v in s.items():
            v.to_parquet(out / "surprisal" / f"{k}.parquet")
            est, *_ = calibration.tstar(w, t, v, ["GD"], n_boot=20)
            (out / "calibration" / k).mkdir(parents=True)
            est.to_csv(out / "calibration" / k / "tstar.csv", index=False)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, str(ROOT / "export_trials.py"), "--measures", "GD",
                        "--typ-lm", "EleutherAI__pythia-70m"], cwd=tmp, env=env, check=True)
        tr = pd.read_csv(out / "trials" / "trials.csv")
        assert {"s__pythia-70m", "ps__pythia-70m", "s__pythia-1.4b__GD", "typ"} <= set(tr.columns)
        assert tr["s__pythia-1.4b__GD"].notna().all()
        subprocess.run(["Rscript", str(ROOT / "r" / "scaling_lmer.R"), "--measures", "GD",
                        "--boot", "200", "--cores", "2"], cwd=tmp, check=True)
        con = pd.read_csv(out / "lmer" / "scaling" / "scaling_contrasts.csv")
        summ = pd.read_csv(out / "lmer" / "scaling" / "scaling_summary.csv").set_index(["group", "variant"])
        assert summ.loc[("provo", "T1"), "rel_slope"] < 0 and summ.loc[("emtec", "T1"), "rel_slope"] < 0
        assert abs(summ.loc[("provo", "Tstar"), "rel_slope"]) < abs(summ.loc[("provo", "T1"), "rel_slope"])
        c = con.set_index("variant").loc["Tstar"]
        assert c["hi"] < 0, c
        for split in ["decoding", "typicality", "generator"]:
            subprocess.run(["Rscript", str(ROOT / "r" / "scaling_lmer.R"), "--measures", "GD", "--split", split,
                            "--boot", "50", "--cores", "2", "--no-tstar"], cwd=tmp, check=True)
            assert (out / "lmer" / f"scaling_{split}" / "scaling_contrasts.csv").exists()
    print("ok")


if __name__ == "__main__":
    main()
