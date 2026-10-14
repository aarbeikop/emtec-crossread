#!/bin/zsh
# Overnight batch: refresh every number in the paper with all 9 LMs and all corpora, plus a
# log-RT robustness check. Each step logs to logs/<step>.txt; a failing step does not stop the rest.
#
#   caffeinate -i zsh overnight.sh            # caffeinate keeps the Mac awake
#   BIG=1 caffeinate -i zsh overnight.sh      # also score Pythia-6.9B first (needs >= 32 GB RAM)

mkdir -p logs
TEMPS=(0.75 1.25 1.5 2 2.5 3 4 5 6 8)
step() { local name=$1; shift; echo "== $(date +%H:%M) $name"; "$@" > logs/$name.txt 2>&1 || echo "   FAILED: see logs/$name.txt"; }

if [[ -n "$BIG" ]]; then
  step score_6.9b   python run.py surprisal --lm EleutherAI/pythia-6.9b --temps $TEMPS --device mps --dtype float16
  step calib_6.9b   python calibrate.py --lm EleutherAI/pythia-6.9b
fi

# word-level (cross-validated) robustness, now with 9 LMs and the chat corpora
step rob_main        python calib_scaling.py
step rob_genre       python calib_scaling.py --emtec-genres non-fiction article_synopsis summarization
step rob_decoding    python calib_scaling.py --split decoding
step rob_typicality  python calib_scaling.py --split typicality
step rob_generator   python calib_scaling.py --split generator
step rob_single_tok  python calib_scaling.py --words single
step rob_multi_tok   python calib_scaling.py --words multi
step rob_quadratic   python calib_scaling.py --link quadratic

# trial-level mixed models: main + splits, raw ms and log ms
step export          python export_trials.py
step lmer_main       Rscript r/scaling_lmer.R --cores 6
step lmer_decoding   Rscript r/scaling_lmer.R --split decoding --cores 6
step lmer_typicality Rscript r/scaling_lmer.R --split typicality --cores 6
step lmer_generator  Rscript r/scaling_lmer.R --split generator --cores 6
step lmer_main_log   Rscript r/scaling_lmer.R --log --cores 6

# tables + figures (word-level and LMM robustness versions)
step tables          python make_tables.py
step figures         python make_figures.py --corpora emtec onestop provo
step figures_lmm     python make_figures.py --source lmm --corpora emtec onestop provo
echo "== $(date +%H:%M) done; logs in logs/"
