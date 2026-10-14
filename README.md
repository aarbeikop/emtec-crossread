# crossread: predictability effects on machine- vs human-written text

Starter pipeline for asking, across eye-tracking corpora:

> Is the effect of LM surprisal on reading the same when the text was produced by an LM (EMTeC) as when a human wrote it (Provo, MECO-en, ...)? Does it depend on decoding strategy and genre?

Why this is interesting: decoding strategies push machine text into a different region of the surprisal distribution. Greedy and beam search produce low-surprisal, flat text; ancestral sampling produces a long tail. Surprisal theory is usually tested on human text, so EMTeC lets you test whether the RT ~ surprisal link (slope, linearity) holds where machine text actually lives.

## Pipeline

```
python run.py columns emtec        # print raw columns, then fix configs/corpora.yaml
python run.py harmonize            # -> out/words.parquet (reader x word), out/texts.parquet
python run.py surprisal            # -> out/surprisal/<lm>.parquet, one file per LM in config
python run.py analyze              # -> out/results/<lm>/{delta_ll,slopes,binned}.csv + plots
```

| step | file | what it does |
|---|---|---|
| harmonize | `src/crossread/loaders.py` | config-driven mapping of any word-level table (csv / parquet / .rda) to one schema: `FFD, GD, TFT, skip` + `source, model, decoding, genre` |
| surprisal | `src/crossread/surprisal.py` | word-level surprisal + entropy in bits from any HF causal LM, with the Pimentel & Meister (2024) beginning-of-word correction and a sliding window for long texts |
| features | `src/crossread/features.py` | length, wordfreq Zipf, spillover (previous word), per-word means over readers, standard exclusions |
| analyze | `src/crossread/analysis.py` | (1) held-out ΔLL with text-grouped 10-fold CV and a text-level sign-flip permutation test; (2) surprisal slope in ms/bit with text-cluster bootstrap CIs; (3) residual RT by shared surprisal bins |
| trial-level | `r/crossdata_lmer.R` | lme4 model with `surprisal × source` and, within EMTeC, `surprisal × (decoding + genre)` |

Groupings run automatically: per corpus, human vs machine, and within EMTeC per decoding strategy, genre and generating model.

## Before running on real data

1. Put files under `data/` (paths in `configs/corpora.yaml`), comment out corpora you don't have in `use:`.
2. Every column mapping marked `VERIFY` is a guess. Run `python run.py columns <corpus>` and fix.
3. EMTeC: each item-condition was read by only ~7–8 people (14-list Latin square). Word-level means are noisy, so treat the R trial-level model as the main inferential test and the Python ΔLL/slopes as the overview.
4. Model weights: `lms:` takes HF ids or local folders (`gpt2`, `EleutherAI/pythia-160m`, ...). Use the same LMs for every corpus.

Tests: `python tests/test_analysis.py` checks that the estimators recover planted slopes. `python tests/make_synthetic.py` plus `--config tests/synthetic.yaml` runs the whole pipeline on fake data; its slopes are meaningless because unigram surprisal is collinear with frequency, so use it only for plumbing.

## Where to push it

- **Distribution matching.** Machine and human texts differ in mean surprisal. Compare slopes on surprisal-matched word subsets (or include `surprisal × source` within the shared bins) so a slope difference isn't just a range effect.
- **Generator vs evaluator surprisal (EMTeC only).** Add the generator's own transition log-probs (from EMTeC's transition scores) as a second predictor. The decoding-truncated distribution *q* vs the model's *p* (top-k/top-p renormalization, temperature 0.8) gives "decoding-aware surprisal" log q − log p as a distortion term.
- **LM scale curve.** Run the pythia suite (70m to 12b where feasible) on all corpora. Does the inverse-scaling effect (Oh & Schuler 2023) look different on machine text, which was produced by models of similar scale?
- **Cross-lingual check.** PoTeC (de) and CopCo (da) with a multilingual LM, as a generalization appendix rather than the main contrast. The config has stubs.
