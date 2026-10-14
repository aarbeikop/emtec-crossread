# Trial-level test: does the RT ~ surprisal curve differ between human and machine text?
# Difference-smooth GAM (ordered-factor trick): the s(surprisal):source_omachine term IS the
# machine-minus-human curve; its p-value tests whether the curves differ anywhere.
# Run from the repo root after `python run.py analyze`.
library(arrow); library(dplyr); library(mgcv)

LM <- "gpt2"
words <- read_parquet("out/words.parquet")
preds <- read_parquet(file.path("out/results", LM, "analysis_table.parquet")) |>
  select(corpus, text_id, word_idx, surprisal, prev_surprisal, length, zipf, prev_length, prev_zipf)

d <- inner_join(words, preds, by = c("corpus", "text_id", "word_idx")) |>
  filter(!is.na(GD), GD > 0) |>
  mutate(source = factor(source, levels = c("human", "machine")),
         source_o = as.ordered(source),
         subject_id = factor(subject_id), text_id = factor(text_id))
contrasts(d$source_o) <- "contr.treatment"

m <- bam(GD ~ source_o +
           s(surprisal, k = 8) + s(surprisal, by = source_o, k = 8) +   # reference curve + difference
           s(prev_surprisal, k = 6) +
           te(length, zipf) + te(prev_length, prev_zipf) +
           s(subject_id, bs = "re") + s(subject_id, surprisal, bs = "re") + s(text_id, bs = "re"),
         data = d, discrete = TRUE, nthreads = 4)
summary(m)   # look at the s(surprisal):source_omachine row

# plot the difference curve (machine - human) restricted to the shared surprisal range
rng <- with(d, c(max(tapply(surprisal, source, quantile, .01)), min(tapply(surprisal, source, quantile, .99))))
pdf(file.path("out/results", LM, "extras", "gam_difference_smooth.pdf"), 6, 4)
plot(m, select = 2, shade = TRUE, xlim = rng, xlab = "surprisal (bits)",
     ylab = "machine - human GD (ms)", main = "difference smooth"); abline(h = 0, lty = 2)
dev.off()

# robustness: same model on log(GD), and on TFT
m_log <- update(m, log(GD) ~ .); summary(m_log)$s.table
