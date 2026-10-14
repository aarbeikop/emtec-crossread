# Trial-level mixed model for the human vs machine contrast.
# Input: out/results/<lm>/analysis_table.parquet is word-level (averaged);
# for trial-level models, join out/words.parquet (one row per reader x word)
# with the word predictors. Do that join in Python and write a CSV, or adapt here.
#
# Key term: surprisal_z:source. Is the surprisal effect different on machine text?
# Caveat: corpora differ in lab, tracker, readers and layout, so interpret the
# interaction as "slope relative to each corpus's own baseline", never raw RT levels.

library(arrow); library(dplyr); library(lme4)

words <- read_parquet("out/words.parquet")
preds <- read_parquet("out/results/gpt2/analysis_table.parquet") |>
  select(corpus, text_id, word_idx, surprisal, prev_surprisal, length, zipf,
         prev_length, prev_zipf)

d <- inner_join(words, preds, by = c("corpus", "text_id", "word_idx")) |>
  filter(!is.na(GD)) |>
  mutate(across(c(surprisal, prev_surprisal, length, zipf, prev_length, prev_zipf),
                ~ as.numeric(scale(.x)), .names = "{.col}_z"),
         source = factor(source, levels = c("human", "machine")))
contrasts(d$source) <- contr.sum(2) / 2

m <- lmer(log(GD) ~ (surprisal_z + prev_surprisal_z + length_z * zipf_z +
                      prev_length_z * prev_zipf_z) * source +
            (1 + surprisal_z | subject_id) + (1 | text_id),
          data = d, control = lmerControl(optimizer = "bobyqa"))
summary(m)

# EMTeC only: does the surprisal effect depend on decoding strategy / genre?
e <- filter(d, corpus == "emtec") |> mutate(decoding = factor(decoding), genre = factor(genre))
contrasts(e$decoding) <- contr.sum(nlevels(e$decoding))
contrasts(e$genre) <- contr.sum(nlevels(e$genre))
m_e <- lmer(log(GD) ~ surprisal_z * (decoding + genre) + prev_surprisal_z +
              length_z * zipf_z + (1 + surprisal_z | subject_id) + (1 | text_id), data = e)
summary(m_e)
