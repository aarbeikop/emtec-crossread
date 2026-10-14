# Trial-level mixed-model version of the main result: does inverse scaling depend on
# whether the text was written by an LLM?
#
#   python export_trials.py                      # once: out/trials/trials.csv + lms.csv
#   Rscript r/scaling_lmer.R                     # per corpus
#   Rscript r/scaling_lmer.R --split decoding    # EMTeC by decoding strategy
#   Rscript r/scaling_lmer.R --split typicality  # every corpus by text-typicality tertile
#   Rscript r/scaling_lmer.R --split generator   # EMTeC by the LLM that wrote the text
#
# Options: --measures FFD GD TFT | --log (log RT) | --cores 4 | --boot 1000 | --no-tstar
#          --corpora emtec onestop | --lms gpt2 pythia-70m ...  (subsets for quick runs)
#
# For each group x measure x LM, as in Oh & Schuler (2023) / Liu et al. (2024):
#   base: RT ~ length * zipf + prev_length * prev_zipf + (1|subject) + (1|item) [+ (1|text)]
#   full: base + surprisal + prev_surprisal              (ML fits, REML = FALSE)
# Delta-LL per observation = conditional log density (fitted values include the random
# effects) of full minus base. Summed per item, it feeds the same scale-free scaling slope as
# the Python pipeline: regress Delta-LL / group mean on log10(params) with LM-family
# intercepts. CIs: item bootstrap, one draw per corpus shared across LMs, measures and that
# corpus's groups (so within-corpus contrasts are paired). The marginal lmer logLik
# difference is reported too, for comparability with earlier papers.
# Also written: the surprisal coefficient (ms per bit, or log-ms per bit with --log) per LM.

suppressPackageStartupMessages({ library(data.table); library(lme4); library(parallel) })

# ---------- arguments ----------
args <- commandArgs(trailingOnly = TRUE)
opt <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) return(default)
  j <- i + 1
  vals <- character(0)
  while (j <= length(args) && !startsWith(args[j], "--")) { vals <- c(vals, args[j]); j <- j + 1 }
  if (length(vals) == 0) TRUE else vals
}
SPLIT    <- opt("split", "none")
MEASURES <- opt("measures", c("FFD", "GD", "TFT"))
LOGRT    <- isTRUE(opt("log", FALSE))
CORES    <- as.integer(opt("cores", max(1, detectCores() - 1)))
NBOOT    <- as.integer(opt("boot", 1000))
TSTAR    <- !isTRUE(opt("no-tstar", FALSE))
CORPORA  <- opt("corpora", NULL)
LMSEL    <- opt("lms", NULL)
INDIR    <- opt("in", "out/trials")
set.seed(1)

d   <- fread(file.path(INDIR, "trials.csv"), na.strings = c("", "NA", "NaN"))
lms <- fread(file.path(INDIR, "lms.csv"))
if (!is.null(LMSEL)) lms <- lms[key %in% LMSEL]
if (!is.null(CORPORA)) d <- d[corpus %in% CORPORA]
lms <- lms[order(family, params)]
stopifnot(nrow(lms) >= 3)

d[, group := switch(SPLIT,
  none = corpus,
  decoding = ifelse(corpus == "emtec", paste0("emtec/", decoding), corpus),
  typicality = paste0(corpus, "/typ-", typ),
  generator = ifelse(corpus == "emtec", paste0("emtec/", model), corpus),
  stop("--split must be none, decoding, typicality or generator"))]

# variants: T=1 always, T* if the export has the columns
variants <- function(m) {
  out <- list()
  for (k in lms$key) {
    out[[paste(k, "T1", sep = "|")]] <- c(paste0("s__", k), paste0("ps__", k))
    sc <- paste0("s__", k, "__", m)
    if (TSTAR && sc %in% names(d)) out[[paste(k, "Tstar", sep = "|")]] <- c(sc, paste0("ps__", k, "__", m))
  }
  out
}

zs <- function(x) (x - mean(x)) / sd(x)
ctrl <- lmerControl(optimizer = "bobyqa", calc.derivs = FALSE,
                    check.conv.singular = .makeCC(action = "ignore", tol = 1e-4))

fit_group <- function(g, m) {
  x <- d[group == g & !is.na(get(m)) & get(m) > 0]
  vs <- variants(m)
  t1 <- unlist(vs[grepl("\\|T1$", names(vs))])
  need <- unique(c(t1, "length", "zipf", "prev_length", "prev_zipf"))
  x <- x[complete.cases(x[, ..need])]
  if (nrow(x) < 500) { cat(sprintf("  %-22s %-4s skipped: %d complete rows\n", g, m, nrow(x))); return(NULL) }
  # T* variants only where that LM has T* for this corpus (calibrate.py must have seen it)
  ok <- sapply(vs, function(cl) all(cl %in% names(x)) && !anyNA(x[[cl[1]]]) && mean(is.na(x[[cl[2]]])) < .01)
  if (any(!ok)) cat(sprintf("  %-22s %-4s no T* for: %s\n", g, m,
                            paste(sub("\\|.*", "", names(vs)[!ok]), collapse = ", ")))
  vs <- vs[ok]
  for (cl in vs) x <- x[!is.na(get(cl[2]))]
  x[, y := if (LOGRT) log(get(m)) else get(m)]
  for (v in c("length", "zipf", "prev_length", "prev_zipf")) x[, (paste0(v, "_z")) := zs(get(v))]
  x[, `:=`(subject_id = factor(subject_id), item_id = factor(item_id), text_id = factor(text_id))]
  re <- "(1 | subject_id) + (1 | item_id)"
  if (nlevels(x$text_id) > nlevels(x$item_id)) re <- paste(re, "+ (1 | text_id)")
  fb <- as.formula(paste("y ~ length_z * zipf_z + prev_length_z * prev_zipf_z +", re))
  base <- lmer(fb, data = x, REML = FALSE, control = ctrl)
  ll_base <- dnorm(x$y, fitted(base), sigma(base), log = TRUE)
  item <- as.character(x$item_id)

  res <- mclapply(names(vs), function(nm) {
    cols <- vs[[nm]]
    x[, `:=`(s_ = get(cols[1]), ps_ = get(cols[2]))]
    full <- lmer(update(fb, . ~ . + s_ + ps_), data = x, REML = FALSE, control = ctrl)
    dll <- dnorm(x$y, fitted(full), sigma(full), log = TRUE) - ll_base
    cf <- summary(full)$coefficients
    list(nm = nm, sums = tapply(dll, item, sum),
         marg = as.numeric(logLik(full) - logLik(base)) / nrow(x),
         coef = cf["s_", "Estimate"], se = cf["s_", "Std. Error"])
  }, mc.cores = CORES)
  list(group = g, measure = m, n = nrow(x), n_items = nlevels(x$item_id),
       count = table(item), res = res)
}

# ---------- fit everything ----------
groups <- sort(unique(d$group))
fits <- list()
for (g in groups) for (m in MEASURES) {
  t0 <- Sys.time()
  f <- fit_group(g, m)
  if (!is.null(f)) {
    fits[[paste(g, m)]] <- f
    cat(sprintf("  %-22s %-4s %8d obs  %4d items  %.1f min\n", g, m, f$n, f$n_items,
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
}

# ---------- scaling slopes ----------
slope <- function(y, x, fam) {
  X <- if (length(unique(fam)) > 1) cbind(x, model.matrix(~ 0 + factor(fam))) else cbind(x, 1)
  qr.coef(qr(X), y)[1]
}
lm_of <- function(nm) sub("\\|.*", "", nm)
var_of <- function(nm) sub(".*\\|", "", nm)
corp_of <- function(g) sub("/.*", "", g)

per_lm <- rbindlist(lapply(fits, function(f) rbindlist(lapply(f$res, function(r) data.table(
  group = f$group, measure = f$measure, lm = lm_of(r$nm), variant = var_of(r$nm),
  dLL_cond = sum(r$sums) / f$n, dLL_marg = r$marg, coef = r$coef, se = r$se)))))
per_lm <- merge(per_lm, lms[, .(lm = key, family, params)], by = "lm")

rel <- function(f, w) {  # f = one group x measure fit; w = named item weights (or NULL)
  out <- list()
  for (v in unique(var_of(sapply(f$res, `[[`, "nm")))) {
    rs <- Filter(function(r) var_of(r$nm) == v, f$res)
    if (length(rs) < 3) next
    ks <- sapply(rs, function(r) lm_of(r$nm))
    info <- lms[match(ks, key)]
    items <- names(f$count)
    ww <- if (is.null(w)) rep(1, length(items)) else w[items]
    ww[is.na(ww)] <- 0
    tot <- sum(f$count * ww)
    y <- sapply(rs, function(r) sum(r$sums[items] * ww) / tot)
    out[[v]] <- slope(y / mean(y), log10(info$params), info$family)
  }
  out
}

point <- lapply(fits, rel, w = NULL)
draws <- lapply(fits, function(f) list())
for (c in unique(corp_of(groups))) {
  keys <- names(fits)[corp_of(sapply(fits, `[[`, "group")) == c]
  if (!length(keys)) next
  items <- unique(unlist(lapply(fits[keys], function(f) names(f$count))))
  for (b in seq_len(NBOOT)) {
    w <- tabulate(sample.int(length(items), replace = TRUE), length(items))
    names(w) <- items
    for (k in keys) draws[[k]][[b]] <- rel(fits[[k]], w)
  }
}
getd <- function(k, v) sapply(draws[[k]], function(z) if (is.null(z[[v]])) NA else z[[v]])
ci <- function(x) c(quantile(x, c(.025, .975), na.rm = TRUE),
                    p = min(1, 2 * min(mean(x <= 0, na.rm = TRUE), mean(x >= 0, na.rm = TRUE))))

summ <- rbindlist(lapply(names(fits), function(k) rbindlist(lapply(names(point[[k]]), function(v) {
  q <- ci(getd(k, v))
  data.table(group = fits[[k]]$group, measure = fits[[k]]$measure, variant = v,
             n_obs = fits[[k]]$n, n_items = fits[[k]]$n_items,
             rel_slope = point[[k]][[v]], lo = q[1], hi = q[2], p_boot = q[3])
}))))

# ---------- contrasts ----------
con <- list()
# weighted sums of group slopes; draw b of different corpora come from independent item
# resamples, so differences across corpora are unpaired and within a corpus are paired
add <- function(label, m, v, ka, kb, paired, wa = NULL, wb = NULL) {
  if (!all(c(ka, kb) %in% names(fits))) return()
  if (any(sapply(c(ka, kb), function(k) is.null(point[[k]][[v]])))) return()
  wa <- if (is.null(wa)) rep(1 / length(ka), length(ka)) else wa
  wb <- if (is.null(wb)) rep(1 / length(kb), length(kb)) else wb
  pt <- function(ks, ws) sum(ws * sapply(ks, function(k) point[[k]][[v]]))
  dr <- function(ks, ws) Reduce(`+`, Map(function(k, w) w * getd(k, v), ks, ws))
  q <- ci(dr(ka, wa) - dr(kb, wb))
  con[[length(con) + 1]] <<- data.table(measure = m, variant = v, contrast = label, paired = paired,
                                        diff = pt(ka, wa) - pt(kb, wb), lo = q[1], hi = q[2], p_boot = q[3])
}
K <- function(g, m) paste(g, m)
LLM <- c("emtec", "great")  # corpora of LLM-written text; contrasts are LLM - human
for (m in MEASURES) for (v in c("T1", "Tstar")) {
  if (SPLIT == "none") {
    for (L in intersect(LLM, groups)) for (c in setdiff(groups, LLM))
      add(paste(L, "-", c), m, v, K(L, m), K(c, m), FALSE)
  }
  if (SPLIT == "decoding") {
    em <- function(s) K(paste0("emtec/", s), m)
    det <- em(c("greedy_search", "beam_search")); sto <- em(c("sampling", "topk", "topp"))
    trunc <- em(c("topk", "topp"))
    add("emtec: deterministic - stochastic", m, v, det, sto, TRUE)
    add("emtec: sampling - truncated (topk, topp)", m, v, em("sampling"), trunc, TRUE)
    add("emtec: sampling - all others", m, v, em("sampling"), c(det, trunc), TRUE)
    for (c in setdiff(groups, c(grep("^emtec", groups, value = TRUE), LLM))) {
      add(paste("emtec truncated+greedy+beam -", c), m, v, c(det, trunc), K(c, m), FALSE)
      add(paste("emtec sampling -", c), m, v, em("sampling"), K(c, m), FALSE)
    }
  }
  if (SPLIT == "generator") {
    gens <- sort(grep("^emtec/", groups, value = TRUE))
    for (i in seq_along(gens)) {
      for (j in seq_along(gens)) if (j > i) add(paste(gens[i], "-", gens[j]), m, v, K(gens[i], m), K(gens[j], m), TRUE)
      for (c in setdiff(groups, c(gens, LLM))) add(paste(gens[i], "-", c), m, v, K(gens[i], m), K(c, m), FALSE)
    }
  }
  if (SPLIT == "typicality") {
    tk <- function(c, t) K(paste0(c, "/typ-", t), m)
    cs <- unique(corp_of(groups))
    for (c in cs) add(paste0(c, ": high - low typicality"), m, v, tk(c, "high"), tk(c, "low"), TRUE)
    if ("emtec" %in% cs) for (c in setdiff(cs, LLM)) {
      for (t in c("high", "mid", "low")) add(sprintf("emtec - %s (typ-%s)", c, t), m, v, tk("emtec", t), tk(c, t), FALSE)
      # difference in differences: (emtec high - low) - (c high - low)
      add(paste("typicality effect: emtec -", c), m, v,
          c(tk("emtec", "high"), tk("emtec", "low"), tk(c, "low")),
          c(tk(c, "high")), FALSE, wa = c(1, -1, 1), wb = 1)
    }
  }
}
con <- rbindlist(con)

# ---------- write ----------
tag <- paste0(if (SPLIT == "none") "" else paste0("_", SPLIT), if (LOGRT) "_log" else "")
od <- file.path("out", "lmer", paste0("scaling", tag)); dir.create(od, recursive = TRUE, showWarnings = FALSE)
fwrite(per_lm, file.path(od, "per_lm.csv")); fwrite(summ, file.path(od, "scaling_summary.csv"))
fwrite(con, file.path(od, "scaling_contrasts.csv"))
options(width = 200, digits = 3)
cat("\n== relative scaling slope per group (< 0 = bigger LMs fit worse; trial-level LMMs) ==\n")
print(summ[order(measure, variant, group)], row.names = FALSE)
if (nrow(con)) { cat("\n== contrasts (< 0 = more inverse scaling in the first group) ==\n"); print(con, row.names = FALSE) }
cat(sprintf("\n== surprisal coefficient (%s per bit) at T=1 ==\n", if (LOGRT) "log-ms" else "ms"))
print(dcast(per_lm[variant == "T1"], measure + group ~ lm, value.var = "coef"), row.names = FALSE)
cat("\nwrote", od, "\n")
