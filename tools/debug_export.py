import sys, glob
import pandas as pd
sys.path.insert(0, "src")
from crossread import features, scaling
w = pd.read_parquet("out/words.parquet"); t = pd.read_parquet("out/texts.parquet")
print("words:", w.corpus.value_counts().to_dict())
print("texts:", t.corpus.value_counts().to_dict())
print("dtypes words:", w[["text_id", "word_idx"]].dtypes.to_dict(), "texts:", t[["text_id", "word_idx"]].dtypes.to_dict())
rt = features.aggregate_rts(w)
print("aggregated:", rt.corpus.value_counts().to_dict())
for f in sorted(glob.glob("out/surprisal/*.parquet")):
    s = pd.read_parquet(f)
    sub = s[s.corpus.isin(["great", "oasst"])]
    p = features.word_predictors(t, s)
    a = features.analysis_table(rt, p)
    print(f.split("/")[-1], "| surp rows", sub.corpus.value_counts().to_dict(),
          "| surp NaN", int(sub.surprisal.isna().sum()), "| word_idx", s.word_idx.dtype,
          "| analysis", a[a.corpus.isin(["great", "oasst"])].corpus.value_counts().to_dict())
