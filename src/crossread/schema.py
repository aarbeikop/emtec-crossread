"""Canonical word-level schema every corpus is mapped into.

One row = one reader x one word (an interest area).
Durations are in ms; NaN means "not defined" (e.g. GD of a skipped word).
"""

ID_COLS = ["corpus", "language", "text_id", "subject_id", "word_idx", "word"]

# reading measures we harmonize across corpora
MEASURES = {
    "FFD": "first-fixation duration (first pass)",
    "GD": "gaze duration / first-pass reading time",
    "TFT": "total fixation time",
    "skip": "1 if skipped in first pass, else 0",
}

# text-level metadata (filled where available)
META_COLS = ["source", "model", "decoding", "genre"]  # source: human | machine

ALL_COLS = ID_COLS + list(MEASURES) + META_COLS


def check(df, name="corpus"):
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: harmonized frame is missing {missing}")
    if df.duplicated(["text_id", "subject_id", "word_idx"]).any():
        raise ValueError(f"{name}: duplicate (text_id, subject_id, word_idx) rows")
    return df[ALL_COLS + [c for c in df.columns if c not in ALL_COLS]]
