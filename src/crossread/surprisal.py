"""Word-level surprisal and entropy (in bits) from any HF causal LM.

Word probability = product of its sub-word token probabilities, with the
beginning-of-word correction of Pimentel & Meister (2024, "How to compute the
probability of a word"): for tokenizers whose word-initial tokens carry the
space marker (GPT-2 'Ġ', SentencePiece '▁'), we multiply by P(next token
starts a new word | ctx, w) and divide by the same quantity before w. Without
this, surprisal is systematically biased for words followed by punctuation.

`UnigramScorer` is a weights-free fallback (wordfreq) for smoke tests only.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LN2 = math.log(2)


class UnigramScorer:
    name = "unigram-wordfreq"

    def __init__(self, language: str = "en"):
        from wordfreq import word_frequency

        self.f = word_frequency
        self.lang = language

    def score(self, words: list[str]) -> pd.DataFrame:
        s = [-math.log2(max(self.f(w.strip(".,;:!?\"'()").lower(), self.lang), 1e-9)) for w in words]
        return pd.DataFrame({"surprisal": s, "entropy": np.nan, "n_tokens": 1})


class LMScorer:
    def __init__(self, model_name: str, device: str = "cpu", max_ctx: int | None = None,
                 bow_correction: bool = True, dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.max_ctx = max_ctx or min(getattr(self.model.config, "max_position_embeddings", 1024), 2048)
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        markers = ("Ġ", "▁", "Ċ", "<0x0A>")
        is_bow = np.array([bool(t) and t.startswith(markers) for t in vocab])
        if self.tok.eos_token_id is not None:
            is_bow[self.tok.eos_token_id] = True
        n_logits = self.model.get_output_embeddings().weight.shape[0]
        mask = np.zeros(n_logits, dtype=bool)
        mask[: len(is_bow)] = is_bow[:n_logits]
        self.bow_mask = torch.tensor(mask, device=device)
        # how does the tokenizer mark a mid-sentence word? ("Ġ" byte-level BPE, "▁" SentencePiece)
        probe = self.tok.convert_ids_to_tokens(self.tok("x the", add_special_tokens=False).input_ids)
        mid = next((t for t in reversed(probe) if t and t.lstrip("Ġ▁") in {"the", "t"}), probe[-1])
        marks_bow = mid.startswith(("Ġ", "▁"))
        # byte-level BPE needs an explicit leading space so word 1 looks like every other word
        self.space_prefix = mid.startswith("Ġ")
        # the correction only makes sense if word-initial tokens carry the marker
        self.bow_correction = bow_correction and marks_bow

    def _positionwise(self, seq: list[int]):
        """For every source position p: log p(seq[p+1]), log P(bow at p+1), entropy at p+1."""
        torch = self.torch
        n = len(seq)
        lp_next = np.full(n, np.nan)
        bow_lse = np.full(n, np.nan)
        ent = np.full(n, np.nan)
        stride = max(1, self.max_ctx // 2)
        p0 = 0
        ids = torch.tensor(seq, device=self.device)
        with torch.no_grad():
            while p0 < n:
                p1 = min(p0 + stride, n)
                c0 = max(0, p1 - self.max_ctx)
                logits = self.model(ids[c0:p1].unsqueeze(0)).logits[0].float()
                logp = torch.log_softmax(logits, -1)[p0 - c0: p1 - c0]  # rows = sources p0..p1-1
                bow_lse[p0:p1] = torch.logsumexp(logp[:, self.bow_mask], -1).cpu().numpy()
                ent[p0:p1] = (-(logp.exp() * logp).sum(-1)).cpu().numpy()
                tgt = ids[p0 + 1: p1 + 1]
                k = len(tgt)
                lp_next[p0:p0 + k] = logp[:k].gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy()
                p0 = p1
        return lp_next, bow_lse, ent

    def score(self, words: list[str]) -> pd.DataFrame:
        text, spans, pos = "", [], 0
        if self.space_prefix:
            text, pos = " ", 1
        for i, w in enumerate(words):
            if i:
                text += " "
                pos += 1
            spans.append((pos, pos + len(w)))
            text += w
            pos += len(w)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        seq = [self.bos] + enc.input_ids
        lp_next, bow_lse, ent = self._positionwise(seq)

        ends = np.array([e for _, e in spans])
        word_of_tok = np.searchsorted(ends, [e for _, e in enc.offset_mapping], side="left")
        word_of_tok = np.minimum(word_of_tok, len(words) - 1)

        out = []
        for w in range(len(words)):
            toks = np.where(word_of_tok == w)[0] + 1  # positions in seq
            if len(toks) == 0:
                out.append((np.nan, np.nan, 0))
                continue
            first, last = toks.min(), toks.max()
            logp = lp_next[toks - 1].sum()
            if self.bow_correction:
                logp += bow_lse[last] - bow_lse[first - 1]
            out.append((-logp / LN2, ent[first - 1] / LN2, len(toks)))
        return pd.DataFrame(out, columns=["surprisal", "entropy", "n_tokens"])


def score_texts(texts: pd.DataFrame, scorer, lm_label: str) -> pd.DataFrame:
    """texts: rows (corpus, text_id, word_idx, word) as returned by loaders.texts_from_words."""
    res = []
    for (corpus, tid), g in texts.groupby(["corpus", "text_id"], sort=False):
        g = g.sort_values("word_idx")
        s = scorer.score(g["word"].tolist())
        s.insert(0, "word_idx", g["word_idx"].values)
        s.insert(0, "text_id", tid)
        s.insert(0, "corpus", corpus)
        res.append(s)
    out = pd.concat(res, ignore_index=True)
    out["lm"] = lm_label
    return out
