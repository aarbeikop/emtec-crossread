"""Mechanical checks of LMScorer with a tiny random GPT-2 (no downloads needed).

1. every token is assigned to exactly one word
2. without BOW correction, word surprisal == sum of token surprisals from a plain forward pass
3. BOW correction telescopes: sum over words changes only by the first/last boundary terms
4. sliding window (tiny context) runs and leaves no NaNs
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from crossread.surprisal import LMScorer  # noqa: E402

CORPUS = ["The cat sat on the mat, and the dog barked loudly.",
          "Language models assign probabilities to tokens; readers fixate words.",
          "Unbelievably, the reindeer's antlers weren't symmetrical!"] * 50
WORDS = "Unbelievably, the reindeer's antlers weren't symmetrical! The cat sat on the mat.".split()


def build(tmp):
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, eos_token="<|endoftext|>",
                                  bos_token="<|endoftext|>")
    tok.save_pretrained(tmp)
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    GPT2LMHeadModel(cfg).save_pretrained(tmp)
    return tok


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tok = build(tmp)
        raw = LMScorer(tmp, bow_correction=False)
        cor = LMScorer(tmp, bow_correction=True)
        assert raw.space_prefix and cor.bow_correction

        r = raw.score(WORDS)
        ids = tok(" " + " ".join(WORDS)).input_ids
        assert r.n_tokens.sum() == len(ids), (r.n_tokens.sum(), len(ids))  # check 1

        seq = torch.tensor([[tok.eos_token_id] + ids])
        with torch.no_grad():
            lp = torch.log_softmax(raw.model(seq).logits[0], -1)
        tok_surp = -lp[torch.arange(len(ids)), seq[0, 1:]].numpy() / math.log(2)
        assert np.isclose(r.surprisal.sum(), tok_surp.sum(), atol=1e-3)  # check 2

        c = cor.score(WORDS)
        bow = torch.logsumexp(lp[:, cor.bow_mask], -1).numpy() / math.log(2)
        expected = r.surprisal.sum() - bow[len(ids)] + bow[0]
        assert np.isclose(c.surprisal.sum(), expected, atol=1e-3)  # check 3
        assert not np.allclose(c.surprisal, r.surprisal)

        # check 5: temperature columns match a manual softmax(logits / T); T=1 unchanged
        rt = LMScorer(tmp, bow_correction=False, temps=[2.0]).score(WORDS)
        assert np.allclose(rt.surprisal, r.surprisal, atol=1e-4)
        with torch.no_grad():
            lp2 = torch.log_softmax(raw.model(seq).logits[0] / 2.0, -1)
        tok2 = -lp2[torch.arange(len(ids)), seq[0, 1:]].numpy() / math.log(2)
        assert np.isclose(rt["surprisal_T2"].sum(), tok2.sum(), atol=1e-3)
        assert not np.allclose(rt["surprisal_T2"], rt.surprisal)

        small = LMScorer(tmp, max_ctx=8).score(WORDS * 5)  # check 4
        assert small.surprisal.notna().all() and len(small) == len(WORDS) * 5

        print(c.assign(word=WORDS, raw=r.surprisal).round(2).to_string())
        print("ok")


if __name__ == "__main__":
    main()
