"""A small, self-contained eval suite.

Designed for sensitivity to quantization, not absolute capability:
every eval should produce a measurable number on a 0.5B base model
that moves as the model is quantized more aggressively.

  toy_mcq          — sanity check; trivial world knowledge
  arith_mcq        — multiple-choice arithmetic (tractable without CoT)
  perplexity       — exact next-token log-likelihood on a fixed text,
                     the single most sensitive signal we have
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class MCQExample:
    question: str
    choices: list[str]
    answer_idx: int


def score_mcq_by_logprob(backend, examples: list[MCQExample]) -> dict:
    correct = 0
    margins = []
    for ex in examples:
        prompt = f"Question: {ex.question}\nAnswer:"
        lps = [backend.logprob_of_continuation(prompt, " " + c) for c in ex.choices]
        top = max(range(len(lps)), key=lps.__getitem__)
        if top == ex.answer_idx:
            correct += 1
        sorted_lps = sorted(lps, reverse=True)
        margins.append(sorted_lps[0] - sorted_lps[1])
    return {
        "n": len(examples),
        "correct": correct,
        "acc": correct / max(1, len(examples)),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


TOY_MCQ = [
    MCQExample("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    MCQExample("The capital of France is", ["Berlin", "Madrid", "Paris", "Rome"], 2),
    MCQExample("Water freezes at", ["0 C", "10 C", "50 C", "100 C"], 0),
    MCQExample("The largest planet is", ["Mars", "Earth", "Jupiter", "Venus"], 2),
]


def arith_mcq_examples(n: int = 30, seed: int = 0) -> list[MCQExample]:
    """Multiple-choice `(a + b) * c` — tractable via logprob scoring.

    Distractors are near-misses (operator-precedence confusions, off-by-one
    on each operand). Candidates are generated then filtered for uniqueness.
    """
    import random
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(2, 9)
        ans = (a + b) * c
        candidates = [
            (a + b + 1) * c,
            (a + b - 1) * c,
            a + b * c,
            a * c + b,
            (a + b) * (c + 1),
            (a + b) * (c - 1),
            ans + 1,
            ans - 1,
        ]
        seen = {ans}
        uniq = []
        for x in candidates:
            if x not in seen and x > 0:
                seen.add(x)
                uniq.append(x)
        wrong = rng.sample(uniq, 3)
        choices = [str(x) for x in ([ans] + wrong)]
        rng.shuffle(choices)
        idx = choices.index(str(ans))
        examples.append(MCQExample(f"What is ({a} + {b}) * {c}?", choices, idx))
    return examples


# Fixed snippet for perplexity. English, nontrivial enough that a 0.5B
# model produces a non-trivial loss. Kept short so eval is cheap.
PERPLEXITY_TEXT = (
    "The development of general artificial intelligence has become a central question "
    "of twenty-first century engineering. Researchers have begun to ask not only whether "
    "such systems can be built, but how they should be evaluated, deployed, and constrained. "
    "Progress in large language models has outpaced progress in tools for understanding "
    "them, and this gap is arguably the most important unsolved problem in the field. "
    "Mechanistic interpretability, automated evaluation, and hardware-aware profiling all "
    "aim, in different ways, at the same underlying goal: knowing what a model will do "
    "before it is run at scale."
)


def score_perplexity(backend, text: str = PERPLEXITY_TEXT) -> dict:
    """Sum logprob over all but the first token, then convert to perplexity.

    Uses the backend's own tokenizer via its logprob_of_continuation API.
    """
    words = text.split()
    prompt = words[0]
    continuation = " " + " ".join(words[1:])
    lp = backend.logprob_of_continuation(prompt, continuation)
    n_tokens = max(1, len(text.split()) - 1)
    ppl = math.exp(-lp / n_tokens)
    return {"n_tokens_approx": n_tokens, "logprob": lp, "perplexity_per_word": ppl}


EVAL_REGISTRY: dict[str, Callable] = {
    "toy_mcq": lambda b: score_mcq_by_logprob(b, TOY_MCQ),
    "arith_mcq": lambda b: score_mcq_by_logprob(b, arith_mcq_examples()),
    "perplexity": lambda b: score_perplexity(b),
}
