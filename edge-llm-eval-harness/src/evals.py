"""A small, self-contained eval suite.

Keeps external dependencies minimal so the harness runs without pulling
the full lm-evaluation-harness stack. Each eval returns a dict of
metrics that can be aggregated across backends.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class MCQExample:
    question: str
    choices: list[str]
    answer_idx: int


def score_mcq_by_logprob(backend, examples: list[MCQExample]) -> dict:
    """Classic lm-eval-harness style: score each choice by full-sequence logprob."""
    correct = 0
    for ex in examples:
        prompt = f"Question: {ex.question}\nAnswer:"
        lps = [backend.logprob_of_continuation(prompt, " " + c) for c in ex.choices]
        if max(range(len(lps)), key=lps.__getitem__) == ex.answer_idx:
            correct += 1
    return {"n": len(examples), "correct": correct, "acc": correct / max(1, len(examples))}


TOY_MCQ = [
    MCQExample("What is 2 + 2?", ["3", "4", "5", "6"], 1),
    MCQExample("The capital of France is", ["Berlin", "Madrid", "Paris", "Rome"], 2),
    MCQExample("Water freezes at", ["0 C", "10 C", "50 C", "100 C"], 0),
    MCQExample("The largest planet is", ["Mars", "Earth", "Jupiter", "Venus"], 2),
]


def multi_step_arithmetic(n: int = 20, seed: int = 0):
    """Generative probe for multi-step reasoning degradation under quantization."""
    import random
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 20)
        problems.append({"prompt": f"Compute step by step: ({a} + {b}) * {c} =",
                         "answer": (a + b) * c})
    return problems


def score_generative_arith(backend, problems) -> dict:
    import re
    correct = 0
    for p in problems:
        out = backend.generate(p["prompt"], max_new_tokens=64)
        nums = re.findall(r"-?\d+", out)
        if nums and int(nums[-1]) == p["answer"]:
            correct += 1
    return {"n": len(problems), "correct": correct, "acc": correct / max(1, len(problems))}


EVAL_REGISTRY: dict[str, Callable] = {
    "toy_mcq": lambda b: score_mcq_by_logprob(b, TOY_MCQ),
    "multi_step_arith": lambda b: score_generative_arith(b, multi_step_arithmetic()),
}
