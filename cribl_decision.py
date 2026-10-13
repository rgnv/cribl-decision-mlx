"""Prompt building and scoring for Cribl Decision 1.0. No ML imports.

A question is rendered as a prompt whose last token is where the model would
write its answer. Each candidate maps to one or more vocabulary ids; their
logits at that position, softmaxed against each other, are the answer.
The exact prompt layout is a `PromptFormat`, so layouts can be swapped.
"""
from __future__ import annotations

import json, math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

BASE_ID = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"   # pinned by the model card
ADAPTER_ID = "cribl-ai/cribl-decision-1.0"
ADAPTER_REVISION = "75cb0dd9d551e774664c01ba8f7b25c0087b4abf"  # HF main on 2026-10-10

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Question:
    """`criteria` by type: choice -> {name: description}; score -> [level, ...]
    lowest first; noul -> optional {"true": desc, "false": desc}."""
    type: str
    instructions: str
    criteria: Any = None

    def candidates(self) -> list[tuple[str, str | None]]:
        if self.type == "choice":
            if not isinstance(self.criteria, Mapping) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("choice needs 1..255 named candidates")
            return [(str(k), None if v is None else str(v)) for k, v in self.criteria.items()]
        if self.type == "score":
            if not isinstance(self.criteria, Sequence) or isinstance(self.criteria, str) \
                    or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score needs 2..10 ordered levels")
            return [(str(i), str(v)) for i, v in enumerate(self.criteria)]
        if self.type == "noul":
            c = self.criteria or {}
            return [("true", c.get("true")), ("false", c.get("false"))]
        raise ValueError(f"unknown question type {self.type!r}")


@dataclass
class Answer:
    type: str
    probabilities: dict[str, float]
    choice: str | None = None        # winning candidate key
    noul: float | None = None        # P(true), noul only
    score: float | None = None       # probability-weighted level, score only
    confidence: float = 0.0          # 0 = no preference, 1 = certain
    p_selected: float = 0.0          # probability of `choice`
    candidate_mass: float = 0.0      # how much of the whole-vocab softmax hit the candidates
    logits: list[float] = field(default_factory=list)


def softmax(xs: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(xs)
    es = [math.exp((x - m) / temperature) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def choice_confidence(probs: Sequence[float]) -> float:
    """Rescale the top probability so uniform -> 0 and one-hot -> 1."""
    n = len(probs)
    if n == 1:
        return 1.0
    u = 1.0 / n
    return (max(probs) - u) / (1.0 - u)


def score_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n == 1:
        return 1.0
    mode = max(range(n), key=probs.__getitem__)
    dist = sum(p * abs(i - mode) for i, p in enumerate(probs))
    centre = (n - 1) / 2
    uniform_dev = sum(abs(i - centre) for i in range(n)) / n
    return max(0.0, 1.0 - dist / uniform_dev)


def make_answer(q: Question, keys: list[str], logits: list[float], mass: float,
                temperature: float = 1.0) -> Answer:
    p = softmax(logits, temperature)
    probs = dict(zip(keys, p))
    best = max(range(len(p)), key=p.__getitem__)
    a = Answer(type=q.type, probabilities=probs, choice=keys[best], p_selected=p[best],
               candidate_mass=mass, logits=list(logits))
    if q.type == "score":
        a.score = sum(i * x for i, x in enumerate(p))
        a.confidence = score_confidence(p)
    else:
        if q.type == "noul":
            a.noul = probs["true"]
        a.confidence = choice_confidence(p)
    return a


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(", ", ": "))


@dataclass(frozen=True)
class PromptFormat:
    """A prompt layout.

    template     "plain" ends with "Answer:"; "chat" uses the Qwen chat template
                 (thinking off) and answers at the first assistant token.
    labels       "letter" scores candidates by A/B/C...; "name" by the name's
                 first token (must be unique); "name_mean" by the mean logit of
                 all of the name's tokens.
    state_first  False puts the state at the end, which makes everything before
                 it reusable across calls (see static_prefix).
    """
    name: str
    template: str = "chat"
    labels: str = "letter"
    state_first: bool = True

    def body(self, state: Any, q: Question) -> str:
        cands = q.candidates()
        if self.labels == "letter" and len(cands) > len(LETTERS):
            raise ValueError("letter labels support at most 26 candidates")
        if q.type == "noul":
            yes, no = (("A. Yes", "B. No") if self.labels == "letter" else ("Yes", "No"))
            lines = [yes + (f": {cands[0][1]}" if cands[0][1] else ""),
                     no + (f": {cands[1][1]}" if cands[1][1] else "")]
        else:
            lines = []
            for i, (k, d) in enumerate(cands):
                if self.labels == "letter":
                    lines.append(f"{LETTERS[i]}. {k}: {d}" if d and q.type == "choice" else
                                 f"{LETTERS[i]}. {d if d else k}")
                else:
                    lines.append(f"{k}: {d}" if d else k)
        what = "letter" if self.labels == "letter" else "candidate"
        parts_state = f"State:\n{render_state(state)}"
        parts_q = (f"Question ({q.type}): {q.instructions}\n\n"
                   f"Candidates:\n" + "\n".join(lines) +
                   f"\n\nAnswer with the {what} of the single best candidate.")
        return (parts_state + "\n\n" + parts_q) if self.state_first else \
               (parts_q + "\n\n" + parts_state)

    def static_prefix(self, state: Any, q: Question) -> str | None:
        """Text before the state, if the layout allows caching it (plain, state last)."""
        if self.state_first or self.template != "plain":
            return None
        body = self.body(state, q)
        return body[:body.index("State:\n")]

    def prompt(self, tokenizer, state: Any, q: Question) -> str:
        body = self.body(state, q)
        if self.template == "chat":
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        return body + "\nAnswer:"

    def candidate_surface(self, i: int, key: str, q: Question) -> str:
        """What the answer for candidate i looks like right after the prompt."""
        if q.type == "noul":
            word = ("A" if i == 0 else "B") if self.labels == "letter" else ("Yes" if i == 0 else "No")
        else:
            word = LETTERS[i] if self.labels == "letter" else key
        # "Answer:" is followed by a space; the chat template ends in a newline.
        return (" " + word) if self.template == "plain" else word

    def candidate_rows(self, tokenizer, prompt: str, q: Question) -> list[list[int]]:
        """Vocabulary ids per candidate: tokenize prompt and prompt+answer, keep
        the extra ids. Raises if an answer would merge into the prompt's tail."""
        base = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        rows = []
        for i, (k, _) in enumerate(q.candidates()):
            full = tokenizer(prompt + self.candidate_surface(i, k, q),
                             add_special_tokens=False)["input_ids"]
            if full[:len(base)] != base:
                raise ValueError(f"candidate {k!r} merges with the prompt tail")
            ids = full[len(base):]
            if not ids:
                raise ValueError(f"candidate {k!r} has no tokens")
            rows.append(ids if self.labels == "name_mean" else ids[:1])
        if self.labels != "name_mean":
            firsts = [r[0] for r in rows]
            if len(set(firsts)) != len(firsts):
                raise ValueError(f"format {self.name}: candidates share a scoring "
                                 f"token ({firsts}); use letter labels")
        return rows


FORMATS: dict[str, PromptFormat] = {f.name: f for f in [
    PromptFormat("plain_letter_qfirst", "plain", "letter", state_first=False),  # default: cacheable
    PromptFormat("plain_letter", "plain", "letter"),
    PromptFormat("chat_letter", "chat", "letter"),
    PromptFormat("chat_letter_qfirst", "chat", "letter", state_first=False),
    PromptFormat("plain_name", "plain", "name"),
    PromptFormat("chat_name", "chat", "name"),
    PromptFormat("chat_name_mean", "chat", "name_mean"),
]}
