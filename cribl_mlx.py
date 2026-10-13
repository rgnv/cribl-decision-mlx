"""Run Cribl Decision 1.0 with MLX.

First load: fold the LoRA into the base weights (fp32 math, stored as bf16)
and save the result under $CRIBL_DECISION_CACHE, default
~/.cache/cribl-decision. Every load after that just reads the folded copy.

Per call: one forward pass per question. The text before the state is run
through the model once and its cache is cloned for later calls, so only the
state tokens are processed each time.
"""
from __future__ import annotations

import json, os, shutil, time
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx

from cribl_decision import (ADAPTER_ID, ADAPTER_REVISION, BASE_ID, BASE_REVISION, FORMATS,
                            Answer, PromptFormat, Question, make_answer)

CACHE = Path(os.environ.get("CRIBL_DECISION_CACHE",
                            Path.home() / ".cache" / "cribl-decision"))


def merged_checkpoint() -> Path:
    out = CACHE / f"merged-{ADAPTER_REVISION[:12]}-{BASE_REVISION[:12]}"
    if (out / "DONE").exists():
        return out
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(BASE_ID, revision=BASE_REVISION))
    adir = Path(snapshot_download(ADAPTER_ID, revision=ADAPTER_REVISION))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora: dict[str, dict[str, mx.array]] = {}
    for k, t in mx.load(str(adir / "adapter_model.safetensors")).items():
        # adapter key "base_model.model.X.lora_A.weight" patches base key "X.weight"
        mod, which = k[len("base_model.model."):].rsplit(".lora_", 1)
        lora.setdefault(mod + ".weight", {})[which.split(".")[0]] = t.astype(mx.float32)
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = mx.load(str(base / shard))
        for k, t in tensors.items():
            if k in lora:
                ab = lora[k]
                tensors[k] = (t.astype(mx.float32) + scale * (ab["B"] @ ab["A"])).astype(t.dtype)
                merged += 1
        mx.save_safetensors(str(out / shard), tensors, metadata={"format": "pt"})
        del tensors
    if merged != len(lora):
        shutil.rmtree(out)
        raise RuntimeError(f"merged {merged} of {len(lora)} LoRA modules")
    for name in os.listdir(base):
        if not name.endswith(".safetensors") and not (out / name).exists():
            shutil.copy(base / name, out / name)
    (out / "MERGE.json").write_text(json.dumps({
        "base": BASE_ID, "base_revision": BASE_REVISION, "adapter": ADAPTER_ID,
        "adapter_revision": ADAPTER_REVISION, "scale": scale, "modules_merged": merged,
        "method": "float32 W + scale * B @ A, cast to bfloat16"}, indent=1))
    (out / "DONE").write_text("ok\n")
    return out


class CriblDecisionModelMLX:
    def __init__(self, fmt: str | PromptFormat = "plain_letter_qfirst", temperature: float = 1.0,
                 prefix_cache: bool = True):
        from mlx_lm import load
        from transformers import AutoTokenizer
        t0 = time.monotonic()
        path = merged_checkpoint()
        self.model, _ = load(str(path))
        # Load the tokenizer from the folded dir: identical files, no network.
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
        self.temperature = temperature
        self.prefix_cache = prefix_cache
        self.merge_report = json.loads((path / "MERGE.json").read_text())
        self.load_seconds = time.monotonic() - t0
        self._row_cache: dict = {}
        self._prefix: dict = {}       # prefix text -> (ids, cache snapshot)

    def build(self, state: Any, q: Question):
        prompt = self.fmt.prompt(self.tokenizer, state, q)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        keys = [k for k, _ in q.candidates()]
        # Candidate ids only change with the prompt ending and the candidate keys.
        key = (self.fmt.name, prompt[-16:], q.type, tuple(keys))
        rows = self._row_cache.get(key)
        if rows is None:
            rows = self._row_cache[key] = self.fmt.candidate_rows(self.tokenizer, prompt, q)
        return ids, rows, keys

    def ask(self, state: Any, questions: Mapping[str, Question]) -> tuple[dict[str, Answer], dict]:
        """Answer every question about one state. All prompts are built first,
        so an invalid question fails before any model time is spent."""
        t0 = time.monotonic()
        built = {name: self.build(state, q) for name, q in questions.items()}
        answers, cached, fwd = {}, 0, 0.0
        tokens = sum(len(b[0]) for b in built.values())
        for name, (ids, rows, keys) in built.items():
            q = questions[name]
            t1 = time.monotonic()
            cache, start = self._cached_prefix(state, q, ids) if self.prefix_cache else (None, 0)
            cached += start
            logits = self.model(mx.array([ids[start:]]), cache=cache)[0, -1].astype(mx.float32)
            logz = mx.logsumexp(logits)
            if all(len(r) == 1 for r in rows):
                cl = logits[mx.array([r[0] for r in rows])]
                mass = mx.exp(mx.logsumexp(cl) - logz)
            else:   # name_mean: lm_head has no bias, so mean of rows == mean of logits
                cl = mx.stack([logits[mx.array(r)].mean() for r in rows])
                mass = mx.array(float("nan"))
            mx.eval(cl, mass)
            fwd += time.monotonic() - t1
            answers[name] = make_answer(q, keys, cl.tolist(), float(mass.item()), self.temperature)
        t2 = time.monotonic()
        return answers, {"prep_ms": (t2 - t0 - fwd) * 1000, "forward_ms": fwd * 1000,
                         "total_ms": (t2 - t0) * 1000, "tokens": tokens,
                         "cached_tokens": cached, "questions": len(questions)}

    def _cached_prefix(self, state: Any, q: Question, ids: list[int]):
        """Return (cache, n) where the cache already holds the first n prompt
        tokens, or (None, 0) when the layout has no cacheable prefix."""
        from mlx_lm.models.cache import KVCache
        prefix = self.fmt.static_prefix(state, q)
        if not prefix:
            return None, 0
        hit = self._prefix.get(prefix)
        if hit is None:
            pids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            cache = self.model.make_cache()
            mx.eval(self.model(mx.array([pids]), cache=cache))
            snap = []
            for c in cache:
                if isinstance(c, KVCache):
                    # trim to `offset` so the next write allocates fresh memory
                    # rather than overwriting the stored copy
                    k, v = c.keys[..., :c.offset, :], c.values[..., :c.offset, :]
                    mx.eval(k, v)
                    snap.append(("kv", k, v, c.offset))
                else:
                    # recurrent state: copy the list, layers replace its entries
                    mx.eval(*[a for a in c.cache if a is not None])
                    snap.append(("arr", list(c.cache)))
            hit = self._prefix[prefix] = (pids, snap)
        pids, snap = hit
        if ids[:len(pids)] != pids:          # tokenisation differs across the cut
            return None, 0
        cache = self.model.make_cache()
        for c, s in zip(cache, snap):
            if s[0] == "kv":
                c.keys, c.values, c.offset = s[1], s[2], s[3]
            else:
                c.cache = list(s[1])
        return cache, len(pids)

    def choose(self, state: Any, q: Question) -> tuple[Answer, dict]:
        a, t = self.ask(state, {"q": q})
        return a["q"], t
