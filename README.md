# cribl-decision-mlx

Local inference for [`cribl-ai/cribl-decision-1.0`](https://huggingface.co/cribl-ai/cribl-decision-1.0) on a Mac.

You give it a state and a question. It returns a probability for each answer. The model never writes text: it runs one forward pass and reads the scores for each answer's token.

```
state + question ──► prompt ──► Qwen3.5-4B + LoRA (MLX) ──► logits for A/B/C… ──► softmax ──► answer
```

## Setup

You need Apple Silicon, Python 3.11–3.13, about 20 GB of free disk and 16 GB of RAM or more.

```bash
git clone https://github.com/rgnv/cribl-decision-mlx
cd cribl-decision-mlx
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[test]"
```

The first start takes a few minutes. It downloads `Qwen/Qwen3.5-4B-Base` (9.3 GB) and the adapter (58 MB), merges them, and writes the result to `~/.cache/cribl-decision` (another 9.3 GB). Set `CRIBL_DECISION_CACHE` to put it somewhere else. Later starts load the merged copy directly.

## Run it

**As a server**

```bash
.venv/bin/python serve.py            # http://127.0.0.1:8077
```

```bash
curl -s localhost:8077/v1/decide -d '{
  "state": {"host": "web-01", "disk_used_pct": 97, "mount": "/var"},
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Pick the correct action for this disk reading.",
      "criteria": {
        "none":   "Do nothing. Correct only below 80%.",
        "ticket": "Open a ticket. Correct between 80% and 95%.",
        "page":   "Page on-call. Correct above 95%."
      }
    },
    "urgent": {"type": "noul", "instructions": "Does this need a human in the next hour?"}
  }
}'
```

The response is `{"answers": {name: answer}, "timing": {...}}`. Each answer has the fields listed under [Question types](#question-types). `timing` gives `total_ms`, `tokens` and `cached_tokens`.

| endpoint | |
|---|---|
| `POST /v1/decide` | `{"state": object or string, "questions": {name: question}}`. Bad input returns 400. |
| `GET /healthz` | prompt format in use and model load time |

Options: `--host`, `--port`, `--format`. The server handles one request at a time.

**From Python**

```python
from cribl_decision import Question
from cribl_mlx import CriblDecisionModelMLX

model = CriblDecisionModelMLX()
q = Question("choice", "Pick the correct action for this disk reading.",
             {"none": "below 80%", "ticket": "80-95%", "page": "above 95%"})
answer, timing = model.choose({"disk_used_pct": 97}, q)
answer.choice, answer.probabilities
```

`model.ask(state, {name: Question, ...})` answers several questions about the same state.

## Question types

| `type` | `criteria` | read the answer from |
|---|---|---|
| `choice` | `{key: description}`, up to 26 keys | `choice` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`, P(true) |
| `score` | `[lowest, ..., highest]`, 2–10 levels | `score`, the expected level index |

Every answer also has `probabilities`, `logits`, `p_selected` and `confidence` (0 = no preference, 1 = certain). `candidate_mass` is the share of the model's total next-token probability that went to the listed answers. Values near 1 mean the prompt is working. A low value means the model wanted to say something else.

## Why it's fast

- **Merged weights.** The LoRA is merged into the base once and saved, so no adapter code runs on each request. The merge runs in MLX, so PyTorch isn't installed at all. It matches a PyTorch/PEFT merge on all but 1,775 of 4.66 B weights, and those differ by one bf16 step.
- **Cached prompt head.** In the default `plain_letter_qfirst` layout, the question and its answers come before the state. That part runs through the model once per question shape, and each request only processes the state tokens. On a CPU test this cut a call from 284 s to about 90 s.
- **Cached token lookups.** The vocabulary ids for each answer are worked out once per question shape.
- **Fail before compute.** Every question in a request is validated before the first forward pass.

Other layouts (`plain_letter`, `chat_letter`, …) are available with `--format` or `fmt=`. Only `*_qfirst` layouts with the plain template use the cache.

## Tests

```bash
.venv/bin/python -m pytest -q                        # no model needed, a few seconds
CRIBL_MODEL_TESTS=1 .venv/bin/python -m pytest -q    # loads the model
```

The model tests check that every LoRA module was merged, that the model follows a stated threshold rule, and that cached and uncached calls give the same answers.

## Layout

| file | |
|---|---|
| `cribl_decision.py` | questions, answers, prompt layouts, scoring. Pure Python. |
| `cribl_mlx.py` | weight merge, model loading, inference, prompt cache |
| `serve.py` | HTTP server (stdlib only) |

## License

MIT for this code. The adapter (Apache-2.0) and the Qwen base model (its own license) are downloaded at runtime and are not part of this repo.
