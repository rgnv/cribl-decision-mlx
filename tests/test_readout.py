"""Default run: tokenizer-only checks. CRIBL_MODEL_TESTS=1 adds tests that
load the full model."""
from __future__ import annotations

import math, os

import pytest

import cribl_decision as cd

needs_model = pytest.mark.skipif(os.environ.get("CRIBL_MODEL_TESTS") != "1",
                                 reason="set CRIBL_MODEL_TESTS=1 to load the 4B model")

# Known-answer probe: the correct action follows from a stated threshold rule.
DISK_Q = cd.Question(
    "choice",
    "You are the on-call monitor for a fleet of servers. "
    "Pick the single correct action for this disk reading right now.",
    {"ALERT_NONE": "Do nothing. Correct only when disk_used_pct is BELOW 80.",
     "ALERT_TICKET": "Open a ticket. Correct only when disk_used_pct is between 80 and 95.",
     "ALERT_PAGE": "Page on-call now. Correct only when disk_used_pct is ABOVE 95."})


def disk_state(pct, host="web-01"):
    return {"host": host, "disk_used_pct": pct, "mount": "/var"}


# --- pure maths -------------------------------------------------------------

def test_softmax_normalises_and_is_shift_invariant():
    p = cd.softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(p), 1.0)
    assert p == pytest.approx(cd.softmax([101.0, 102.0, 103.0]))
    assert p[2] > p[1] > p[0]


def test_choice_confidence_matches_system_one_formula():
    assert cd.choice_confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert cd.choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert cd.choice_confidence([0.5, 0.25, 0.25]) == pytest.approx((0.5 - 1 / 3) / (2 / 3))


def test_make_answer_types():
    a = cd.make_answer(DISK_Q, ["ALERT_NONE", "ALERT_TICKET", "ALERT_PAGE"], [0.0, 2.0, 1.0], 0.9)
    assert a.choice == "ALERT_TICKET" and a.p_selected == max(a.probabilities.values())
    n = cd.make_answer(cd.Question("noul", "x?"), ["true", "false"], [2.0, 0.0], 1.0)
    assert n.noul == pytest.approx(1 / (1 + math.exp(-2)))
    s = cd.make_answer(cd.Question("score", "x?", ["lo", "mid", "hi"]), ["0", "1", "2"],
                       [0.0, 0.0, 50.0], 1.0)
    assert s.score == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("q", [cd.Question("choice", "x", {}), cd.Question("score", "x", ["one"]),
                               cd.Question("score", "x", "abc"), cd.Question("bogus", "x")])
def test_question_validation(q):
    with pytest.raises(ValueError):
        q.candidates()


# --- prompt formats (tokenizer only) ----------------------------------------

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cd.BASE_ID, revision=cd.BASE_REVISION)


@pytest.mark.parametrize("fmt", ["plain_letter_qfirst", "plain_letter", "chat_letter"])
def test_letter_rows_are_single_distinct_tokens(tok, fmt):
    f = cd.FORMATS[fmt]
    rows = f.candidate_rows(tok, f.prompt(tok, disk_state(50), DISK_Q), DISK_Q)
    want = [" A", " B", " C"] if f.template == "plain" else ["A", "B", "C"]
    assert [tok.decode(r) for r in rows] == want


def test_static_prefix_is_state_independent_and_tokenises_as_prompt_head(tok):
    f = cd.FORMATS["plain_letter_qfirst"]
    pre = f.static_prefix(disk_state(50), DISK_Q)
    assert pre == f.static_prefix(disk_state(99, "db-07"), DISK_Q)
    prompt = f.prompt(tok, disk_state(50), DISK_Q)
    assert prompt.startswith(pre) and prompt.endswith("\nAnswer:")
    pids = tok(pre, add_special_tokens=False)["input_ids"]
    assert tok(prompt, add_special_tokens=False)["input_ids"][:len(pids)] == pids
    assert cd.FORMATS["plain_letter"].static_prefix(disk_state(50), DISK_Q) is None


def test_name_labels_reject_shared_first_token(tok):
    """ALERT_NONE/ALERT_TICKET/ALERT_PAGE share the first token; must fail loudly."""
    f = cd.FORMATS["plain_name"]
    with pytest.raises(ValueError, match="share a scoring token"):
        f.candidate_rows(tok, f.prompt(tok, disk_state(50), DISK_Q), DISK_Q)


# --- model (MLX) ------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    from cribl_mlx import CriblDecisionModelMLX
    return CriblDecisionModelMLX()


@needs_model
def test_every_lora_module_merged(model):
    assert model.merge_report["modules_merged"] == 152      # 304 adapter tensors = A + B


@needs_model
@pytest.mark.parametrize("pct,want", [(20, "ALERT_NONE"), (99, "ALERT_PAGE")])
def test_follows_threshold_rule(model, pct, want):
    a, _ = model.choose(disk_state(pct), DISK_Q)
    assert a.choice == want and a.p_selected > 0.5 and a.candidate_mass > 0.8


@needs_model
def test_prefix_cache_is_stable_and_matches_uncached(model):
    """Cached calls repeat exactly, and agree with uncached calls."""
    pcts = [50, 99]
    model.prefix_cache = False
    full = [model.choose(disk_state(p), DISK_Q)[0].probabilities for p in pcts]
    model.prefix_cache = True
    first = [model.choose(disk_state(p), DISK_Q)[0].probabilities for p in pcts]
    again = [model.choose(disk_state(p), DISK_Q)[0].probabilities for p in pcts]
    assert first == again
    for f, c in zip(full, first):
        for k in f:
            assert abs(f[k] - c[k]) < 0.05
