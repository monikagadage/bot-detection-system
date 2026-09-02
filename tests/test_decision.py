"""The policy layer — scores in, ALLOW / CHALLENGE / BLOCK out."""

from __future__ import annotations

from dataclasses import dataclass

from botshield.decision import ALLOW, BLOCK, BLOCK_AT, CHALLENGE, CHALLENGE_AT, decide
from botshield.features import FEATURE_NAMES
from botshield.store import Store


class _StubModel:
    def __init__(self, proba):
        self._p = proba

    def predict_proba(self, _vector):
        return self._p


@dataclass
class _Ev:
    ip: str = "1.2.3.4"
    ts: float = 1.0


def _decide(hard_reason=None, rule_score=0.0, rule_hits=None, model_proba=0.0):
    features = {name: 0.0 for name in FEATURE_NAMES}
    return decide(_Ev(), features, hard_reason, rule_score, rule_hits or [],
                  _StubModel(model_proba), Store())


def test_hard_rule_blocks_regardless_of_model():
    d = _decide(hard_reason="honeypot filled", model_proba=0.0)
    assert d.action == BLOCK
    assert d.score == 1.0


def test_low_blended_score_allows():
    d = _decide(rule_score=0.0, model_proba=0.1)
    assert d.action == ALLOW


def test_mid_blended_score_challenges_and_issues_a_token():
    # blend = 0.4*rule + 0.6*model; land it between CHALLENGE_AT and BLOCK_AT.
    d = _decide(rule_score=0.5, model_proba=0.6)
    assert CHALLENGE_AT <= d.score < BLOCK_AT
    assert d.action == CHALLENGE
    assert d.challenge_token is not None


def test_high_blended_score_blocks():
    d = _decide(rule_score=0.9, model_proba=0.95)
    assert d.score >= BLOCK_AT
    assert d.action == BLOCK


def test_reasons_are_populated():
    d = _decide(rule_score=0.2, model_proba=0.2)
    assert any("model p(bot)" in r for r in d.reasons)
