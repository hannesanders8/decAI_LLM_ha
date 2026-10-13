from datetime import date

import pytest
from pydantic import ValidationError

from locassess.config import Secrets, config_hash, load_config
from tests.conftest import EXAMPLE_CONFIG


def test_example_config_loads():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.location.name == "Leipzig"
    assert cfg.corpus.target_articles == 15
    assert cfg.scoring.type_weights == {"fact": 1.0, "announcement": 0.6, "opinion": 0.3}
    assert cfg.scoring.extraction.review_threshold == 0.6
    assert cfg.models.decision_provider == "llm_structured"
    assert cfg.period.run_date is None
    assert cfg.resolved_run_date() == date.today()


def test_fixed_run_date():
    cfg = load_config(EXAMPLE_CONFIG)
    cfg.period.run_date = date(2026, 10, 1)
    assert cfg.resolved_run_date() == date(2026, 10, 1)


def test_config_hash_stable_and_sensitive():
    a, b = load_config(EXAMPLE_CONFIG), load_config(EXAMPLE_CONFIG)
    assert config_hash(a) == config_hash(b)
    b.scoring.half_life_months = 12
    assert config_hash(a) != config_hash(b)


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8") + "typo_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(p)


def test_secrets_not_in_repr(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-very-secret")
    s = Secrets()
    assert "sk-test-very-secret" not in repr(s)
    assert "sk-test-very-secret" not in str(s.model_dump())
    assert s.values() == ["sk-test-very-secret"]
