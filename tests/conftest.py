from pathlib import Path

import pytest

from locassess.db import repo

ROOT = Path(__file__).resolve().parents[1]
CRITERIA_CSV = ROOT / "data" / "criteria.csv"
EXAMPLE_CONFIG = ROOT / "config.example.yaml"


@pytest.fixture
def conn(tmp_path):
    c = repo.connect(tmp_path / "test.db")
    repo.init_db(c)
    yield c
    c.close()
