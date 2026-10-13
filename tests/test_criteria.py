import pytest

from locassess.criteria import CriteriaError, load_criteria_csv
from locassess.db import repo
from tests.conftest import CRITERIA_CSV

HEADER = ("factor_en,factor_de,main_id,main_criterion_en,main_criterion_de,sub_id,"
          "sub_criterion_en,sub_criterion_de,paper_sources,paper_table\n")


def test_counts():
    crit = load_criteria_csv(CRITERIA_CSV)
    assert len(crit) == 26
    assert sum(len(m.sub_criteria) for m in crit) == 94
    assert [m.main_id for m in crit] == [f"H{i:02d}" for i in range(1, 27)]
    assert len({m.factor for m in crit}) == 6


def test_criteria_without_sub_criteria():
    crit = {m.main_id: m for m in load_criteria_csv(CRITERIA_CSV)}
    assert {k for k, m in crit.items() if not m.sub_criteria} == {"H05", "H06", "H07", "H08"}


def test_english_names_used():
    crit = {m.main_id: m for m in load_criteria_csv(CRITERIA_CSV)}
    assert crit["H02"].name == "Transportation system"
    assert crit["H20"].name == "Population potential"


def test_upsert_idempotent(conn):
    crit = load_criteria_csv(CRITERIA_CSV)
    repo.upsert_criteria(conn, crit)
    repo.upsert_criteria(conn, crit)
    assert repo.count_criteria(conn) == (26, 94)


@pytest.mark.parametrize("row", [
    "Physical factors,,X1,Name,,,,,,\n",                         # bad main_id
    "Physical factors,,H01,Name,,H02.01,Sub,,,\n",               # sub belongs to other main
    "Physical factors,,H01,,,,,,,\n",                            # missing name
])
def test_malformed_rows_rejected(tmp_path, row):
    p = tmp_path / "c.csv"
    p.write_text(HEADER + row, encoding="utf-8")
    with pytest.raises(CriteriaError):
        load_criteria_csv(p, expected_main=None, expected_sub=None)


def test_wrong_count_rejected(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(HEADER + "Physical factors,,H01,Name,,H01.01,Sub,,,\n", encoding="utf-8")
    with pytest.raises(CriteriaError, match="expected 26"):
        load_criteria_csv(p)
