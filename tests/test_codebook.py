"""SPEC §5.2 codebook: draft generation, validation, use in Stage 0."""

from datetime import date

import pytest
import yaml

from locassess.codebook_draft import write_codebook_draft
from locassess.config import load_config
from locassess.criteria import CriteriaError, load_codebook, load_criteria_csv
from locassess.stage0_corpus import criterion_search_info, generate_queries
from tests.conftest import CRITERIA_CSV, EXAMPLE_CONFIG
from tests.fakes import FakeLLM

CRITERIA = load_criteria_csv(CRITERIA_CSV)


@pytest.fixture
def draft(tmp_path):
    path = tmp_path / "codebook.yaml"
    llm = FakeLLM()
    generated = write_codebook_draft(path, llm, CRITERIA, "gpt-test")
    return path, llm, generated


def test_draft_covers_all_criteria_and_validates(draft):
    path, llm, generated = draft
    assert len(generated) == 26 and sum(1 for p, _ in llm.calls if p == "codebook_draft") == 26
    cb = load_codebook(path, CRITERIA)
    assert cb.version == "0.1" and len(cb.criteria) == 26 and not cb.is_reviewed
    h02 = cb.criteria["H02"]
    assert h02.status == "draft" and h02.name == "Transportation system"
    assert h02.sub_criteria[0].startswith("H02.01 ")
    assert h02.search_terms == ("H02 term a", "H02 term b")          # whitespace normalised
    assert h02.positive_examples == ("H02 good news.",)                # duplicates removed
    assert cb.criteria["H05"].sub_criteria == ()                       # no sub-criteria in the CSV
    assert set(cb.intensity_definition) == {1, 2, 3}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["draft_generated_with"]["model"] == "gpt-test"
    assert path.read_text(encoding="utf-8").startswith("# LocAssess codebook")


def test_existing_file_needs_force(draft):
    path, _, _ = draft
    with pytest.raises(FileExistsError):
        write_codebook_draft(path, FakeLLM(), CRITERIA, "gpt-test")


def test_force_keeps_reviewed_entries_and_author_edits(draft):
    path, _, _ = draft
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["criteria"]["H02"].update(status="reviewed", definition="Author's own definition.")
    raw["direction_definition"] = "Author's direction text."
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

    llm = FakeLLM()
    generated = write_codebook_draft(path, llm, CRITERIA, "gpt-test", force=True)
    assert "H02" not in generated and len(generated) == 25
    cb = load_codebook(path, CRITERIA)
    assert cb.criteria["H02"].definition == "Author's own definition." and cb.criteria["H02"].status == "reviewed"
    assert cb.direction_definition == "Author's direction text."
    assert cb.drafts() == [k for k in cb.criteria if k != "H02"]


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["criteria"].pop("H07"), "missing criteria"),
    (lambda r: r["criteria"]["H02"].update(name="Transport"), "differs from criteria.csv"),
    (lambda r: r["criteria"]["H02"]["sub_criteria"].pop(), "sub_criteria do not match"),
    (lambda r: r["criteria"]["H02"].update(status="final"), "status must be"),
    (lambda r: r["criteria"]["H02"].update(definition=""), "definition is empty"),
    (lambda r: r["criteria"]["H02"].update(search_terms=[]), "no search_terms"),
    (lambda r: r.pop("intensity_definition"), "intensity_definition"),
])
def test_validation_errors(draft, mutate, match):
    path, _, _ = draft
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(CriteriaError, match=match):
        load_codebook(path, CRITERIA)


def test_stage0_queries_use_codebook_terms(draft):
    path, _, _ = draft
    cb = load_codebook(path, CRITERIA)
    main = CRITERIA[1]
    assert criterion_search_info(main, cb) == ("Definition of H02.", ["H02 term a", "H02 term b"])
    assert criterion_search_info(main)[1][0] == "Railway"              # fallback without codebook
    llm = FakeLLM()
    generate_queries(llm, load_config(EXAMPLE_CONFIG), main, date(2021, 10, 13), date(2026, 10, 13), cb)
    v = llm.calls[0][1]
    assert v["search_terms"] == "H02 term a, H02 term b" and v["definition"] == "Definition of H02."
