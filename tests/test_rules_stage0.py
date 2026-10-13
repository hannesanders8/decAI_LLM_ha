"""SPEC §7.4 rule filter, §7.6 selection, and helpers of Stage 0."""

from datetime import date

import pytest

from locassess.config import load_config
from locassess.criteria import load_criteria_csv
from locassess.stage0_corpus import (
    Candidate, FilterInput, WhitelistEntry, WhitelistError, filter_reason, find_duplicates,
    generate_queries, load_whitelist, match_whitelist, mentions_location, normalize_url, period,
    select_articles, sentence_rows, split_sentences,
)
from tests.conftest import CRITERIA_CSV, EXAMPLE_CONFIG
from tests.fakes import FakeLLM

WL = [WhitelistEntry("lvz.de", "LVZ", "regional", "de", "DE"),
      WhitelistEntry("mdr.de", "MDR", "regional", "de", "DE")]
FROM, TO = date(2021, 10, 13), date(2026, 10, 13)
NAMES = ["Leipzig", "Region Leipzig"]
TEXT = "Leipzig " + " ".join(f"word{i}" for i in range(300))


def fi(url="https://www.lvz.de/a", pub=date(2025, 1, 1), text=TEXT, aid=1, title=None):
    return FilterInput(aid, url, pub, text, title)


def reason(a, min_words=200):
    return filter_reason(a, WL, FROM, TO, NAMES, min_words)


# --- whitelist / URLs ------------------------------------------------------

@pytest.mark.parametrize("url,ok", [
    ("https://www.lvz.de/x", True), ("https://lvz.de/x", True), ("https://sport.mdr.de/x", True),
    ("https://notlvz.de/x", False), ("https://lvz.de.evil.com/x", False), ("https://example.com", False),
])
def test_whitelist_match(url, ok):
    assert (match_whitelist(url, WL) is not None) == ok


def test_load_whitelist(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("domain,outlet_name,media_type,language,country\nhttps://www.LVZ.de/,LVZ,regional,de,DE\n",
                 encoding="utf-8")
    assert load_whitelist(p)[0].domain == "lvz.de"


def test_empty_whitelist_rejected(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("domain,outlet_name,media_type,language,country\n", encoding="utf-8")
    with pytest.raises(WhitelistError):
        load_whitelist(p)


def test_normalize_url():
    assert normalize_url("http://www.LVZ.de/a/?utm_source=x&id=2#top") == normalize_url("https://lvz.de/a?id=2")
    assert normalize_url("https://lvz.de/a?id=2") != normalize_url("https://lvz.de/a?id=3")


def test_period():
    assert period(date(2026, 10, 13), 5) == (FROM, TO)
    assert period(date(2028, 2, 29), 1) == (date(2027, 2, 28), date(2028, 2, 29))


# --- 7.4 rule filter -------------------------------------------------------

def test_passes_all_rules():
    assert reason(fi()) is None


def test_rule_domain():
    assert reason(fi(url="https://example.com/a")) == "filter:domain_not_whitelisted"


def test_rule_date_missing_and_outside():
    assert reason(fi(pub=None)) == "filter:no_publication_date"
    assert reason(fi(pub=date(2021, 10, 12))) == "filter:outside_period"
    assert reason(fi(pub=date(2026, 10, 14))) == "filter:outside_period"
    assert reason(fi(pub=FROM)) is None and reason(fi(pub=TO)) is None


def test_rule_location_mention_case_insensitive():
    assert reason(fi(text=TEXT.replace("Leipzig", "LEIPZIG"))) is None
    assert reason(fi(text=TEXT.replace("Leipzig", "Dresden"))) == "filter:location_not_mentioned"
    assert not mentions_location("Leipziger Allerlei", ["Leipzig"])  # whole word only
    assert mentions_location("in der region leipzig", NAMES)


def test_rule_location_in_title_counts():
    no_place = TEXT.replace("Leipzig", "Dresden")
    assert reason(fi(text=no_place, title="Trauernde in Leipzig")) is None
    assert reason(fi(text=no_place, title="Trauernde in Dresden")) == "filter:location_not_mentioned"


def test_rule_min_words():
    assert reason(fi(), min_words=301) is None
    assert reason(fi(), min_words=302) == "filter:too_short"


def test_rule_order_first_failure_reported():
    assert reason(fi(url="https://example.com", pub=None, text="short")) == "filter:domain_not_whitelisted"


def test_duplicates_same_url_and_near_identical_text():
    other = "Leipzig " + " ".join(f"other{i}" for i in range(300))
    near = TEXT.replace("word5 ", "changed ")  # one word differs
    arts = [fi(aid=1), fi(url="https://lvz.de/a/?utm_source=x", text=other, aid=2),
            fi(url="https://mdr.de/copy", text=near, aid=3), fi(url="https://mdr.de/b", text=other + " x", aid=4)]
    dups = find_duplicates(arts)
    assert dups[2] == 1 and dups[3] == 1
    assert 4 not in dups  # different from 1; 2 was a duplicate, so it is not a reference


def test_low_similarity_not_duplicate():
    half = "Leipzig " + " ".join(f"word{i}" for i in range(150)) + " " + " ".join(f"new{i}" for i in range(150))
    assert find_duplicates([fi(aid=1), fi(url="https://mdr.de/b", text=half, aid=2)]) == {}


# --- 7.6 selection ---------------------------------------------------------

def cand(aid, rel=2, d=date(2025, 1, 1), crit=("H01",)):
    return Candidate(aid, rel, d, frozenset(crit))


def test_min_relevance():
    assert select_articles([cand(1, rel=0), cand(2, rel=1)], 1, 4, 15) == [2]


def test_max_per_year():
    cs = [cand(i, crit=(f"H{i:02d}",)) for i in range(1, 6)]
    assert len(select_articles(cs, 1, 4, 15)) == 4


def test_target_stops():
    cs = [cand(i, d=date(2021 + i % 5, 1, 1), crit=(f"H{i:02d}",)) for i in range(1, 20)]
    assert len(select_articles(cs, 1, 4, 15)) == 15


def test_uncovered_criteria_first():
    cs = [cand(1, rel=2, crit=("H01",)), cand(2, rel=2, crit=("H01",)), cand(3, rel=1, crit=("H02",))]
    assert select_articles(cs, 1, 4, 2) == [1, 3]


def test_ties_relevance_then_newer_then_id():
    cs = [cand(1, rel=1, d=date(2026, 1, 1)), cand(2, rel=2, d=date(2024, 1, 1)),
          cand(3, rel=2, d=date(2025, 1, 1)), cand(4, rel=2, d=date(2025, 1, 1))]
    assert select_articles(cs, 1, 4, 4) == [3, 4, 2, 1]


def test_selection_deterministic_regardless_of_order():
    cs = [cand(i, rel=1 + i % 2, d=date(2022 + i % 4, 1 + i % 12, 1), crit=(f"H{1 + i % 7:02d}",))
          for i in range(1, 30)]
    assert select_articles(cs, 1, 4, 15) == select_articles(list(reversed(cs)), 1, 4, 15)


# --- 7.1 query rules -------------------------------------------------------

def test_generate_queries_rules():
    cfg = load_config(EXAMPLE_CONFIG)
    main = load_criteria_csv(CRITERIA_CSV)[1]  # H02 Transportation system
    llm = FakeLLM(queries=lambda v: [
        {"language": "en", "query": "Leipzig  railway"},
        {"language": "en", "query": "leipzig railway"},          # duplicate (case-insensitive)
        {"language": "en", "query": "airport freight"},          # no location → prefixed
        {"language": "en", "query": "Leipzig extra"},            # beyond n
        {"language": "de", "query": "Leipzig Autobahn"},
        {"language": "de", "query": "Hafen Region Leipzig"},
    ])
    queries, version = generate_queries(llm, cfg, main, FROM, TO)
    assert queries == [("en", "Leipzig railway"), ("en", "Leipzig airport freight"),
                       ("de", "Leipzig Autobahn"), ("de", "Hafen Region Leipzig")]
    assert version == "1.0"
    variables = llm.calls[0][1]
    assert "Railway" in variables["search_terms"] and variables["languages"] == "en, de"


# --- sentences -------------------------------------------------------------

def test_split_sentences_and_sections():
    s = split_sentences("Der Hafen wurde erweitert. Die Stadt investiert 5 Mio. Euro.\n\nNeuer Absatz hier.", "de")
    assert s == ["Der Hafen wurde erweitert.", "Die Stadt investiert 5 Mio. Euro.", "Neuer Absatz hier."]
    rows = sentence_rows(["a", "b", "c", "d", "e"], 2)
    assert rows == [("S1", 1, "a"), ("S2", 1, "b"), ("S3", 2, "c"), ("S4", 2, "d"), ("S5", 3, "e")]


def test_split_unknown_language_falls_back():
    assert split_sentences("One. Two.", "xx") == ["One.", "Two."]
