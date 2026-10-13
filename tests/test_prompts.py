import pytest

from locassess.prompts import PromptError, load_prompt


@pytest.mark.parametrize("pid", ["query_generation", "relevance", "web_search"])
def test_prompts_have_version_header(pid):
    p = load_prompt(pid)
    assert p.id == pid and p.version


def test_render_and_missing_variables(tmp_path):
    (tmp_path / "x.md").write_text('---\nid: x\nversion: "2.1"\n---\nHello {{ name }}, {{name}}!\n', encoding="utf-8")
    p = load_prompt("x", tmp_path)
    assert p.version == "2.1"
    assert p.render({"name": "Leipzig"}) == "Hello Leipzig, Leipzig!\n"
    with pytest.raises(PromptError):
        p.render({})


def test_header_required(tmp_path):
    (tmp_path / "y.md").write_text("no header", encoding="utf-8")
    with pytest.raises(PromptError):
        load_prompt("y", tmp_path)
