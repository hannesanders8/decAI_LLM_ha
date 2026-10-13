"""Versioned prompt templates in prompts/*.md.

Each file starts with a front-matter header:

    ---
    id: relevance
    version: "1.0"
    ---
    Template text with {{placeholders}}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    template: str

    def render(self, variables: dict) -> str:
        missing = {m for m in _VAR_RE.findall(self.template) if m not in variables}
        if missing:
            raise PromptError(f"prompt {self.id}: missing variables {sorted(missing)}")
        return _VAR_RE.sub(lambda m: str(variables[m.group(1)]), self.template)


def load_prompt(prompt_id: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    path = prompts_dir / f"{prompt_id}.md"
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        raise PromptError(f"{path}: missing front-matter header")
    header = yaml.safe_load(m.group(1)) or {}
    if header.get("id") != prompt_id or not header.get("version"):
        raise PromptError(f"{path}: header needs id: {prompt_id} and a version")
    return Prompt(prompt_id, str(header["version"]), m.group(2).strip() + "\n")
