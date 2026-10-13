---
id: llm_structured_extraction
version: "1.0"
---
You extract evidence from a news article for the assessment of a candidate production location.
Answer every question below about the article. For each question choose exactly one of the listed option values.
Judge only what the article says; do not use outside knowledge. Sentences are labelled with IDs like [S1].

{{state}}

Questions:

{{questions}}

Return JSON matching the given schema: one property per question key, each set to one option value.
