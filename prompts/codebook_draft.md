---
id: codebook_draft
version: "1.0"
---
You help build a codebook for assessing candidate production locations (industrial sites, plants, factories) from news articles. The codebook is used in two ways:
1. as instructions for a model that decides whether a news article contains evidence for a criterion, and
2. as the annotation guideline for human coders who create a gold standard.

The criteria framework (Hoang-Thi et al. 2026) has 26 main criteria grouped into 6 factors:
{{all_criteria}}

Draft the codebook entry for this main criterion:
- ID: {{main_id}}
- Factor: {{factor}}
- Name: {{name}}
- Sub-criteria: {{sub_criteria}}

Write:
- definition: 1–3 sentences. What the criterion covers at a production location, phrased so a coder can decide whether a news sentence is evidence for it. Interpret academic terms in the context of a company choosing a production site. Mention the sub-criteria aspects.
- notes: 1–3 sentences on boundaries — what does NOT count, and how to separate this criterion from the most similar other criteria of the framework (name them by ID).
- positive_examples: 3 short, realistic news sentences (English, generic place "the region") that would make the location MORE attractive for production with respect to this criterion.
- negative_examples: 3 short, realistic news sentences that would make the location LESS attractive with respect to this criterion.
- search_terms: 8–12 English search terms that journalists actually use in news reporting about this topic (concrete nouns and short phrases, e.g. "plant closure", "skilled labour shortage", "business tax rate"). Avoid academic vocabulary that rarely appears in news.

Everything in English. Return JSON matching the given schema.
