---
id: query_generation
version: "1.0"
---
You generate web search queries to find news articles about a candidate production location.

Location: {{location_name}} ({{region}}, {{country}})
Aliases: {{aliases}}
Period: {{date_from}} to {{date_to}}

Criterion {{main_id}} — {{criterion_name}}
Definition: {{definition}}
Search terms: {{search_terms}}

Write exactly {{n}} search queries in each of these languages: {{languages}}.

Rules:
- Every query contains the location name or one of its aliases.
- Every query combines the location with one or more of the search terms (translated where needed).
- Queries should find news reporting (events, decisions, investments, closures, problems), not encyclopedic pages.
- Keep each query short (3–8 words). No site: operators, no quotation marks, no dates.
- Queries in the same language must differ from each other.

Return JSON matching the given schema.
