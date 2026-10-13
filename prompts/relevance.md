---
id: relevance
version: "1.0"
---
You rate how strongly a news article concerns a candidate production location with respect to location-assessment criteria.

Location: {{location_name}} ({{region}}, {{country}})
Aliases: {{aliases}}

Criteria (main criteria of the assessment framework):
{{criteria_list}}

Rating scale:
- 0 = not relevant: the location is only mentioned in passing, or the article does not inform any criterion for this location.
- 1 = somewhat relevant: the article contains at least one concrete piece of information about the location that relates to a criterion.
- 2 = highly relevant: the article is mainly about the location and clearly informs one or more criteria.

Give the rating and a one-sentence justification in English. The article may be in another language; judge its content as written.

Article title: {{title}}
Publication date: {{published_at}}

Article text:
"""
{{text}}
"""

Return JSON matching the given schema.
