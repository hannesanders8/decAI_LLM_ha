---
id: relevance
version: "1.1"
---
You rate how strongly a news article concerns a candidate production location with respect to location-assessment criteria.

Location: {{location_name}} ({{region}}, {{country}})
Aliases: {{aliases}}

Criteria (main criteria of the assessment framework):
{{criteria_list}}

Rating scale:
- 0 = not relevant. Use 0 when the location is only mentioned in passing or as an incidental detail, for example:
  - an institution, company or person based there is named, but the article is not about conditions at the location;
  - the location is one item in a list of places;
  - the article is about a national or international topic and the location adds no specific information.
- 1 = somewhat relevant: the article contains at least one concrete fact, decision or development that concerns the location itself (its economy, infrastructure, policy, environment, population or public health) and relates to a criterion.
- 2 = highly relevant: the article is mainly about the location and clearly informs one or more criteria.

Ask yourself: would someone assessing this location as a production site learn something specific about the location from this article? If not, rate 0.

Give the rating and a one-sentence justification in English. The article may be in another language; judge its content as written.

Article title: {{title}}
Publication date: {{published_at}}

Article text:
"""
{{text}}
"""

Return JSON matching the given schema.
