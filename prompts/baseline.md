---
id: baseline
version: "1.0"
---
You assess a candidate production location (a site for an industrial plant) using only the news articles below.

Location: {{location}}
Assessment period: {{period}}

Assess each of the 26 main criteria of the framework:
{{criteria_list}}

For every criterion give:
- score: from -3 to +3 (one decimal allowed). +3 = the articles show the location is very attractive for a production site with respect to this criterion, -3 = very unattractive, 0 = mixed or neutral evidence. Use null if the articles contain no information on this criterion for this location.
- rating: "green" if score >= +1, "red" if score <= -1, otherwise "yellow"; null if the score is null.
- justification: one or two sentences in English.
- evidence: the passages your assessment is based on — for each, the article label (e.g. "A3") and an exact quote copied verbatim from that article (one sentence, in the article's original language). Empty list if the score is null.

Then give an overall judgement of the location: an overall score from -3 to +3 (null if nothing can be said) and a short judgement of at most three sentences.

Use only the articles; do not use outside knowledge.

Articles:

{{articles}}

Return JSON matching the given schema.
