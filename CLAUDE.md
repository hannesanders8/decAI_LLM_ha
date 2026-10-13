# CLAUDE.md

Project: **LocAssess** — news-based assessment of production locations (Master's thesis).

- The full build specification is in `SPEC.md`. Read it completely before writing code and follow it. Sections marked MUST are part of the research design; ask before changing them.
- Build milestone by milestone (SPEC §16). After each milestone: run the tests, summarise what was built, and wait for my go-ahead.
- Everything in the program is English: code, comments, prompts, UI, reports, database content.
- Never hard-code or print API keys. Secrets come from `.env` only; keep `.env` and `data/locassess.db` git-ignored.
- Do not invent external API endpoints or request formats (Jev, OpenAI Decisions, search). Read the provider's official docs first; if access or details are missing, build the adapter skeleton with a clear `ProviderNotConfigured` error and tell me.
- Stage 2 (scoring) must stay free of model calls and fully deterministic; the worked example in SPEC §9.8 must be a passing unit test.
- Mock all model and search calls in tests.
- When something in the spec is ambiguous, ask instead of guessing.
