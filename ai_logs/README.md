# AI chat logs

Available user-facing transcripts and outstanding exports for AI sessions that contributed to this
project (see `docs/AI_USE.md`). This directory is not yet a complete archive of the project's earlier
sessions.

Naming: `YYYY-MM-DD_<short-topic>.<md|json|txt>`, e.g. `2026-11-15_repo-implementation.md`.

Expected entries:

| Session | Status |
| --- | --- |
| Paper discussion and specification drafting (before 2026-11-15) | Unavailable on this machine; export still needed from the original session |
| 2026-11-15 repository creation and local implementation | Unavailable on this machine; export still needed from the original session |
| 2026-11-16 UTC GPU validation, explanations, backend diagnosis, worker recovery, bucketing, reporting, and packaging request | [Authentic user-facing snapshot](2026-11-16_gpu-investigation.md): all 20 recorded human messages, visible replies/completion summaries, and aggregate tool inventory |

The current snapshot is generated from actual CLI events, with marked privacy redactions; it is not
reconstructed dialogue. It excludes private reasoning, system instructions, raw tool payloads, and
messages after its cutoff (including the packaging completion acknowledgement). Historical
citation markers may not resolve outside the original session. `docs/AI_USE.md` tracks each request
and explains corrections; measured claims are supported by `results/`, not by chat text alone.

Checklist before committing a log:

- [ ] coverage/cutoff and omissions declared; no fabricated missing dialogue
- [ ] no edited visible wording other than marked redactions and whitespace normalization
- [ ] no credentials, tokens, keys, or private URLs
- [ ] private names, hostnames, and home-directory usernames redacted where needed, marked `[REDACTED: reason]`
- [ ] entry added to the table above and to `docs/AI_USE.md`
