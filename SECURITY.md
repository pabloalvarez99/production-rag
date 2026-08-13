# Security policy

## Supported versions

There are no tagged releases. Only the tip of `main` is supported; fixes land there and are
not backported.

## Never put a credential in this repository

This applies to issues, pull requests, discussions, commit messages, test fixtures, and
code — anywhere the value could be read.

- Real values go in `.env`, which is gitignored and dockerignored.
- `.env.example` is the committed template. It carries names, shapes, and defaults, and
  must never contain a value.
- When a log or traceback is useful evidence, redact the credential **before** pasting it.
  Do not rely on a reviewer noticing.

The entire documented free path — demo, ingest, query, UI, both evaluation tiers, the
scorecard — runs on deterministic local providers and needs no credential at all. If a
reproduction seems to need a key, say so in the report instead of attaching one; that is
usually the bug.

### If a key is exposed anyway

1. **Rotate it first.** Revoking and reissuing is cheap; assuming nobody saw it is not.
2. Then remove it from the working tree.
3. If it was already committed, deleting the file is not enough — the value stays in the
   history and in every clone and fork. Rotation is the fix; history rewriting is cleanup.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: the repository's **Security** tab →
**Report a vulnerability**. That keeps the details out of public view while the issue is
open.

Please include the affected path or endpoint, the exact request or command, what happened,
and what you expected. A minimal reproduction against the free local path is ideal.

Do not open a public issue for an exploitable finding, and do not include a credential
value in the report.

This is a portfolio project maintained by one person and is not operated as a hosted
service, so there is no response-time commitment. Reports are handled on a best-effort
basis, and the outcome is recorded in the repository rather than in a private advisory
whenever disclosing it costs nothing.

## What the code already guarantees

These properties have tests behind them, so a change that breaks one fails the suite rather
than the next reviewer's attention:

- `Settings.safe_dump()` is the only sanctioned way to serialise configuration, and it masks
  every declared secret field. Nothing in the codebase logs a plain settings dump.
- `/health` and `/v1/ready` never echo a credential in a response body or header, including
  when one is configured.
- Caller-controlled `debug` output is allowlisted to node timings and invalid citation markers.
  It never returns prompt text, the system prompt, rendered evidence blocks, passage text
  beyond the citations already in the answer, the collection name, or provider identity.
- The UI references no external origin: assets are vendored under `src/production_rag/static/`,
  and a test scans the templates and static files to keep it that way.
- Model output reaches the page as text through the template engine, never as markup, and a
  citation marker only becomes a link when it resolves to a citation the answer carries.
- A hosted-provider run refuses to start without an explicit `--yes-spend` after printing its
  cost estimate, so a credential present in the environment cannot quietly become a bill.

Trace export is the one path that would send prompts and answers to a third party. It is off by
default, and installing the optional extra does not enable it.

## Known boundaries — by design, not vulnerabilities

The service is built to be read and run locally. The following are stated positions, and
reporting them as findings is not necessary:

- **No authentication, no authorization, no rate limiting.** These are named as out of
  scope in the README's capability table and assigned to a later project in the series.
  Anyone who can reach the port can query the service.
- **The container binds `0.0.0.0`** so the host can reach it through Compose. Do not expose
  the published port to an untrusted network.
- **The local Qdrant container runs without an API key.** `QDRANT_API_KEY` exists for
  Qdrant Cloud; the development stack does not use it.
- **Ingest reads local files you point it at** and stores their text as retrievable chunks.
  Anything ingested becomes answerable and citable — treat the corpus as public relative to
  whoever can query the service.
- **Answers cite their sources; they do not verify them.** Citations provide provenance,
  not automatic truth.

What *is* in scope: anything that leaks a credential, escapes the documented input path,
makes the service perform a billed provider call the caller did not opt into, or lets a
crafted document or question reach beyond the retrieval and generation contracts described
in [docs/architecture.md](docs/architecture.md).
