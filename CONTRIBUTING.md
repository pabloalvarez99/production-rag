# Contributing

This is a portfolio repository: it is public so the engineering can be read and run, not
because it needs feature contributions. Bug reports, correctness fixes, and documentation
corrections are welcome. Before proposing a new capability, read
[docs/PORTFOLIO.md](docs/PORTFOLIO.md) — most "missing" features are deliberately assigned
to a later project in the series rather than absent by oversight.

Everything below runs on deterministic local providers. No credential, no billed call, no
signup.

## Prerequisites

- Docker (with Compose)
- Python 3.12+
- Git

## Run the free demo

```powershell
git clone https://github.com/pabloalvarez99/production-rag
cd production-rag
.\scripts\demo_setup.ps1          # macOS or Linux: ./scripts/demo_setup.sh
```

That starts Qdrant, rebuilds the `prag_demo` collection from `data/corpus` with the
deterministic embedder, and serves the UI at <http://localhost:8000/>. Ask *why does hybrid
search use reciprocal rank fusion?* for a cited answer, then *who won the Antarctic
underwater chess championship?* for an explicit refusal.

The [README](README.md#try-it-free-0-no-api-key) covers the rest of the free path: ingest
and query over HTTP and the CLI, both evaluation tiers, and regenerating the scorecard
artefact.

## Run the checks

Install the development extras once, then run the same four commands CI runs:

```bash
pip install -e ".[dev]"
ruff check .
mypy --strict
pytest -q
python tools/render_docs.py --check
```

`make test` runs the suite inside the API container; `make test-host` runs it on the host
against a local virtual environment with the dev extras. `make help` lists every target.

CI runs the suite, the ingest, both evaluation tiers, and an HTTP query with
`OPENAI_API_KEY`, `COHERE_API_KEY`, and `QDRANT_API_KEY` all set to the empty string. That
is deliberate: if a change makes any of those paths require a credential, CI goes red. Keep
it that way — a new test must pass without keys, or be marked as an integration test that
does not run by default.

Two invariants sit underneath that. The suite makes **no network call**: anything that
reaches Qdrant is `@pytest.mark.integration` and skipped unless `RUN_QDRANT_TESTS=1`, and a
library that downloads a file on first use — a tokenizer, a model — is stubbed rather than
trusted to a warm cache, because a cache makes the gap invisible on the machine that wrote
the test. And it reads **no ambient configuration**: settings come from explicit values with
the on-disk `.env` disabled, so a stray exported variable cannot turn a passing suite red.

## No secrets, ever

- `.env` and `.env.*` are gitignored; `.env.example` is the only committed template and
  carries names, shapes, and defaults — never a value.
- Do not paste a key, token, or connection string into code, tests, fixtures, commit
  messages, issues, or pull requests. See [SECURITY.md](SECURITY.md) for what to do if one
  is exposed anyway.
- The pull request checklist has a line for this. It is not a formality; the diff is read
  for it.

## What the review looks for

1. **Claims match code.** The README describes only what exists at the integrated tip.
   `tests/test_readme_claims.py` fails when an unimplemented surface is mentioned without
   an honesty marker, so an aspirational sentence breaks the build rather than shipping.
2. **Published numbers carry their provenance.** Any metric that reaches documentation
   carries its embedder, LLM, judge, sample size, date, and commit. Never hand-edit the
   region between the `SCORECARD:START` and `SCORECARD:END` markers in the README —
   regenerate it with `python tools/render_docs.py` from
   `data/eval/reports/scorecard.json`, and let `--check` prove the two agree. A guard applies
   the same rule to every tracked Markdown file, so a number that is history or an
   illustration rather than a current claim needs a marker on the line above it —
   `<!-- provenance-allow: historical-measurement: why this is not a current claim -->` —
   with the category one of `historical-measurement`, `explanatory-example`, or
   `external-citation`.
3. **Local-provider runs are not quality results.** A deterministic run proves plumbing.
   Do not present one as a measurement of retrieval or answer quality, and do not add a
   paid-provider number that has not actually been run.
4. **Non-obvious trade-offs become records.** A decision with a rejected alternative
   belongs in `docs/adr/` alongside the change, not only in the commit body.
5. **Tests are offline by default.** Prefer unit tests with no network. Integration tests
   are marked and opt-in.
6. **Configuration status stays commented.** A key in `configs/default.yaml` carries `LIVE`
   or `DECLARED` with its reason. A change that flips one updates that comment, so the YAML
   never describes a capability the code does not have.
7. **UI captures are regenerated, not redrawn.** A change to the templates or the stylesheet
   refreshes `docs/assets/*.png` in the same pull request: `pip install -e ".[docs]"`,
   `playwright install chromium`, `python scripts/capture_ui.py`. Identical inputs produce
   byte-identical PNGs and the script prints each digest, so a stale capture is provable
   rather than arguable.

## Commits and pull requests

Commit subjects use a conventional prefix and, where the change belongs to a milestone
wave, the seat that owned it — `feat(m4-a1)`, `feat(m5-a3)`, `docs(portfolio)`. Keep the
history readable as a sequence of decisions.

The [pull request template](.github/pull_request_template.md) encodes the multi-agent
ownership protocol used to build this repository: each wave assigns files to one seat, the
merge order is A1 core → A2 docs → A3 glue, and every seat verifies the integrated tip
rather than a shared uncommitted workspace. An outside contribution does not need a seat —
tick the boxes that apply and leave the ownership line blank.

## Reporting a bug

Open an issue with the command you ran, the providers involved (`fake` unless you opted
in), and what you expected instead. A failing test is the most useful form of the report.
Do not include a credential value — not even a redacted-looking one.
