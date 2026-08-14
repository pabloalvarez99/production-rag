"""Render data/eval/reports/scorecard.json into a human-readable HTML page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = ROOT / "data/eval/reports/scorecard.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/scorecard.html"
METRICS = (
    "hit_at_5",
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
    "citation_precision",
    "invalid_marker_rate",
    "refusal_accuracy",
)


def render(scorecard: Path) -> str:
    """Return HTML for the free-path scorecard with honest labels."""
    data = json.loads(scorecard.read_text(encoding="utf-8"))
    prov = data["provenance"]
    rows: list[str] = []
    for name, metrics in data["configs"].items():
        cells = "".join(f"<td>{float(metrics[m]):.4f}</td>" for m in METRICS)
        rows.append(f'<tr><th scope="row">{name}</th>{cells}</tr>')
    n = prov["golden"]["items"]
    # CSS lines are long by design; keep the style block compact rather than
    # wrapping every property for a linter that is not the audience.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>production-rag free-path scorecard</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ max-width: 52rem; margin: 2rem auto; padding: 0 1rem 3rem; line-height: 1.5; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    .banner {{
      border: 1px solid #c45c26; background: #fff4ec; color: #3b1d0f;
      padding: 0.85rem 1rem; border-radius: 0.5rem; margin: 1rem 0 1.5rem;
    }}
    .banner strong {{ display: block; margin-bottom: 0.25rem; }}
    .meta {{
      display: grid; grid-template-columns: auto 1fr;
      gap: 0.25rem 1rem; font-size: 0.95rem; margin-bottom: 1.5rem;
    }}
    .meta dt {{ font-weight: 600; }}
    table {{
      width: 100%; border-collapse: collapse;
      font-variant-numeric: tabular-nums; font-size: 0.9rem;
    }}
    th, td {{ border-bottom: 1px solid #ccc; padding: 0.45rem 0.4rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }}
    footer {{ margin-top: 2rem; font-size: 0.85rem; opacity: 0.85; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Free-path scorecard</h1>
  <p>Human-readable view of <code>data/eval/reports/scorecard.json</code> for production-rag.</p>
  <div class="banner" role="note">
    <strong>Contract / plumbing fixture — not SOTA, not quality, not billed.</strong>
    Deterministic local providers (<code>embedder=fake</code>, <code>llm=fake</code>,
    <code>judge=none</code>). These numbers prove the evaluation pipeline and report
    contract work end-to-end. They do <em>not</em> measure semantic retrieval or answer
    quality. Do not cite them as a hosted baseline.
  </div>
  <dl class="meta">
    <dt>billed</dt><dd><code>{str(prov["billed"]).lower()}</code></dd>
    <dt>n (golden items)</dt><dd>{n}</dd>
    <dt>cost_usd</dt><dd>{prov["cost_usd"]}</dd>
    <dt>generated_at</dt><dd>{data["generated_at"]}</dd>
    <dt>commit</dt><dd><code>{prov["commit"][:12]}</code></dd>
    <dt>corpus</dt><dd>{prov["corpus"]["path"]}
      ({prov["corpus"]["documents"]} docs / {prov["corpus"]["chunks"]} chunks)</dd>
    <dt>schema</dt><dd>v{data["schema_version"]}</dd>
  </dl>
  <table>
    <thead>
      <tr>
        <th scope="col">config</th>
        <th scope="col">hit@5</th>
        <th scope="col">recall@5</th>
        <th scope="col">mrr</th>
        <th scope="col">ndcg@5</th>
        <th scope="col">cit. prec.</th>
        <th scope="col">inv. marker</th>
        <th scope="col">refusal acc.</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <footer>
    <p>Source of truth: <code>data/eval/reports/scorecard.json</code>. README region is
    rendered by <code>tools/render_docs.py --check</code>. Open this file locally or
    serve it from <code>GET /evals</code> on a running free-path stack. No hosting, no
    API keys, no claim of SOTA.</p>
  </footer>
</body>
</html>
"""


def main() -> int:
    """CLI entry: write the HTML scorecard or check it is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if output would change")
    args = parser.parse_args()
    html = render(args.scorecard)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != html:
            print(f"{args.output} is stale; run: python tools/render_scorecard_html.py")
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
