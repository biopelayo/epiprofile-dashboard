"""Self-contained HTML report for one experiment.

``build_report(sections, title, subtitle)`` takes a list of
(heading, description, plotly.graph_objects.Figure | pandas.DataFrame | str)
and returns an HTML string. Plotly.js is loaded from the CDN once, so the
file stays small and opens offline-free in any browser.
"""
from __future__ import annotations

import html as _html
from datetime import datetime

import pandas as pd

_CSS = """
body{font-family:Inter,'Segoe UI',Arial,sans-serif;margin:0;background:#f8faf8;color:#0f1f13}
header{background:linear-gradient(135deg,#0a1a0e,#14532d);color:#fff;padding:28px 40px}
header h1{margin:0;font-size:28px;letter-spacing:-.5px}
header p{margin:6px 0 0;color:#86efac;font-size:14px}
main{max-width:1200px;margin:0 auto;padding:24px 40px}
section{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px 24px;margin:0 0 20px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
section h2{margin:0 0 4px;font-size:18px}
section p.desc{margin:0 0 12px;color:#6b7280;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left}
th{background:#ecfdf5;color:#065f46;font-weight:600}
footer{color:#9ca3af;font-size:12px;text-align:center;padding:20px}
.toc a{color:#15803d;text-decoration:none;margin-right:14px;font-size:13px}
"""


def _fig_html(fig, idx):
    try:
        return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig{idx}",
                           config={"displaylogo": False, "responsive": True})
    except Exception as e:  # pragma: no cover
        return f"<pre>figure error: {_html.escape(str(e))}</pre>"


def build_report(sections, title="EpiProfile-Plants report", subtitle="", max_table_rows=200) -> str:
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{_html.escape(title)}</title>",
             "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
             f"<style>{_CSS}</style></head><body>",
             f"<header><h1>{_html.escape(title)}</h1><p>{_html.escape(subtitle)} &middot; generated {datetime.now():%Y-%m-%d %H:%M}</p></header><main>"]
    toc = ["<section class='toc'>"]
    for i, (h, _, _) in enumerate(sections):
        toc.append(f"<a href='#s{i}'>{_html.escape(h)}</a>")
    toc.append("</section>")
    parts.extend(toc)
    for i, (heading, desc, body) in enumerate(sections):
        parts.append(f"<section id='s{i}'><h2>{_html.escape(str(heading))}</h2>")
        if desc:
            parts.append(f"<p class='desc'>{_html.escape(str(desc))}</p>")
        if isinstance(body, pd.DataFrame):
            parts.append(body.head(max_table_rows).to_html(index=False, border=0, float_format=lambda v: f"{v:.4g}"))
            if len(body) > max_table_rows:
                parts.append(f"<p class='desc'>showing {max_table_rows} of {len(body)} rows</p>")
        elif isinstance(body, str):
            parts.append(f"<div>{body}</div>")
        elif body is None:
            pass
        else:
            parts.append(_fig_html(body, i))
        parts.append("</section>")
    parts.append("</main><footer>EpiProfile-Plants Dashboard &middot; https://github.com/biopelayo/epiprofile-dashboard</footer></body></html>")
    return "\n".join(parts)
