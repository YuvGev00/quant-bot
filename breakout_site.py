#!/usr/bin/env python3
"""breakout_site.py — regenerates breakout_site/ (static research site).

The index leads with the EARLY DISCOVERY section (from breakout_discover.json) and
one disc_<TICKER>.html detail page per name. If breakout_ideas.json (the large-cap
'breakout' mode) is present it is appended as a secondary section; if absent the site
renders discovery-only, which is fine.

Pure stdlib, no deps. Run: python3 breakout_site.py
"""
from __future__ import annotations
import json, os, html
from datetime import datetime

OUT_DIR = "breakout_site"
DISCOVER = "breakout_discover.json"
IDEAS = "breakout_ideas.json"

FIT_COLOR = {"STRONG": "#16c784", "MODERATE": "#f3b13e", "WEAK": "#8a93a6"}
TIER_LABEL = {"EMERGING": "EMERGING · mid-cap, steadier",
              "SPECULATIVE_EARLY": "SPECULATIVE EARLY · micro/IPO, lottery-risk"}
FIT_RANK = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
TIER_RANK = {"EMERGING": 0, "SPECULATIVE_EARLY": 1}

CSS = """
:root{--bg:#0b0e14;--card:#141925;--ink:#e6e9ef;--mut:#8a93a6;--line:#232a3a;--acc:#4f8cff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px}
.sub{color:var(--mut);font-size:13px;margin:0 0 18px}
.disc{background:#1a130a;border:1px solid #5a3d12;color:#f1c98a;padding:11px 14px;border-radius:10px;font-size:12.5px;margin:14px 0 26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 16px}
.card h3{margin:0;font-size:18px}
.row{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
.badge{font-size:11px;font-weight:700;letter-spacing:.4px;padding:3px 9px;border-radius:999px;color:#0b0e14}
.tier{font-size:11px;color:var(--mut);border:1px solid var(--line);padding:3px 9px;border-radius:999px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:14px 0}
.metric{background:#0f1420;border:1px solid var(--line);border-radius:10px;padding:9px 11px}
.metric .k{font-size:11px;color:var(--mut)}.metric .v{font-size:15px;font-weight:600;margin-top:2px}
.thesis{margin:10px 0 0;color:#cdd3df}
.risk{margin:8px 0 0;color:#f0a3a3;font-size:13.5px}
.sig{margin:10px 0 0;padding:0;list-style:none;font-size:13px;color:var(--mut)}
.sig li{margin:2px 0}.sig li:before{content:"▸ ";color:var(--acc)}
.news{margin:10px 0 0;font-size:12.5px;color:var(--mut)}
.foot{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:14px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600}
.pill{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:999px;color:#0b0e14}
"""


def esc(x):
    return html.escape(str(x)) if x is not None else ""


def fmt(v, suffix="", pct=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        v = round(v, 2)
    return f"{v}{'%' if pct else ''}{suffix}"


def disc_sort_key(d):
    return (FIT_RANK.get(d.get("early_fit"), 9), TIER_RANK.get(d.get("tier"), 9),
            -(d.get("market_cap_b") or 0))


def card_html(d):
    fit = d.get("early_fit", "WEAK")
    color = FIT_COLOR.get(fit, "#8a93a6")
    tkr = esc(d.get("ticker"))
    r = d.get("ratings") or {}
    ratings = f"B{r.get('buy','?')}/H{r.get('hold','?')}/S{r.get('sell','?')}" if r else "—"
    metrics = [
        ("Market cap", fmt(d.get("market_cap_b"), "B")),
        ("Price", "$" + fmt(d.get("price"))),
        ("52w range pos", fmt(d.get("base_pos_pct"), "%")),
        ("Analysts", fmt(d.get("n_analysts")) + (f" ({esc(d.get('coverage'))})" if d.get("coverage") else "")),
        ("Ratings B/H/S", ratings),
        ("Rev growth", fmt(d.get("rev_growth_pct"), pct=True)),
        ("EPS surprise", fmt(d.get("earnings_surprise_pct"), pct=True)),
        ("EPS rising", "Yes" if d.get("eps_rising") else ("No" if d.get("eps_rising") is False else "—")),
        ("Cheap?", "Yes" if d.get("cheap") else ("No" if d.get("cheap") is False else "—")),
    ]
    mh = "".join(f'<div class="metric"><div class="k">{esc(k)}</div><div class="v">{v}</div></div>' for k, v in metrics)
    sigs = "".join(f"<li>{esc(s)}</li>" for s in d.get("early_signals", []))
    news = "".join(
        f'<div>&middot; {esc(n.get("headline"))} '
        f'<span style="color:#6f7890">({esc(n.get("source"))} — {esc(n.get("date"))}, {esc(n.get("tone"))})</span></div>'
        for n in d.get("news", []))
    return f"""
<div class="card" id="{tkr}">
  <div class="row">
    <h3><a href="disc_{tkr}.html">{tkr}</a> · {esc(d.get('name'))}</h3>
    <span class="badge" style="background:{color}">{esc(fit)} FIT</span>
    <span class="tier">{esc(TIER_LABEL.get(d.get('tier'), d.get('tier')))}</span>
  </div>
  <div style="color:var(--mut);font-size:12.5px;margin-top:6px">{esc(d.get('valuation'))}</div>
  <div class="metrics">{mh}</div>
  <p class="thesis"><b>Thesis.</b> {esc(d.get('thesis'))}</p>
  <p class="risk"><b>Key risk.</b> {esc(d.get('key_risk'))}</p>
  <ul class="sig">{sigs}</ul>
  <div class="news">{news}</div>
</div>"""


def detail_html(d, asof, disclaimer):
    tkr = esc(d.get("ticker"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{tkr} — Early Discovery</title><style>{CSS}</style></head>
<body><div class="wrap">
<p class="sub"><a href="index.html">← back to Early Discovery</a> · as of {esc(asof)}</p>
{card_html(d)}
<div class="disc">{esc(disclaimer)}</div>
<div class="foot">Source: Bigdata.com live discovery sweep. Speculative research — not advice.</div>
</div></body></html>"""


def build():
    if not os.path.exists(DISCOVER):
        raise SystemExit(f"{DISCOVER} not found — nothing to render.")
    data = json.load(open(DISCOVER))
    asof = data.get("_asof", "")
    disclaimer = data.get("_disclaimer", "")
    discoveries = sorted(data.get("discoveries", []), key=disc_sort_key)
    os.makedirs(OUT_DIR, exist_ok=True)

    cards = "".join(card_html(d) for d in discoveries)
    n_strong = sum(1 for d in discoveries if d.get("early_fit") == "STRONG")

    # optional large-cap breakout section
    ideas_html = ""
    if os.path.exists(IDEAS):
        try:
            ideas = json.load(open(IDEAS)).get("ideas", [])
            rows = "".join(
                f"<tr><td>{esc(i.get('ticker'))}</td><td>{esc(i.get('name',''))}</td>"
                f"<td>{esc(i.get('verdict',''))}</td><td>{esc(i.get('note',''))}</td></tr>"
                for i in ideas)
            if rows:
                ideas_html = (f'<h2>Large-cap breakout watch</h2>'
                              f'<table><tr><th>Ticker</th><th>Name</th><th>Verdict</th><th>Note</th></tr>{rows}</table>')
        except Exception:
            ideas_html = ""

    index = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Early Discovery — pre-hype small-caps</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>Early Discovery — pre-hype small &amp; mid-caps</h1>
<p class="sub">Live-discovered via Bigdata.com · as of {esc(asof)} · {len(discoveries)} names ({n_strong} STRONG) ·
ranked STRONG → WEAK, EMERGING before SPECULATIVE_EARLY</p>
<div class="disc">{esc(disclaimer)}</div>
<h2>Discoveries</h2>
{cards}
{ideas_html}
<div class="foot">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} by breakout_site.py ·
Source: Bigdata.com live discovery sweep (small/mid-cap, pre-hype).</div>
</div></body></html>"""

    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(index)
    for d in discoveries:
        with open(os.path.join(OUT_DIR, f"disc_{d.get('ticker')}.html"), "w") as f:
            f.write(detail_html(d, asof, disclaimer))

    print(f"Wrote {OUT_DIR}/index.html + {len(discoveries)} detail pages "
          f"({n_strong} STRONG, {len(discoveries)-n_strong} other).")


if __name__ == "__main__":
    build()
