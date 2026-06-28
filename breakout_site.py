#!/usr/bin/env python3
"""breakout_site.py — build the standalone multi-page BREAKOUT SCOUT research site.

Reads breakout_ideas.json (Stage-B live research) + breakout_shortlist.json (Stage-A screen +
per-sector cohorts) and writes breakout_site/:
  index.html        — ranked CONFIRM -> CAUTION -> VETO overview, full disclaimer, fingerprint note
  <TICKER>.html      — one full dossier per researched name: price chart, fingerprint scorecard,
                       thesis/key-risk/verdict, analyst panel, dated news, competitor table, and the
                       ★ same-sector beaten-down COHORT table answering 'why this and not its peers'.

Self-contained HTML (no server, no CDN) so it commits cleanly and renders offline.
Run: python3 breakout_site.py
"""
from __future__ import annotations
import json, os, html
import pandas as pd

OUT = "breakout_site"

# friendly labels + formatters for the learned-fingerprint feature vector
FEAT_META = {
    "dd_from_high": ("Drawdown from 1y high", "pct"),
    "base_pos":     ("Position in 52w range", "pct"),
    "ret_1m":       ("1-month return",        "pct"),
    "ret_3m":       ("3-month return",        "pct"),
    "accel":        ("Momentum acceleration", "num"),
    "above_ma50":   ("Price vs 50d MA",       "pct"),
    "above_ma200":  ("Price vs 200d MA",      "pct"),
    "ma50_slope":   ("50d MA slope (curl)",   "pct"),
    "vol_surge":    ("Volume surge (20/63d)", "x"),
    "base_age":     ("Base age (since low)",  "ratio"),
}
VCLASS = {"CONFIRM": "confirm", "CAUTION": "caution", "VETO": "veto"}
TONE = {"positive": "pos", "neutral": "neu", "negative": "neg"}


def _fmt(v, kind):
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v*100:+.1f}%"
    if kind == "x":
        return f"{v:.2f}x"
    if kind == "ratio":
        return f"{v:.2f}"
    return f"{v:+.2f}"


def price_chart(ticker, w=860, h=240):
    """A 1-year close chart from the bot's on-disk bars, drawn as an inline SVG."""
    p = f"bars_{ticker}_1d.parquet"
    if not os.path.exists(p):
        return '<div class=note>no local price history on disk for this name</div>'
    d = pd.read_parquet(p)[["ts", "close"]].dropna()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    d = d.sort_values("ts").tail(252)
    v = d["close"].values
    if len(v) < 5:
        return '<div class=note>insufficient price history</div>'
    lo, hi = float(v.min()), float(v.max())
    rng = (hi - lo) or 1.0
    pad = 30
    pts = [(pad + (w - 2 * pad) * i / (len(v) - 1), h - pad - (h - 2 * pad) * (x - lo) / rng)
           for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    hi_y = h - pad - (h - 2 * pad) * (hi - lo) / rng
    lo_y = h - pad - (h - 2 * pad) * (lo - lo) / rng
    last = v[-1]
    last_y = h - pad - (h - 2 * pad) * (last - lo) / rng
    d0, d1 = d["ts"].iloc[0].date(), d["ts"].iloc[-1].date()
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" class=chart preserveAspectRatio="none">'
        f'<defs><linearGradient id=cg x1=0 x2=0 y1=0 y2=1>'
        f'<stop offset=0 stop-color="#00e5ff" stop-opacity=.30/><stop offset=1 stop-color="#00e5ff" stop-opacity=0/></linearGradient></defs>'
        f'<line x1={pad} y1={hi_y:.1f} x2={w-pad} y2={hi_y:.1f} stroke="rgba(255,45,146,.35)" stroke-dasharray="4 4"/>'
        f'<line x1={pad} y1={lo_y:.1f} x2={w-pad} y2={lo_y:.1f} stroke="rgba(29,255,176,.35)" stroke-dasharray="4 4"/>'
        f'<polygon points="{pad},{h-pad} {line} {w-pad},{h-pad}" fill="url(#cg)"/>'
        f'<polyline fill=none stroke="#00e5ff" stroke-width=2 points="{line}"/>'
        f'<circle cx={pts[-1][0]:.1f} cy={last_y:.1f} r=4 fill="#00e5ff"/>'
        f'<text x={pad} y={hi_y-6:.1f} class=cax fill="#ff2d92">1y high {hi:.2f}</text>'
        f'<text x={pad} y={lo_y+14:.1f} class=cax fill="#1dffb0">1y low {lo:.2f}</text>'
        f'<text x={w-pad} y={last_y-8:.1f} class=cax text-anchor=end fill="#bfe9ff">last {last:.2f}</text>'
        f'<text x={pad} y={h-8} class=cax fill="#5a7390">{d0}</text>'
        f'<text x={w-pad} y={h-8} class=cax text-anchor=end fill="#5a7390">{d1}</text>'
        f'</svg>'
        f'<div class=note>1-year close from the bot\'s on-disk bars (structural screen). Live quote/targets are from Bigdata.com.</div>'
    )


CSS = """
:root{--bg:#04070f;--gl:rgba(12,22,40,.55);--ln:rgba(0,229,255,.18);--cy:#00e5ff;--mg:#ff2d92;--gr:#1dffb0;--am:#ffcf3a;--tx:#bfe9ff;--dim:#5a7390}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--tx);font:13.5px/1.6 'SF Mono',ui-monospace,Menlo,monospace;max-width:920px;margin:0 auto;padding:30px 20px 90px;
background-image:linear-gradient(rgba(0,229,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.04) 1px,transparent 1px),radial-gradient(1000px 500px at 50% -10%,rgba(0,229,255,.10),transparent);
background-size:44px 44px,44px 44px,100% 100%}
a{color:var(--cy);text-decoration:none}
.hd{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ln);padding-bottom:14px;flex-wrap:wrap;gap:8px}
.lg{font-size:17px;letter-spacing:5px;font-weight:700;color:#fff;text-shadow:0 0 14px var(--cy)}.lg span{color:var(--cy)}
.as{font-size:10px;letter-spacing:2px;color:var(--dim)}
.glass{background:var(--gl);border:1px solid var(--ln);border-radius:14px;backdrop-filter:blur(8px);box-shadow:0 0 30px -14px var(--cy);padding:18px 20px;margin-top:16px}
.disc{border-color:var(--am);color:var(--am);font-size:11.5px;line-height:1.7}
.sec-h{font-size:10px;letter-spacing:4px;color:var(--cy);text-transform:uppercase;margin:30px 0 6px;display:flex;align-items:center;gap:10px;text-shadow:0 0 8px var(--cy)}
.sec-h::before{content:"//"}.sec-h::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--ln),transparent)}
.card{display:block;margin-top:14px;padding:16px 20px;border-radius:14px;background:var(--gl);border:1px solid var(--ln);transition:.15s;position:relative;overflow:hidden}
.card:hover{border-color:var(--cy);box-shadow:0 0 26px -10px var(--cy);transform:translateY(-1px)}
.card .row1{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.tk{font-size:22px;font-weight:700;color:#fff;letter-spacing:1px;min-width:74px}
.vd{font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 11px;border-radius:20px;border:1px solid}
.vd.confirm{color:var(--gr);border-color:var(--gr);box-shadow:0 0 12px -3px var(--gr)}
.vd.caution{color:var(--am);border-color:var(--am)}.vd.veto{color:var(--mg);border-color:var(--mg)}
.up{margin-left:auto;font-size:19px;font-weight:700;font-variant-numeric:tabular-nums}
.up.g{color:var(--gr)}.up.m{color:var(--mg)}
.meta{color:var(--dim);font-size:11.5px;margin-top:4px;letter-spacing:.3px}
.th{margin-top:10px;font-size:13px;color:var(--tx)}
.kr{margin-top:8px;font-size:12px;color:var(--am)}.kr b{color:var(--am)}
.pill{display:inline-block;font-size:10px;color:var(--dim);border:1px solid var(--ln);border-radius:12px;padding:2px 9px;margin:3px 5px 0 0}
table{width:100%;border-collapse:collapse;margin-top:6px}
th{font-size:9px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;text-align:left;padding:9px 10px;border-bottom:1px solid var(--ln);font-weight:500}
td{padding:9px 10px;border-bottom:1px solid rgba(0,229,255,.06);font-size:12.5px;font-variant-numeric:tabular-nums}
tr:last-child td{border:none}
td.tkc{font-weight:700;color:#fff}.g{color:var(--gr)}.m{color:var(--mg)}.dim{color:var(--dim)}
tr.self td{background:linear-gradient(90deg,rgba(0,229,255,.12),transparent)}
.matchbar{display:block;height:7px;border-radius:4px;background:linear-gradient(90deg,var(--mg),var(--cy));box-shadow:0 0 10px -2px var(--cy)}
.matchwrap{background:rgba(0,229,255,.08);border-radius:4px;width:120px;height:7px;display:inline-block;vertical-align:middle}
.score{font-size:11px;color:var(--cy)}
.chart{display:block;margin-top:6px}.cax{font-size:10px;font-family:'SF Mono',monospace}
.news{margin-top:6px}.ni{padding:9px 0;border-bottom:1px solid rgba(0,229,255,.06)}.ni:last-child{border:none}
.nh{color:var(--tx);font-size:12.5px}.nm{color:var(--dim);font-size:10.5px;margin-top:2px}
.tone{font-size:9px;font-weight:700;letter-spacing:1px;padding:1px 7px;border-radius:10px;border:1px solid;margin-right:7px}
.tone.pos{color:var(--gr);border-color:var(--gr)}.tone.neu{color:var(--am);border-color:var(--am)}.tone.neg{color:var(--mg);border-color:var(--mg)}
.back{font-size:11px;letter-spacing:2px;color:var(--cy);border:1px solid var(--cy);border-radius:18px;padding:4px 12px}
.ft{margin-top:44px;border-top:1px solid var(--ln);padding-top:16px;color:var(--dim);font-size:10px;letter-spacing:.8px;line-height:1.8;text-align:center}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-top:6px}
.kvi{padding:10px 12px;border:1px solid var(--ln);border-radius:10px;background:rgba(0,0,0,.15)}
.kvi .l{font-size:9px;letter-spacing:1px;color:var(--dim);text-transform:uppercase}
.kvi .v{font-size:17px;font-weight:700;margin-top:3px;color:#fff}
"""


def verdict_badge(v):
    return f'<span class="vd {VCLASS.get(v,"")}">{v}</span>'


def analyst_panel(idea):
    r = idea["ratings"]
    eps = idea.get("eps_trend", [])
    eps_txt = " → ".join(f"{e:.2f}" for e in eps) if eps else "—"
    trend = "rising ▲" if idea.get("eps_rising") else "flat/falling ▼"
    tcls = "g" if idea.get("eps_rising") else "m"
    up = idea["upside_pct"]
    return (
        '<div class=kv>'
        f'<div class=kvi><div class=l>Price</div><div class=v>${idea["price"]:.2f}</div></div>'
        f'<div class=kvi><div class=l>Consensus target</div><div class=v>${idea["target_consensus"]:.2f}</div></div>'
        f'<div class=kvi><div class=l>Upside</div><div class="v {"g" if up>0 else "m"}">{up:+.1f}%</div></div>'
        f'<div class=kvi><div class=l>Earnings surprise</div><div class="v {"g" if idea["earnings_surprise_pct"]>0 else "m"}">{idea["earnings_surprise_pct"]:+.1f}%</div></div>'
        f'<div class=kvi><div class=l>Net margin</div><div class=v>{idea["margin_pct"]:.0f}%</div></div>'
        f'<div class=kvi><div class=l>ROE</div><div class=v>~{idea["roe_pct"]:.0f}%</div></div>'
        '</div>'
        '<table><tr><th>Strong Buy</th><th>Buy</th><th>Hold</th><th>Sell</th><th>Consensus</th></tr>'
        f'<tr><td class=g>{r["strong_buy"]}</td><td class=g>{r["buy"]}</td><td class=dim>{r["hold"]}</td>'
        f'<td class=m>{r["sell"]}</td><td class=tkc>{r["consensus"]}</td></tr></table>'
        f'<div class=meta style="margin-top:8px">Forward EPS (next quarters): <b>{eps_txt}</b> '
        f'— <span class={tcls}>{trend}</span></div>'
    )


def fingerprint_scorecard(short_row):
    if not short_row:
        return '<div class=note>not in the Stage-A structural shortlist (research-added name)</div>'
    bd = short_row["breakdown"]
    rows = ""
    for k, (label, kind) in FEAT_META.items():
        if k not in bd:
            continue
        cell = bd[k]
        m = cell["match"]
        rows += (
            f'<tr><td>{label}</td>'
            f'<td class=tkc>{_fmt(cell["value"], kind)}</td>'
            f'<td class=dim>{_fmt(cell["target"], kind)}</td>'
            f'<td><span class=matchwrap><span class=matchbar style="width:{m*100:.0f}%"></span></span> '
            f'<span class=score>{m*100:.0f}%</span></td></tr>'
        )
    return (
        f'<div class=meta>Fingerprint match score: <b class=score>{short_row["score"]:.1f}/100</b> '
        f'(weighted similarity to the learned blow-up centroid). Higher = closer to the shape a name '
        f'has had at the bar it began a major run.</div>'
        '<table><tr><th>Feature</th><th>This name</th><th>Fingerprint target</th><th>Match</th></tr>'
        f'{rows}</table>'
    )


def cohort_table(sector, cohorts, self_tk):
    grp = cohorts.get(sector, [])
    if not grp:
        return '<div class=note>no same-sector cohort on the screen this cycle</div>'
    rows = ""
    for c in grp:
        cls = " class=self" if c["ticker"] == self_tk else ""
        star = " ★" if c["ticker"] == self_tk else ""
        rows += (
            f'<tr{cls}><td class=tkc>{c["ticker"]}{star}</td>'
            f'<td class=score>{c["score"]:.1f}</td>'
            f'<td class=m>{c["dd_from_high_pct"]:.0f}%</td>'
            f'<td>{c["base_pos_pct"]:.0f}%</td>'
            f'<td class="{"g" if c["above_ma50_pct"]>0 else "m"}">{c["above_ma50_pct"]:+.0f}%</td>'
            f'<td>{c["vol_surge"]:.2f}x</td>'
            f'<td class=dim>{c["kind"]}</td></tr>'
        )
    return (
        f'<div class=meta>Every beaten-down <b>{sector}</b> name the Stage-A screen surfaced this '
        f'cycle, ranked by fingerprint match. ★ = this stock. This is the "why this one and not its '
        f'peers" set — a higher match means a shape closer to the pre-run base.</div>'
        '<table><tr><th>Ticker</th><th>Match</th><th>DD from high</th><th>Base pos</th>'
        '<th>vs 50d MA</th><th>Vol surge</th><th>Type</th></tr>'
        f'{rows}</table>'
    )


def competitor_table(idea):
    rows = ""
    for c in idea.get("competitors", []):
        pe = f'{c["fwd_pe"]:.1f}x' if c.get("fwd_pe") else "n/a"
        rows += (
            f'<tr><td class=tkc>{c["ticker"]}</td><td class=dim>{html.escape(c["name"])}</td>'
            f'<td>{pe}</td><td>{c["rev_growth_pct"]:+.0f}%</td><td>{c["margin_pct"]:.0f}%</td>'
            f'<td class=dim>{html.escape(c.get("note",""))}</td></tr>'
        )
    return (
        '<table><tr><th>Peer</th><th>Name</th><th>Fwd P/E</th><th>Rev growth</th><th>Margin</th><th>Note</th></tr>'
        f'{rows}</table>'
        f'<div class=th style="margin-top:8px"><b>Edge vs peers:</b> {html.escape(idea.get("competitor_edge",""))}</div>'
    )


def dossier_page(idea, short_row, cohorts, asof):
    tk = idea["ticker"]
    up = idea["upside_pct"]
    news = ""
    for n in idea.get("news", []):
        tone = TONE.get(n.get("tone", "neutral"), "neu")
        news += (
            f'<div class=ni><div class=nh><span class="tone {tone}">{n.get("tone","neutral").upper()}</span>'
            f'{html.escape(n["headline"])}</div>'
            f'<div class=nm>{html.escape(n.get("source",""))} · {n.get("date","")}</div></div>'
        )
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>BREAKOUT :: {tk}</title>
<style>{CSS}</style></head><body>
<div class=hd><div class=lg>BREAKOUT<span>::</span>{tk}</div>
<a class=back href="index.html">◂ ALL IDEAS</a></div>

<div class="glass" style="margin-top:18px">
  <div class=row1><div class=tk>{tk}</div>{verdict_badge(idea["verdict"])}
    <span class=pill>{idea["sector"]}</span><span class=pill>stage: {idea["stage"]}</span>
    <span class="up {'g' if up>0 else 'm'}">{up:+.1f}% to target</span></div>
  <div class=meta>${idea["price"]:.2f} → ${idea["target_consensus"]:.2f} consensus · {idea["dd_from_high_pct"]:.0f}% off 52w high · base position {idea["base_pos_pct"]:.0f}% of range</div>
</div>

<div class=sec-h>Price · 52-week structure</div>
<div class="glass">{price_chart(tk)}</div>

<div class=sec-h>Thesis &amp; verdict</div>
<div class="glass">
  <div class=th>{html.escape(idea["thesis"])}</div>
  <div class=kr><b>KEY RISK —</b> {html.escape(idea["key_risk"])}</div>
</div>

<div class=sec-h>Blow-up fingerprint scorecard</div>
<div class="glass">{fingerprint_scorecard(short_row)}</div>

<div class=sec-h>Analyst panel · estimates · profitability</div>
<div class="glass">{analyst_panel(idea)}</div>

<div class=sec-h>Recent news (dated, sourced)</div>
<div class="glass"><div class=news>{news or '<div class=note>no news pulled</div>'}</div></div>

<div class=sec-h>Competitors — growth · margin · valuation</div>
<div class="glass">{competitor_table(idea)}</div>

<div class=sec-h>★ Same-sector beaten-down cohort — why this one?</div>
<div class="glass">{cohort_table(idea["sector"], cohorts, tk)}</div>

<div class=ft>BREAKOUT SCOUT :: SPECULATIVE conviction research, NOT a backtested edge (cf the reversal edge p=0.000)<br>
Live Bigdata.com snapshot, not point-in-time · individual-stock survivorship applies · research leads, manual trades only · generated for {asof}</div>
</body></html>"""
    open(os.path.join(OUT, f"{tk}.html"), "w").write(doc)


def index_page(ideas, pat, asof, generated):
    order = {"CONFIRM": 0, "CAUTION": 1, "VETO": 2}
    ideas_sorted = sorted(ideas, key=lambda x: (order.get(x["verdict"], 9), -x["upside_pct"]))
    lead = next((i for i in ideas_sorted if i["verdict"] == "CONFIRM"), None)

    def card(idea):
        tk = idea["ticker"]; up = idea["upside_pct"]
        return (
            f'<a class=card href="{tk}.html"><div class=row1>'
            f'<span class=tk>{tk}</span>{verdict_badge(idea["verdict"])}'
            f'<span class=pill>{idea["sector"]}</span>'
            f'<span class="up {"g" if up>0 else "m"}">{up:+.1f}%</span></div>'
            f'<div class=meta>${idea["price"]:.2f} → ${idea["target_consensus"]:.2f} · {idea["dd_from_high_pct"]:.0f}% off high · '
            f'{idea["ratings"]["consensus"]} ({idea["ratings"]["buy"]+idea["ratings"]["strong_buy"]} buy / {idea["ratings"]["hold"]} hold / {idea["ratings"]["sell"]} sell) · '
            f'EPS surprise {idea["earnings_surprise_pct"]:+.0f}%</div>'
            f'<div class=th>{html.escape(idea["thesis"][:240])}…</div>'
            f'<div class=kr><b>Risk:</b> {html.escape(idea["key_risk"][:150])}…</div></a>'
        )

    confirms = [card(i) for i in ideas_sorted if i["verdict"] == "CONFIRM"]
    cautions = [card(i) for i in ideas_sorted if i["verdict"] == "CAUTION"]
    vetos = [card(i) for i in ideas_sorted if i["verdict"] == "VETO"]
    lead_block = ""
    if lead:
        lead_block = (
            '<div class=sec-h>Lead conviction</div>'
            f'<div class="glass" style="border-color:var(--gr);box-shadow:0 0 30px -12px var(--gr)">'
            f'<div class=row1><span class=tk>{lead["ticker"]}</span>{verdict_badge("CONFIRM")}'
            f'<span class="up g">{lead["upside_pct"]:+.1f}% to target</span></div>'
            f'<div class=th>{html.escape(lead["thesis"])}</div>'
            f'<div class=kr><b>KEY RISK —</b> {html.escape(lead["key_risk"])}</div>'
            f'<div class=meta style="margin-top:8px"><a href="{lead["ticker"]}.html">▸ full {lead["ticker"]} dossier</a></div></div>'
        )

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>BREAKOUT SCOUT</title>
<style>{CSS}</style></head><body>
<div class=hd><div class=lg>BREAKOUT<span>::</span>SCOUT</div>
<div class=as>WEEKLY HIGH-POTENTIAL IDEAS · DATA {asof} · GEN {generated}</div></div>

<div class="glass disc">⚠ SPECULATIVE conviction research, NOT a backtested edge (cf the reversal edge p=0.000).
Research leads, not signals. Manual trades only. Live Bigdata.com snapshot (quotes ~prior close), not point-in-time.
Individual-stock survivorship applies — the screen's universe is today's survivors, so the learned pattern is optimistic.
Verify every name yourself before risking capital.</div>

<div class="glass" style="font-size:11.5px;color:var(--dim)">
<b style="color:var(--cy)">How this works.</b> Stage A screens every beaten-down name on disk against a
<b>learned blow-up fingerprint</b> ({html.escape(pat.get("_method",""))}). Stage B then researches the top names
live on Bigdata.com — price, analyst ratings &amp; targets, forward-EPS trend, earnings surprise, profitability,
dated news, and a competitor comparison — and assigns a CONFIRM / CAUTION / VETO. A VETO on a name that does not
fit the mandate is a deliberate, valuable output, not a failure.</div>

{lead_block}

<div class=sec-h>✅ CONFIRM — beaten down, still executing</div>
{''.join(confirms) or '<div class=note>none this cycle</div>'}

<div class=sec-h>⚠ CAUTION — turnaround unproven</div>
{''.join(cautions) or '<div class=note>none this cycle</div>'}

<div class=sec-h>⛔ VETO — flagged but doesn\'t fit the mandate</div>
{''.join(vetos) or '<div class=note>none this cycle</div>'}

<div class=ft>BREAKOUT SCOUT :: a discretionary research lead generator layered on the quant bot ::
the validated edge in this bot is the reversal array (p&lt;0.01); THIS is conviction research, not a signal ::
NEVER auto-executes · manual fills only</div>
</body></html>"""
    open(os.path.join(OUT, "index.html"), "w").write(doc)


def main():
    ideas_doc = json.load(open("breakout_ideas.json"))
    short = json.load(open("breakout_shortlist.json"))
    ideas = ideas_doc["ideas"]
    cohorts = short.get("cohorts", {})
    pat = short.get("pattern", {})
    asof = ideas_doc.get("_asof", short.get("_asof", ""))
    generated = short.get("_generated", asof)
    short_by_tk = {r["ticker"]: r for r in short.get("shortlist", [])}

    os.makedirs(OUT, exist_ok=True)
    for idea in ideas:
        dossier_page(idea, short_by_tk.get(idea["ticker"]), cohorts, asof)
    index_page(ideas, pat, asof, generated)
    print(f"wrote {OUT}/index.html + {len(ideas)} dossier pages: "
          f"{', '.join(i['ticker'] for i in ideas)}")
    print(f"verdicts: " + ", ".join(f"{i['ticker']}={i['verdict']}" for i in ideas))


if __name__ == "__main__":
    main()
