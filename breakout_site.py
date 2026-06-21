#!/usr/bin/env python3
"""breakout_site.py — builds the standalone multi-page BREAKOUT SCOUT research site.

Reads breakout_shortlist.json (Stage-A fingerprint scorecards + sector cohorts) and
breakout_ideas.json (Stage-B live Bigdata.com research) and writes breakout_site/:
  - index.html            : ranked CONFIRM -> CAUTION -> VETO board
  - <TICKER>.html          : one full dossier per researched name — price chart, fingerprint
                             scorecard, thesis/risk, analyst panel, dated news, competitor
                             table, and the ★ same-sector beaten-down cohort ('why this one').

Self-contained HTML (no server / no CDN), so it commits cleanly and renders offline.
⚠ SPECULATIVE research, NOT a backtested edge. Research leads, not signals.

Run: python3 breakout_site.py
"""
from __future__ import annotations
import json, os, html
import pandas as pd

from early_scanner import load_ohlcv

OUT = "breakout_site"
SHORTLIST = "breakout_shortlist.json"
IDEAS = "breakout_ideas.json"

VCOLOR = {"CONFIRM": "var(--gr)", "CAUTION": "var(--am)", "VETO": "var(--mg)"}
VRANK = {"CONFIRM": 0, "CAUTION": 1, "VETO": 2}
TONE = {"positive": "var(--gr)", "neutral": "var(--dim)", "negative": "var(--mg)"}
FEAT_LABEL = {
    "dd_from_high": "Drawdown depth", "base_position": "52w-range base",
    "downtrend_stall": "Freefall stalled", "vol_compression": "Volatility coil",
    "capitulation_vol": "Capitulation volume", "early_turn": "Early upturn",
}

CSS = """
:root{--bg:#04070f;--gl:rgba(12,22,40,.55);--ln:rgba(0,229,255,.18);--cy:#00e5ff;--mg:#ff2d92;--gr:#1dffb0;--am:#ffcf3a;--tx:#bfe9ff;--dim:#5a7390}
*{box-sizing:border-box;margin:0}
a{color:var(--cy);text-decoration:none}
body{background:var(--bg);color:var(--tx);font:13.5px/1.6 'SF Mono',ui-monospace,Menlo,monospace;max-width:920px;margin:0 auto;padding:28px 20px 90px;
background-image:linear-gradient(rgba(0,229,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.045) 1px,transparent 1px),radial-gradient(1000px 500px at 50% -10%,rgba(0,229,255,.1),transparent),radial-gradient(800px 400px at 100% 8%,rgba(255,45,146,.07),transparent);background-size:42px 42px,42px 42px,100% 100%,100% 100%}
.hd{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ln);padding-bottom:14px;flex-wrap:wrap;gap:8px}
.lg{font-size:17px;letter-spacing:5px;font-weight:700;color:#fff;text-shadow:0 0 14px var(--cy)}.lg span{color:var(--cy)}
.as{font-size:10px;letter-spacing:2px;color:var(--dim)}
.glass{background:var(--gl);border:1px solid var(--ln);border-radius:14px;backdrop-filter:blur(8px);box-shadow:0 0 34px -14px var(--cy),inset 0 1px 0 rgba(255,255,255,.05)}
.disc{padding:13px 18px;margin:16px 0;border-color:var(--am);color:var(--am);font-size:11px;letter-spacing:.4px}
.sec-h{font-size:10px;letter-spacing:4px;color:var(--cy);text-transform:uppercase;margin:30px 0 12px;display:flex;align-items:center;gap:10px;text-shadow:0 0 8px var(--cy)}
.sec-h::before{content:"//"}.sec-h::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--ln),transparent)}
.vd{font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 11px;border-radius:20px;border:1px solid;white-space:nowrap}
table{width:100%;border-collapse:collapse}
th{font-size:9px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;text-align:left;padding:9px 11px;border-bottom:1px solid var(--ln);font-weight:500}
td{padding:10px 11px;border-bottom:1px solid rgba(0,229,255,.06);font-size:13px;vertical-align:middle}
tr:last-child td{border:none}
td.tk{font-weight:700;color:#fff;letter-spacing:1px}td.tk a{color:#fff}
td.sec{color:var(--dim);font-size:11px;font-style:italic}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--gr)}.neg{color:var(--mg)}
.row-link td{cursor:pointer}.row-link:hover td{background:linear-gradient(90deg,rgba(0,229,255,.10),transparent)}
.star td{background:linear-gradient(90deg,rgba(29,255,176,.12),transparent)}
.note{color:var(--dim);font-size:11px;margin-top:9px;letter-spacing:.3px;line-height:1.6}
.ft{margin-top:46px;border-top:1px solid var(--ln);padding-top:16px;color:var(--dim);font-size:10px;letter-spacing:1px;line-height:1.9;text-align:center}
/* dossier */
.crumb{font-size:10px;letter-spacing:2px;color:var(--dim);margin-bottom:18px}
.tophdr{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;margin-top:6px}
.tname{font-size:30px;font-weight:700;color:#fff;letter-spacing:2px;text-shadow:0 0 16px var(--cy)}
.tsub{color:var(--dim);font-size:12px;letter-spacing:1px;margin-top:2px}
.tprice{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;color:var(--cy);text-shadow:0 0 14px var(--cy);text-align:right}
.chartwrap{padding:16px 18px 10px;margin-top:16px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px}
.kpi{padding:14px 12px;text-align:center}
.kpi .l{font-size:8.5px;letter-spacing:1.5px;color:var(--dim);text-transform:uppercase}
.kpi .v{font-size:21px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media(max-width:680px){.grid2{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
.panel{padding:16px 18px}
.panel h3{font-size:10px;letter-spacing:3px;color:var(--cy);text-transform:uppercase;margin-bottom:12px;text-shadow:0 0 8px var(--cy)}
.scrow{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:11.5px}
.scrow .lab{width:148px;color:var(--dim)}
.bar{flex:1;height:8px;border-radius:5px;background:rgba(255,255,255,.06);overflow:hidden}
.bar i{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,var(--mg),var(--cy));box-shadow:0 0 12px -2px var(--cy)}
.scrow .pct{width:34px;text-align:right;color:var(--tx);font-variant-numeric:tabular-nums}
.fp-score{font-size:13px;color:var(--dim);margin-top:6px}.fp-score b{color:var(--gr);font-size:18px}
.thesis{font-size:13.5px;line-height:1.7;color:var(--tx)}
.risk{font-size:13px;line-height:1.7;color:var(--am)}
.ratings{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 2px}
.pill{font-size:10px;padding:3px 9px;border-radius:14px;border:1px solid var(--ln);color:var(--dim)}
.pill b{color:var(--tx)}
.news-it{padding:10px 0;border-bottom:1px solid rgba(0,229,255,.06)}.news-it:last-child{border:none}
.news-h{font-size:12.5px;color:var(--tx);line-height:1.5}
.news-m{font-size:10px;color:var(--dim);letter-spacing:.5px;margin-top:3px}
.tone-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
.epsbar{display:flex;align-items:flex-end;gap:6px;height:64px;margin-top:6px}
.epsbar .b{flex:1;background:linear-gradient(180deg,var(--cy),rgba(0,229,255,.15));border-radius:3px 3px 0 0;position:relative;min-height:4px}
.epsbar .b span{position:absolute;top:-15px;left:0;right:0;text-align:center;font-size:9px;color:var(--dim)}
"""


def price_chart(sym, hi, lo, w=860, h=220, days=520):
    d = load_ohlcv(sym)
    if d is None or len(d) < 30:
        return '<div class=note>no price history on disk for chart</div>'
    s = d["close"].dropna().tail(days)
    v = s.values; n = len(v)
    pmin, pmax = float(v.min()), float(v.max()); rng = (pmax - pmin) or 1
    def X(i): return 8 + (w - 16) * i / (n - 1)
    def Y(p): return 12 + (h - 40) * (1 - (p - pmin) / rng)
    pts = " ".join(f"{X(i):.1f},{Y(p):.1f}" for i, p in enumerate(v))
    last = float(v[-1])
    # 52w hi/lo guide lines (if within view)
    guides = ""
    for val, col, lab in [(hi, "var(--dim)", f"52w hi {hi:.0f}"), (lo, "var(--am)", f"52w lo {lo:.0f}")]:
        if pmin <= val <= pmax:
            y = Y(val)
            guides += (f'<line x1=8 x2={w-8} y1={y:.1f} y2={y:.1f} stroke="{col}" stroke-width=1 '
                       f'stroke-dasharray="4 5" opacity=.5/>'
                       f'<text x={w-10} y={y-4:.1f} fill="{col}" font-size=9 text-anchor=end>{lab}</text>')
    dotc = "var(--gr)"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'<defs><linearGradient id=g x1=0 x2=0 y1=0 y2=1><stop offset=0 stop-color="var(--cy)" stop-opacity=.30/>'
            f'<stop offset=1 stop-color="var(--cy)" stop-opacity=0/></linearGradient></defs>'
            f'<polygon points="{X(0):.1f},{h-28} {pts} {X(n-1):.1f},{h-28}" fill="url(#g)"/>'
            f'<polyline points="{pts}" fill=none stroke="var(--cy)" stroke-width=2 '
            f'style="filter:drop-shadow(0 0 6px var(--cy))"/>'
            f'{guides}'
            f'<circle cx={X(n-1):.1f} cy={Y(last):.1f} r=4 fill="{dotc}" style="filter:drop-shadow(0 0 6px {dotc})"/>'
            f'</svg>'
            f'<div class=note style="text-align:center">~{n} sessions of daily closes (bot price dataset) · '
            f'last {last:.2f} · 52w range {lo:.0f}–{hi:.0f}. Analyst panel & news are live Bigdata.com.</div>')


def scorecard(sc, score):
    rows = ""
    for k in ["dd_from_high", "base_position", "downtrend_stall", "vol_compression", "capitulation_vol", "early_turn"]:
        pv = sc.get(k, 0)
        rows += (f'<div class=scrow><div class=lab>{FEAT_LABEL[k]}</div>'
                 f'<div class=bar><i style="width:{pv:.0f}%"></i></div>'
                 f'<div class=pct>{pv:.0f}</div></div>')
    return (f'<div class="glass panel"><h3>Blow-up fingerprint</h3>{rows}'
            f'<div class=fp-score>composite match <b>{score:.0f}</b>/100 '
            f'<span style="color:var(--dim)">— how closely the chart matches the beaten-down-then-runs setup</span></div></div>')


def eps_panel(idea):
    tr = idea.get("eps_trend") or []
    if not tr:
        bars = '<div class=note>no forward EPS series</div>'
    else:
        mx = max(tr) or 1
        bars = '<div class=epsbar>' + "".join(
            f'<div class=b style="height:{max(6,t/mx*100):.0f}%"><span>{t:.2f}</span></div>' for t in tr
        ) + '</div>'
    arrow = "▲ rising" if idea.get("eps_rising") else "▼ flat/falling"
    acol = "var(--gr)" if idea.get("eps_rising") else "var(--mg)"
    sp = idea.get("earnings_surprise_pct", 0)
    spcol = "var(--gr)" if sp >= 0 else "var(--mg)"
    return (f'<div class="glass panel"><h3>Forward EPS &amp; earnings</h3>{bars}'
            f'<div class=note>next ~4 quarters · trend <b style="color:{acol}">{arrow}</b> · '
            f'last surprise <b style="color:{spcol}">{sp:+.1f}%</b> · ROE ~{idea.get("roe_pct",0):.0f}% · '
            f'net margin ~{idea.get("margin_pct",0):.0f}%</div></div>')


def analyst_panel(idea):
    r = idea.get("ratings", {})
    up = idea.get("upside_pct", 0)
    upcol = "var(--gr)" if up >= 0 else "var(--mg)"
    pills = "".join(f'<span class=pill>{lab} <b>{r.get(key,0)}</b></span>'
                    for lab, key in [("Strong Buy", "strong_buy"), ("Buy", "buy"),
                                     ("Hold", "hold"), ("Sell", "sell")])
    return (f'<div class="glass panel"><h3>Analyst panel (live)</h3>'
            f'<div class=ratings>{pills}</div>'
            f'<div class=note>consensus <b style="color:var(--tx)">{html.escape(str(r.get("consensus","—")))}</b> · '
            f'target ${idea.get("target_consensus",0):.0f} · '
            f'upside <b style="color:{upcol}">{up:+.0f}%</b> vs ${idea.get("price",0):.2f}</div></div>')


def news_panel(idea):
    items = ""
    for nw in idea.get("news", []):
        c = TONE.get(nw.get("tone", "neutral"), "var(--dim)")
        items += (f'<div class=news-it><div class=news-h>'
                  f'<span class=tone-dot style="background:{c};box-shadow:0 0 7px {c}"></span>'
                  f'{html.escape(nw.get("headline",""))}</div>'
                  f'<div class=news-m>{html.escape(nw.get("source",""))} · {html.escape(nw.get("date",""))} · '
                  f'{html.escape(nw.get("tone",""))}</div></div>')
    return f'<div class="glass panel"><h3>Recent news (Bigdata.com)</h3>{items or "<div class=note>—</div>"}</div>'


def competitor_panel(idea):
    rows = ""
    for c in idea.get("competitors", []):
        pe = c.get("fwd_pe", 0); pe_s = f"{pe:.0f}x" if pe else "—"
        rows += (f'<tr><td class=tk>{html.escape(c.get("ticker",""))}</td>'
                 f'<td class=sec>{html.escape(c.get("name",""))}</td>'
                 f'<td class=num>{pe_s}</td>'
                 f'<td class="num">{c.get("rev_growth_pct",0):+.0f}%</td>'
                 f'<td class=num>{c.get("margin_pct",0):.0f}%</td>'
                 f'<td class=sec>{html.escape(c.get("note",""))}</td></tr>')
    edge = html.escape(idea.get("competitor_edge", ""))
    return (f'<div class="glass panel"><h3>Competitor check</h3>'
            f'<table><tr><th>TICK</th><th>NAME</th><th class=num>FWD P/E</th>'
            f'<th class=num>REV g</th><th class=num>MARGIN</th><th>NOTE</th></tr>{rows}</table>'
            f'<div class=note style="margin-top:10px"><b style="color:var(--gr)">Edge:</b> {edge}</div></div>')


def cohort_panel(ticker, sector, cohorts):
    peers = cohorts.get(sector, [])
    rows = ""
    for p in peers:
        star = " star" if p["ticker"] == ticker else ""
        mark = "★ " if p["ticker"] == ticker else ""
        rows += (f'<tr class="{star.strip()}"><td class=tk>{mark}{p["ticker"]}</td>'
                 f'<td class="num"><b>{p["score"]:.0f}</b></td>'
                 f'<td class=num>${p["price"]:.2f}</td>'
                 f'<td class="num neg">{p["dd_from_high_pct"]:.0f}%</td>'
                 f'<td class=num>{p["base_pos_pct"]:.0f}%</td>'
                 f'<td class="num">{p["ret_6m_pct"]:+.0f}%</td></tr>')
    return (f'<div class="glass panel"><h3>★ Same-sector beaten-down cohort — why this one?</h3>'
            f'<table><tr><th>TICK</th><th class=num>FP SCORE</th><th class=num>PRICE</th>'
            f'<th class=num>DD%</th><th class=num>52w POS</th><th class=num>6M%</th></tr>{rows}</table>'
            f'<div class=note style="margin-top:10px">All <b>{sector}</b> names that screened as beaten-down, '
            f'ranked by fingerprint match. The starred row is this dossier — it is the cohort\'s strongest '
            f'(or most-researched) blow-up pattern, which is the answer to "why this stock and not its peers."</div></div>')


def dossier(idea, sl_map, cohorts, asof):
    t = idea["ticker"]; sl = sl_map.get(t, {})
    hi = sl.get("hi_52w", idea["price"]); lo = sl.get("lo_52w", idea["price"])
    score = sl.get("score", 0); sc = sl.get("scorecard", {})
    vc = VCOLOR.get(idea["verdict"], "var(--dim)")
    dd = idea.get("dd_from_high_pct", 0); bp = idea.get("base_pos_pct", 0); up = idea.get("upside_pct", 0)
    kpis = (
        f'<div class="glass kpi"><div class=l>Verdict</div><div class=v style="color:{vc}">{idea["verdict"]}</div></div>'
        f'<div class="glass kpi"><div class=l>Drawdown</div><div class="v neg">{dd:.0f}%</div></div>'
        f'<div class="glass kpi"><div class=l>52w-range pos</div><div class=v>{bp:.0f}%</div></div>'
        f'<div class="glass kpi"><div class=l>Analyst upside</div><div class=v style="color:{"var(--gr)" if up>=0 else "var(--mg)"}">{up:+.0f}%</div></div>'
    )
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>BREAKOUT SCOUT :: {t}</title><style>{CSS}</style></head><body>
<div class=crumb><a href="index.html">&larr; SCOUT BOARD</a> &nbsp;//&nbsp; {t} DOSSIER</div>
<div class=tophdr>
  <div><div class=tname>{t}</div><div class=tsub>{html.escape(idea.get("sector",""))} · stage {idea.get("stage","")} · fingerprint match {score:.0f}/100</div></div>
  <div><div class=tprice>${idea.get("price",0):.2f}</div><div class=tsub style="text-align:right">
    <span class="vd" style="color:{vc};border-color:{vc}">{idea["verdict"]}</span></div></div>
</div>
<div class="glass chartwrap">{price_chart(t, hi, lo)}</div>
<div class=kpis>{kpis}</div>
<div class=sec-h>Thesis &amp; key risk</div>
<div class="glass panel"><h3>Thesis</h3><div class=thesis>{html.escape(idea.get("thesis",""))}</div>
  <h3 style="margin-top:16px">Key risk</h3><div class=risk>{html.escape(idea.get("key_risk",""))}</div></div>
<div class=sec-h>The numbers</div>
<div class=grid2>{scorecard(sc, score)}{eps_panel(idea)}</div>
<div class=grid2>{analyst_panel(idea)}{news_panel(idea)}</div>
<div class=sec-h>Peers</div>
{competitor_panel(idea)}
<div style="height:16px"></div>
{cohort_panel(t, idea.get("sector",""), cohorts)}
<div class=ft>BREAKOUT SCOUT · {asof} · SPECULATIVE CONVICTION RESEARCH — NOT A BACKTESTED EDGE (cf the reversal edge p=0.000)<br>
Research leads, not signals · Manual trades only · Live snapshot, not point-in-time · Analyst &amp; news data: Bigdata.com</div>
</body></html>"""
    open(os.path.join(OUT, f"{t}.html"), "w").write(doc)


def index_page(ideas, asof, pattern, n_cand):
    # stable sort by verdict only — preserves the deliberate conviction order in breakout_ideas.json
    ordered = sorted(ideas, key=lambda i: VRANK.get(i["verdict"], 9))
    lead = next((i for i in ordered if i["verdict"] == "CONFIRM"), ordered[0])
    rows = ""
    for i in ordered:
        vc = VCOLOR.get(i["verdict"], "var(--dim)")
        up = i.get("upside_pct", 0)
        rows += (f'<tr class=row-link onclick="location.href=\'{i["ticker"]}.html\'">'
                 f'<td class=tk><a href="{i["ticker"]}.html">{i["ticker"]}</a></td>'
                 f'<td class=sec>{html.escape(i.get("sector",""))}</td>'
                 f'<td><span class="vd" style="color:{vc};border-color:{vc}">{i["verdict"]}</span></td>'
                 f'<td class=num>${i.get("price",0):.2f}</td>'
                 f'<td class="num neg">{i.get("dd_from_high_pct",0):.0f}%</td>'
                 f'<td class=num style="color:{"var(--gr)" if up>=0 else "var(--mg)"}">{up:+.0f}%</td>'
                 f'<td class=sec style="font-style:normal;color:var(--tx)">{html.escape(i.get("thesis","")[:118])}…</td></tr>')
    leadc = VCOLOR.get(lead["verdict"], "var(--gr)")
    lead_block = (
        f'<div class="glass panel" style="border-color:{leadc};box-shadow:0 0 34px -10px {leadc}">'
        f'<h3 style="color:{leadc}">◢ Top conviction — {lead["ticker"]}</h3>'
        f'<div class=thesis>{html.escape(lead.get("thesis",""))}</div>'
        f'<div class=note style="margin-top:10px"><a href="{lead["ticker"]}.html">open full {lead["ticker"]} dossier &rarr;</a> · '
        f'verdict {lead["verdict"]} · upside {lead.get("upside_pct",0):+.0f}%</div></div>')
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>BREAKOUT SCOUT :: weekly board</title><style>{CSS}</style></head><body>
<div class=hd><div class=lg>BREAKOUT<span>::</span>SCOUT</div>
  <div class=as>WEEKLY · {asof} · {n_cand} BEATEN-DOWN CANDIDATES · {len(ideas)} RESEARCHED</div></div>
<div class="glass disc">⚠ SPECULATIVE conviction research, NOT a backtested edge (cf the reversal edge p=0.000).
Research leads, not signals. Live snapshot, not point-in-time. Individual-stock survivorship applies. Manual trades only.</div>
<div class=sec-h>Lead idea</div>
{lead_block}
<div class=sec-h>The board · CONFIRM &rarr; CAUTION &rarr; VETO</div>
<div class="glass" style="padding:4px 6px"><table>
<tr><th>TICK</th><th>SECTOR</th><th>VERDICT</th><th class=num>PRICE</th><th class=num>DD%</th><th class=num>UPSIDE</th><th>THESIS</th></tr>
{rows}</table></div>
<div class=note>Each row links to a full dossier: price chart, blow-up fingerprint scorecard, thesis &amp; key risk,
live analyst panel, dated news, a competitor check, and the ★ same-sector beaten-down cohort answering "why this one, not its peers."
Stage-A screen vs fingerprint <b>{html.escape(str(pattern))}</b>; Stage-B research via Bigdata.com.</div>
<div class=ft>BREAKOUT SCOUT · generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} · NEVER AUTO-EXECUTES · MANUAL FILL ONLY<br>
VERDICT = HUMAN CONVICTION CALL, NOT A SIGNAL · A VETO IS A VALUABLE OUTPUT · BACKTEST &ne; PROMISE</div>
</body></html>"""
    open(os.path.join(OUT, "index.html"), "w").write(doc)


def main():
    os.makedirs(OUT, exist_ok=True)
    sl = json.load(open(SHORTLIST))
    ideas_doc = json.load(open(IDEAS))
    ideas = ideas_doc.get("ideas", [])
    asof = ideas_doc.get("_asof", sl.get("_asof", ""))
    sl_map = {r["ticker"]: r for r in sl.get("shortlist", [])}
    cohorts = sl.get("cohorts", {})

    for idea in ideas:
        dossier(idea, sl_map, cohorts, asof)
    index_page(ideas, asof, sl.get("_pattern", "blow-up fingerprint"), sl.get("n_candidates", 0))

    print(f"wrote {OUT}/index.html + {len(ideas)} dossier pages: "
          + ", ".join(i["ticker"] for i in ideas))


if __name__ == "__main__":
    main()
