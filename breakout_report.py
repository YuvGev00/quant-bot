#!/usr/bin/env python3
"""breakout_report.py — illustrated, self-contained HTML report for the Breakout Scout.

Reads breakout_ideas.json (Stage-B enriched theses) + breakout_pattern.json (what Phase 0 learned) and
writes breakout_report.html: a "what we learned from history" header, then one rich card per idea with a
price sparkline, an EPS-estimate-trend sparkline, the 6-factor scorecard, the written thesis + key risk,
a competitor comparison table, analyst ratings, and the dated news evidence. No server/CDN (emails fine).

Run:  python3 breakout_report.py
"""
from __future__ import annotations
import os, json
import pandas as pd
from dashboard import sparkline
from jump_model import load_close

IDEAS = "breakout_ideas.json"
PATTERN = "breakout_pattern.json"
OUT = "breakout_report.html"

VCOLOR = {"CONFIRM": "var(--gr)", "CAUTION": "var(--am)", "VETO": "var(--mg)"}
STAGECOLOR = {"EARLY": "var(--am)", "CONFIRMED": "var(--gr)"}


def price_spark(ticker):
    s = load_close(ticker)
    if s is None or s.empty: return ""
    return sparkline(s.iloc[-504:], h=90, color="#00e5ff")


def eps_spark(eps_trend):
    if not eps_trend or len(eps_trend) < 2: return "<div class=nochart>no EPS-trend data</div>"
    return sparkline(pd.Series(eps_trend), h=90, color="#ffcf3a")


def factor_cell(label, state):
    """state: True=present(check), False=absent(cross), None=pending live."""
    if state is True:  return f'<span class="fct ok">✓ {label}</span>'
    if state is False: return f'<span class="fct no">✗ {label}</span>'
    return f'<span class="fct pend">… {label}</span>'


def scorecard(idea):
    cells = [
        factor_cell("beaten base", idea.get("base_pos_pct", 100) < 35),
        factor_cell("deep drawdown", idea.get("dd_from_high_pct", 0) < -30),
        factor_cell("estimates rising", idea.get("eps_rising")),
        factor_cell("earnings beat", (idea.get("earnings_surprise_pct") or 0) > 2),
        factor_cell("analyst upside", (idea.get("upside_pct") or 0) > 20),
        factor_cell("not a falling knife", idea.get("verdict") != "VETO"),
    ]
    return '<div class=scorecard>' + "".join(cells) + '</div>'


def competitor_table(comps):
    if not comps: return ""
    rows = ""
    for c in comps:
        rows += (f'<tr><td class=ctk>{c.get("ticker","?")}</td>'
                 f'<td>{c.get("name","")[:22]}</td>'
                 f'<td>{c.get("fwd_pe","—")}</td>'
                 f'<td>{c.get("rev_growth_pct","—")}%</td>'
                 f'<td>{c.get("margin_pct","—")}%</td>'
                 f'<td class=cnote>{c.get("note","")}</td></tr>')
    return ('<table class=comptbl><tr><th>rival</th><th>name</th><th>fwd P/E</th>'
            '<th>rev gr</th><th>margin</th><th>note</th></tr>' + rows + '</table>')


def news_list(news):
    if not news: return ""
    tone = {"negative": "var(--mg)", "positive": "var(--gr)", "neutral": "var(--dim)"}
    items = ""
    for n in news:
        c = tone.get(n.get("tone", "neutral"), "var(--dim)")
        items += (f'<li><span class=ndot style="background:{c}"></span>'
                  f'<b>{n.get("headline","")}</b> '
                  f'<span class=nsrc>— {n.get("source","")}, {n.get("date","")}</span></li>')
    return f'<ul class=newslist>{items}</ul>'


def idea_card(idea):
    t = idea["ticker"]; v = idea.get("verdict", "CAUTION")
    r = idea.get("ratings", {}) or {}
    ratings = (f"{r.get('strong_buy',0)+r.get('buy',0)} buy / {r.get('hold',0)} hold / "
               f"{r.get('sell',0)+r.get('strong_sell',0)} sell — {r.get('consensus','?')}")
    return f"""
    <div class="glass card">
      <div class=cardhd>
        <div><span class=ctick>{t}</span>
             <span class=badge style="border-color:{STAGECOLOR.get(idea.get('stage'),'var(--dim)')};color:{STAGECOLOR.get(idea.get('stage'),'var(--dim)')}">{idea.get('stage','')}</span>
             <span class=badge style="border-color:{VCOLOR.get(v)};color:{VCOLOR.get(v)}">{v}</span></div>
        <div class=cprice>${idea.get('price','—')}
             <span class=cup>{('+'+str(idea.get('upside_pct'))+'% to target') if idea.get('upside_pct') else ''}</span></div>
      </div>
      <div class=charts>
        <div class=chartbox><div class=chlbl>PRICE · 2yr</div>{price_spark(t)}
             <div class=chsub>{idea.get('base_pos_pct','?')}% up range · {idea.get('dd_from_high_pct','?')}% off high</div></div>
        <div class=chartbox><div class=chlbl>FWD EPS ESTIMATE TREND</div>{eps_spark(idea.get('eps_trend'))}
             <div class=chsub>{'rising ▲' if idea.get('eps_rising') else 'flat/falling ▼'} · last surprise {idea.get('earnings_surprise_pct','—')}%</div></div>
      </div>
      {scorecard(idea)}
      <div class=thesis><b>Thesis.</b> {idea.get('thesis','')}</div>
      <div class=risk><b>Key risk.</b> {idea.get('key_risk','')}</div>
      <div class=metarow>
        <span><b>Analysts:</b> {ratings}</span>
        <span><b>ROE:</b> {idea.get('roe_pct','—')}%</span>
        <span><b>Margin:</b> {idea.get('margin_pct','—')}%</span>
        <span><b>vs rivals:</b> {idea.get('competitor_edge','—')}</span>
      </div>
      <div class=subh>COMPETITOR COMPARISON</div>
      {competitor_table(idea.get('competitors'))}
      <div class=subh>NEWS EVIDENCE (the catalyst / falling-knife check)</div>
      {news_list(idea.get('news'))}
    </div>"""


def learned_section():
    if not os.path.exists(PATTERN): return ""
    p = json.load(open(PATTERN)); sig = p.get("signature", {})
    hit = p.get("hit_rate")
    rows = ""
    ranked = sorted([(f, s) for f, s in sig.items() if s.get("available")],
                    key=lambda x: -abs(x[1]["effect"]))
    for f, s in ranked:
        d = "higher in winners" if s["effect"] > 0 else "lower in winners"
        col = "var(--gr)" if abs(s["effect"]) > 0.8 else "var(--dim)"
        rows += (f'<tr><td class=ctk>{f}</td><td>{s["winner_mean"]}</td><td>{s["control_mean"]}</td>'
                 f'<td style="color:{col}">{s["effect"]:+.2f}</td><td class=cnote>{d}</td></tr>')
    hitline = f"Winners-vs-controls separation hit-rate: <b>{hit*100:.0f}%</b>" if hit else ""
    return f"""
    <div class=sec-h>What we learned from history :: the blow-up fingerprint (Phase 0)</div>
    <div class="glass learned">
      <p class=lead>The screen below is not hand-guessed. We rewound 7 known blow-ups (MU, INTC, DELL, NVDA,
      AVGO, SMCI, PLTR) to their pre-explosion bottoms and compared them, point-in-time, to a control group
      that did <i>not</i> blow up. These are the features that actually discriminated the winners:</p>
      <table class=comptbl><tr><th>feature</th><th>winners</th><th>controls</th><th>effect</th><th>reading</th></tr>{rows}</table>
      <p class=lead style="margin-top:12px">{hitline}. In plain words: <b>deeply beaten down, near multi-year
      lows, volatile/contested, and being trashed in the news (downgrades, losses) — bought at peak
      pessimism.</b> The opposite of what feels safe.</p>
    </div>"""


def main():
    if not os.path.exists(IDEAS):
        print(f"no {IDEAS} — run breakout_scout.py then Stage B first."); return
    d = json.load(open(IDEAS))
    ideas = d.get("ideas", [])
    order = {"CONFIRM": 0, "CAUTION": 1, "VETO": 2}
    ideas.sort(key=lambda i: order.get(i.get("verdict"), 3))
    cards = "".join(idea_card(i) for i in ideas)
    asof = d.get("_asof", "")
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>BREAKOUT SCOUT</title>
<style>
:root{{--bg:#04070f;--gl:rgba(12,22,40,.55);--ln:rgba(0,229,255,.18);--cy:#00e5ff;--mg:#ff2d92;--gr:#1dffb0;--am:#ffcf3a;--tx:#bfe9ff;--dim:#5a7390}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--tx);font:13px/1.55 'SF Mono',ui-monospace,Menlo,monospace;max-width:900px;margin:0 auto;padding:30px 20px 80px;
background-image:linear-gradient(rgba(0,229,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.04) 1px,transparent 1px),radial-gradient(900px 460px at 50% -10%,rgba(0,229,255,.1),transparent);background-size:42px 42px,42px 42px,100% 100%}}
.lg{{font-size:17px;letter-spacing:6px;font-weight:700;color:#fff;text-shadow:0 0 14px var(--cy)}} .lg span{{color:var(--cy)}}
.as{{font-size:10px;letter-spacing:2px;color:var(--dim);margin-top:4px}}
.disc{{color:var(--am);font-size:10.5px;letter-spacing:.4px;border:1px solid rgba(255,207,58,.3);border-radius:8px;padding:9px 12px;margin:14px 0 6px;background:rgba(255,207,58,.05)}}
.sec-h{{font-size:10px;letter-spacing:4px;color:var(--cy);text-transform:uppercase;margin:30px 0 12px;display:flex;align-items:center;gap:10px;text-shadow:0 0 8px var(--cy)}}
.sec-h::before{{content:"//"}} .sec-h::after{{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--ln),transparent)}}
.glass{{background:var(--gl);border:1px solid var(--ln);border-radius:14px;backdrop-filter:blur(8px);box-shadow:0 0 30px -16px var(--cy);padding:18px 20px;margin-bottom:18px}}
.lead{{color:var(--tx);font-size:12px;line-height:1.6}}
.card .cardhd{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid var(--ln);padding-bottom:12px;margin-bottom:14px}}
.ctick{{font-size:24px;font-weight:700;color:#fff;letter-spacing:2px;text-shadow:0 0 12px var(--cy)}}
.badge{{font-size:9px;letter-spacing:2px;border:1px solid;border-radius:20px;padding:3px 10px;margin-left:10px;vertical-align:middle}}
.cprice{{font-size:18px;color:#fff;font-weight:700;text-align:right}} .cup{{display:block;font-size:11px;color:var(--gr);font-weight:400}}
.charts{{display:flex;gap:14px;margin-bottom:12px}} .chartbox{{flex:1;background:rgba(0,0,0,.2);border-radius:10px;padding:10px}}
.chlbl{{font-size:9px;letter-spacing:2px;color:var(--dim);margin-bottom:4px}} .chsub{{font-size:10px;color:var(--dim);margin-top:4px}}
.nochart{{color:var(--dim);font-size:11px;padding:30px 0;text-align:center}}
.scorecard{{display:flex;flex-wrap:wrap;gap:7px;margin:6px 0 14px}}
.fct{{font-size:10.5px;letter-spacing:.3px;border-radius:6px;padding:4px 9px;border:1px solid}}
.fct.ok{{color:var(--gr);border-color:rgba(29,255,176,.35);background:rgba(29,255,176,.06)}}
.fct.no{{color:var(--mg);border-color:rgba(255,45,146,.3);background:rgba(255,45,146,.05)}}
.fct.pend{{color:var(--dim);border-color:rgba(90,115,144,.4)}}
.thesis,.risk{{font-size:12px;line-height:1.6;margin:8px 0;padding:10px 12px;border-radius:8px;background:rgba(0,0,0,.18)}}
.thesis b{{color:var(--gr)}} .risk b{{color:var(--mg)}}
.metarow{{display:flex;flex-wrap:wrap;gap:16px;font-size:11px;color:var(--tx);margin:12px 0;padding:8px 0;border-top:1px solid rgba(0,229,255,.08)}}
.metarow b{{color:var(--cy)}}
.subh{{font-size:9px;letter-spacing:3px;color:var(--dim);margin:14px 0 6px;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse;margin-top:4px}}
.comptbl th{{font-size:9px;letter-spacing:1px;color:var(--dim);text-align:left;padding:6px 8px;border-bottom:1px solid var(--ln);text-transform:uppercase}}
.comptbl td{{padding:6px 8px;border-bottom:1px solid rgba(0,229,255,.06);font-size:11.5px}}
.ctk{{color:#fff;font-weight:700}} .cnote{{color:var(--dim);font-size:10.5px}}
.newslist{{list-style:none;padding:0}} .newslist li{{font-size:11.5px;padding:6px 0;border-bottom:1px solid rgba(0,229,255,.05);line-height:1.5}}
.ndot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.nsrc{{color:var(--dim);font-size:10.5px}}
.spark .pline{{filter:drop-shadow(0 0 4px currentColor)}}
.foot{{color:var(--dim);font-size:10px;text-align:center;margin-top:30px;letter-spacing:.5px}}
</style></head><body>
<div class=lg>BREAKOUT<span>//</span>SCOUT</div>
<div class=as>HIGH-POTENTIAL RESEARCH :: GENERATED {asof} :: {len(ideas)} IDEAS :: SOURCE {d.get('_source','')}</div>
<div class=disc>⚠ {d.get('_disclaimer','')}</div>
{learned_section()}
<div class=sec-h>Today's candidates :: ranked CONFIRM → CAUTION → VETO</div>
{cards}
<div class=foot>Breakout Scout — conviction research, not a backtested edge. You decide. // {asof}</div>
</body></html>"""
    open(OUT, "w").write(doc)
    confirms = [i["ticker"] for i in ideas if i.get("verdict") == "CONFIRM"]
    print(f"wrote {OUT} — {len(ideas)} ideas (CONFIRM: {', '.join(confirms) or 'none'})")


if __name__ == "__main__":
    main()
