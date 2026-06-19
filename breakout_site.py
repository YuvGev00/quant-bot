#!/usr/bin/env python3
"""breakout_site.py — standalone MULTI-PAGE research site for the Breakout Scout.

Replaces the single-file breakout_report.html with a small static site under breakout_site/:
  - index.html          : landing page — learned fingerprint + a card grid of every found stock
  - <TICKER>.html       : ONE FULL PAGE PER STOCK — a complete equity-research dossier:
        header · price & EPS charts · fingerprint scorecard (vs winner means) · full thesis & key risk
        · analyst panel · dated news evidence
        · ★ SAME-SECTOR BEATEN-DOWN COHORT TABLE — ranks the pick vs every other stock in its sector
          in the same setup, factor-by-factor, with a generated "why #1" line (the user's core ask:
          "why this stock and not other stocks from the same sector in the same situation?")
        · secondary business-competitor moat table (if Stage-B supplied competitors)

Design: a PREMIUM RESEARCH-TERMINAL aesthetic — ink/charcoal ground, gold-leaf accent, editorial serif
display + mono data — deliberately distinct from the neon-cyan HUD dashboard. Self-contained pages
(inline CSS, reused dashboard.sparkline), no server/CDN, emailable.

Reads breakout_ideas.json (Stage-B dossiers) + breakout_shortlist.json (cohorts) + breakout_pattern.json.
Run:  python3 breakout_site.py
"""
from __future__ import annotations
import os, json, html
import pandas as pd
from dashboard import sparkline
from jump_model import load_close

IDEAS = "breakout_ideas.json"
SHORT = "breakout_shortlist.json"
PATTERN = "breakout_pattern.json"
OUTDIR = "breakout_site"

GOLD = "#c8a44d"; GOLDLT = "#e8cf8a"; TEAL = "#5fb0a6"
GREEN = "#6fbf73"; RED = "#d86b6b"; AMBER = "#d4a13a"
VCOLOR = {"CONFIRM": GREEN, "CAUTION": AMBER, "VETO": RED}
SEC_LABEL = {"tech": "Technology", "comm": "Communications", "discretionary": "Consumer Discretionary",
             "staples": "Consumer Staples", "health": "Health Care", "financials": "Financials",
             "industrials": "Industrials", "energy": "Energy", "materials": "Materials",
             "utilities": "Utilities", "reits": "Real Estate", "semis": "Semiconductors"}


def esc(s): return html.escape(str(s)) if s is not None else ""


# ---------- shared CSS: premium research-terminal theme ----------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--ink:#0e0f12;--ink2:#16181d;--card:#1a1d23;--line:#2c2f37;--gold:#c8a44d;--goldlt:#e8cf8a;
 --teal:#5fb0a6;--green:#6fbf73;--red:#d86b6b;--amber:#d4a13a;--tx:#d8d4c8;--dim:#8a8678;--faint:#5d5a50}
*{box-sizing:border-box;margin:0}
body{background:var(--ink);color:var(--tx);
 font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:13.5px;line-height:1.65;
 background-image:radial-gradient(1100px 520px at 80% -8%,rgba(200,164,77,.07),transparent),
   radial-gradient(800px 400px at 0% 0%,rgba(95,176,166,.05),transparent);
 background-attachment:fixed}
.wrap{max-width:920px;margin:0 auto;padding:46px 26px 100px}
a{color:var(--gold);text-decoration:none} a:hover{color:var(--goldlt)}
.serif{font-family:'Fraunces',Georgia,serif}
.kicker{font-size:10.5px;letter-spacing:.32em;text-transform:uppercase;color:var(--gold);font-weight:600}
h1.mast{font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:40px;line-height:1.02;color:#f3eee0;margin:8px 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12px;letter-spacing:.04em}
.rule{height:1px;background:linear-gradient(90deg,var(--gold),transparent);margin:22px 0 6px;opacity:.5}
.disc{color:var(--amber);font-size:11px;border-left:2px solid var(--amber);padding:9px 14px;margin:18px 0;background:rgba(212,161,58,.06);line-height:1.6}
.sec-h{font-family:'Fraunces',Georgia,serif;font-size:19px;color:var(--goldlt);margin:34px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line);font-weight:600}
.badge{display:inline-block;font-size:10px;letter-spacing:.14em;text-transform:uppercase;border:1px solid;border-radius:2px;padding:3px 9px;vertical-align:middle;font-weight:500}
table{width:100%;border-collapse:collapse;margin-top:6px}
th{font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);text-align:left;padding:9px 10px;border-bottom:1px solid var(--gold);font-weight:600}
td{padding:9px 10px;border-bottom:1px solid var(--line);font-size:12.5px;vertical-align:top}
tr:last-child td{border-bottom:none}
.mono{font-variant-numeric:tabular-nums}
.spark .pline{filter:drop-shadow(0 0 3px currentColor)}
.foot{color:var(--faint);font-size:10.5px;text-align:center;margin-top:48px;letter-spacing:.06em;line-height:1.8}
/* index card grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:16px;margin-top:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:18px 18px 14px;transition:.18s;position:relative;overflow:hidden}
.tile:hover{border-color:var(--gold);transform:translateY(-2px)}
.tile .vrib{position:absolute;top:0;right:0;font-size:9px;letter-spacing:.12em;padding:3px 9px;border-bottom-left-radius:4px;color:#0e0f12;font-weight:600}
.tile .tk{font-family:'Fraunces',Georgia,serif;font-size:30px;color:#f3eee0;font-weight:600;line-height:1}
.tile .se{font-size:10px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;margin-top:2px}
.tile .up{color:var(--green);font-size:13px;margin-top:8px;font-weight:500}
.tile .th{color:var(--dim);font-size:11px;margin-top:7px;line-height:1.5}
/* stock page */
.phead{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
.pname{font-family:'Fraunces',Georgia,serif;font-size:52px;color:#f3eee0;font-weight:600;line-height:.95}
.pcompany{color:var(--dim);font-size:12px;letter-spacing:.05em;margin-top:4px}
.ppx{text-align:right} .ppx .v{font-size:30px;color:#f3eee0;font-family:'Fraunces',serif} .ppx .u{color:var(--green);font-size:13px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:6px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:14px}
.chart .cl{font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);margin-bottom:6px}
.chart .cs{font-size:10.5px;color:var(--faint);margin-top:5px}
.score{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;margin-top:4px}
.sc{border:1px solid var(--line);border-radius:3px;padding:9px 11px;background:var(--card)}
.sc .l{font-size:10px;color:var(--dim);letter-spacing:.05em} .sc .v{font-size:14px;font-weight:600;margin-top:2px}
.sc.hit{border-color:rgba(111,191,115,.4)} .sc.hit .v{color:var(--green)}
.sc.miss .v{color:var(--dim)} .sc.pend .v{color:var(--faint)}
.prose{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:18px 20px;margin-top:6px;font-size:13px;line-height:1.78;color:#cfcabb}
.prose b{color:var(--goldlt);font-weight:600}
.risk{border-left:2px solid var(--red)} .risk b{color:var(--red)}
.meta{display:flex;flex-wrap:wrap;gap:22px;margin:14px 0;font-size:12px} .meta b{color:var(--gold)}
.newslist{list-style:none;padding:0} .newslist li{padding:9px 0;border-bottom:1px solid var(--line);font-size:12px;line-height:1.55}
.ndot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:9px;vertical-align:middle}
.nsrc{color:var(--faint);font-size:10.5px}
.pickrow td{background:linear-gradient(90deg,rgba(200,164,77,.14),transparent);font-weight:600}
.pickrow .ctk{color:var(--goldlt)}
.why{background:linear-gradient(100deg,rgba(200,164,77,.1),rgba(95,176,166,.05));border:1px solid var(--gold);border-radius:4px;padding:14px 18px;margin-top:10px;font-size:12.5px;line-height:1.7}
.why b{color:var(--goldlt);font-family:'Fraunces',serif}
.nav{font-size:11px;letter-spacing:.1em;margin-bottom:18px} .nav a{color:var(--dim)} .nav a:hover{color:var(--gold)}
"""


def page(title, body):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head>"
            f"<body><div class=wrap>{body}</div></body></html>")


def price_chart(ticker, color=GOLD):
    s = load_close(ticker)
    if s is None or s.empty: return "<div class=cs>no price history</div>"
    return sparkline(s.iloc[-504:], h=92, color=color)


def eps_chart(eps_trend, color=TEAL):
    if not eps_trend or len(eps_trend) < 2: return "<div class=cs>no EPS-trend data</div>"
    return sparkline(pd.Series(eps_trend), h=92, color=color)


# ---------- the headline feature: same-sector beaten-down cohort ----------
def cohort_block(idea, cohorts):
    tk = idea["ticker"]
    # find which sector cohort this pick belongs to
    sec = next((s for s, lst in cohorts.items() if any(c["ticker"] == tk for c in lst)), None)
    if not sec or len(cohorts.get(sec, [])) < 2:
        return "<p class=sub>No other same-sector beaten-down peers were found this scan.</p>"
    coh = cohorts[sec]
    pick = next((c for c in coh if c["ticker"] == tk), None)
    rank = [c["ticker"] for c in coh].index(tk) + 1
    rows = ""
    for i, c in enumerate(coh, 1):
        cls = " class=pickrow" if c["ticker"] == tk else ""
        mark = " ◄ THIS PICK" if c["ticker"] == tk else ""
        rows += (f"<tr{cls}><td class=mono>{i}</td><td class=ctk>{esc(c['ticker'])}{mark}</td>"
                 f"<td class=mono>{c.get('resemblance','—')}</td>"
                 f"<td class=mono>{c.get('base_pos%','—')}%</td>"
                 f"<td class=mono>{c.get('dd%','—')}%</td>"
                 f"<td class=mono>{c.get('vol_ann%','—')}%</td>"
                 f"<td class=mono>{('+'+str(c['upside%'])+'%') if c.get('upside%') else '—'}</td>"
                 f"<td>{esc(c.get('analyst','—'))}</td></tr>")
    # generate the "why #1 (or why ranked here)" line by comparing the pick to the rest of the cohort
    why = why_line(pick, coh, rank, sec)
    return (f"<p class=sub>Of every <b style='color:var(--goldlt)'>{SEC_LABEL.get(sec, sec)}</b> name in the same "
            f"beaten-down setup right now, ranked by resemblance to the learned blow-up fingerprint:</p>"
            f"<table><tr><th>#</th><th>ticker</th><th>resemblance</th><th>base pos</th><th>off high</th>"
            f"<th>volatility</th><th>analyst upside</th><th>rating</th></tr>{rows}</table>"
            f"<div class=why>{why}</div>")


def why_line(pick, coh, rank, sec):
    if pick is None: return ""
    others = [c for c in coh if c["ticker"] != pick["ticker"]]
    if not others:
        return f"<b>{pick['ticker']}</b> is the only beaten-down name in {SEC_LABEL.get(sec, sec)} this scan."
    def avg(k):
        vals = [c[k] for c in others if c.get(k) is not None]
        return sum(vals)/len(vals) if vals else None
    bits = []
    if pick.get("dd%") is not None and avg("dd%") is not None and pick["dd%"] < avg("dd%"):
        bits.append(f"more deeply beaten ({pick['dd%']}% off high vs peer avg {avg('dd%'):.0f}%)")
    if pick.get("vol_ann%") is not None and avg("vol_ann%") is not None and pick["vol_ann%"] > avg("vol_ann%"):
        bits.append(f"more volatile/contested ({pick['vol_ann%']}% vs {avg('vol_ann%'):.0f}%)")
    if pick.get("upside%") is not None and avg("upside%") is not None and pick["upside%"] > avg("upside%"):
        bits.append(f"higher analyst upside (+{pick['upside%']}% vs +{avg('upside%'):.0f}%)")
    if pick.get("base_pos%") is not None and avg("base_pos%") is not None and pick["base_pos%"] < avg("base_pos%"):
        bits.append(f"nearer its lows ({pick['base_pos%']}% of range vs {avg('base_pos%'):.0f}%)")
    head = (f"<b>Why {pick['ticker']} ranks #{rank} of {len(coh)} in {SEC_LABEL.get(sec, sec)}.</b> "
            if rank == 1 else f"<b>{pick['ticker']} ranks #{rank} of {len(coh)} here.</b> ")
    if not bits:
        return head + ("It resembles the winner fingerprint less strongly than its same-sector peers on the "
                       "core factors — a reason to prefer a higher-ranked name in this group.")
    lead = "It best matches the winners' pre-explosion shape: " if rank == 1 else "On the fingerprint it is "
    return head + lead + "; ".join(bits) + ". (Resemblance ranking, not a probability of blowing up.)"


def scorecard(idea):
    def cell(label, val, ok):
        cls = "hit" if ok is True else "miss" if ok is False else "pend"
        return f'<div class="sc {cls}"><div class=l>{label}</div><div class=v>{val}</div></div>'
    bp = idea.get("base_pos_pct"); dd = idea.get("dd_from_high_pct"); su = idea.get("earnings_surprise_pct")
    up = idea.get("upside_pct")
    cells = [
        cell("beaten base", f"{bp}% of range" if bp is not None else "—", (bp is not None and bp < 35)),
        cell("drawdown", f"{dd}% off high" if dd is not None else "—", (dd is not None and dd < -30)),
        cell("estimates", "rising ▲" if idea.get("eps_rising") else "flat/falling ▼", idea.get("eps_rising")),
        cell("earnings surprise", f"{su}%" if su is not None else "—", (su is not None and su > 2)),
        cell("analyst upside", f"+{up}%" if up is not None else "—", (up is not None and up > 20)),
        cell("falling-knife check", idea.get("verdict", "—"), idea.get("verdict") != "VETO"),
    ]
    return '<div class=score>' + "".join(cells) + '</div>'


def news_list(news):
    if not news: return "<p class=sub>No news evidence captured.</p>"
    tone = {"negative": RED, "positive": GREEN, "neutral": "#8a8678"}
    items = ""
    for n in news:
        c = tone.get(n.get("tone", "neutral"), "#8a8678")
        items += (f'<li><span class=ndot style="background:{c}"></span><b>{esc(n.get("headline",""))}</b> '
                  f'<span class=nsrc>— {esc(n.get("source",""))}, {esc(n.get("date",""))}</span></li>')
    return f"<ul class=newslist>{items}</ul>"


def competitor_table(comps):
    if not comps: return ""
    rows = "".join(f"<tr><td class=ctk>{esc(c.get('ticker','?'))}</td><td>{esc(c.get('name','')[:24])}</td>"
                   f"<td class=mono>{c.get('fwd_pe','—')}</td><td class=mono>{c.get('rev_growth_pct','—')}%</td>"
                   f"<td class=mono>{c.get('margin_pct','—')}%</td><td class=sub>{esc(c.get('note',''))}</td></tr>"
                   for c in comps)
    return ('<div class=sec-h>Competitive moat check</div>'
            '<table><tr><th>rival</th><th>name</th><th>fwd P/E</th><th>rev growth</th><th>margin</th><th>note</th></tr>'
            + rows + '</table>')


def stock_page(idea, cohorts, disclaimer):
    t = idea["ticker"]; v = idea.get("verdict", "CAUTION"); vc = VCOLOR.get(v, GOLD)
    r = idea.get("ratings", {}) or {}
    ratings = (f"{r.get('strong_buy',0)+r.get('buy',0)} buy · {r.get('hold',0)} hold · "
               f"{r.get('sell',0)+r.get('strong_sell',0)} sell — {r.get('consensus','?')}")
    up = idea.get("upside_pct")
    body = f"""
    <div class=nav><a href="index.html">← All ideas</a></div>
    <div class=phead>
      <div><div class=pname>{esc(t)}</div><div class=pcompany>{SEC_LABEL.get(idea.get('sector',''),idea.get('sector',''))} · {idea.get('stage','')}</div></div>
      <div class=ppx><div class=v>${idea.get('price','—')}</div>
        <div class=u>{('▲ +'+str(up)+'% to target $'+str(idea.get('target_consensus',''))) if up else ''}</div>
        <div style="margin-top:8px"><span class=badge style="border-color:{vc};color:{vc}">{esc(v)}</span></div></div>
    </div>
    <div class=rule></div>
    <div class=charts>
      <div class=chart><div class=cl>Price · 2-year</div>{price_chart(t)}
        <div class=cs>{idea.get('base_pos_pct','?')}% up its range · {idea.get('dd_from_high_pct','?')}% off high</div></div>
      <div class=chart><div class=cl>Forward EPS estimate trend</div>{eps_chart(idea.get('eps_trend'))}
        <div class=cs>{'rising ▲' if idea.get('eps_rising') else 'flat/falling ▼'} · last surprise {idea.get('earnings_surprise_pct','—')}%</div></div>
    </div>
    <div class=sec-h>Fingerprint scorecard</div>
    {scorecard(idea)}
    <div class=sec-h>Investment thesis</div>
    <div class=prose><b>Thesis.</b> {esc(idea.get('thesis',''))}</div>
    <div class="prose risk" style="margin-top:10px"><b>Key risk.</b> {esc(idea.get('key_risk',''))}</div>
    <div class=meta><span><b>Analysts:</b> {ratings}</span><span><b>ROE:</b> {idea.get('roe_pct','—')}%</span>
      <span><b>Margin:</b> {idea.get('margin_pct','—')}%</span><span><b>vs rivals:</b> {esc(idea.get('competitor_edge','—'))}</span></div>
    <div class=sec-h>★ Why this one — same-sector beaten-down cohort</div>
    {cohort_block(idea, cohorts)}
    {competitor_table(idea.get('competitors'))}
    <div class=sec-h>News evidence</div>
    {news_list(idea.get('news'))}
    <div class=disc>{esc(disclaimer)}</div>
    <div class=foot>BREAKOUT SCOUT · conviction research, not a backtested edge · you decide</div>
    """
    return page(f"{t} — Breakout Scout", body)


def learned_table():
    if not os.path.exists(PATTERN): return ""
    p = json.load(open(PATTERN)); sig = p.get("signature", {}); hit = p.get("hit_rate")
    ranked = sorted([(f, s) for f, s in sig.items() if s.get("available")], key=lambda x: -abs(x[1]["effect"]))
    rows = "".join(f"<tr><td class=ctk>{esc(f)}</td><td class=mono>{s['winner_mean']}</td>"
                   f"<td class=mono>{s['control_mean']}</td>"
                   f"<td class=mono style='color:{GOLD if abs(s['effect'])>0.8 else 'var(--dim)'}'>{s['effect']:+.2f}</td>"
                   f"<td class=sub>{'higher in winners' if s['effect']>0 else 'lower in winners'}</td></tr>"
                   for f, s in ranked)
    hl = f"Winners-vs-controls separation: <b style='color:var(--goldlt)'>{hit*100:.0f}%</b>." if hit else ""
    return (f"<div class=sec-h>What we learned from history</div>"
            f"<p class=sub>The screen is not hand-guessed. We rewound 7 known blow-ups (MU, INTC, DELL, NVDA, "
            f"AVGO, SMCI, PLTR) to their pre-explosion bottoms and compared them — point-in-time — to a control "
            f"group that did <i>not</i> run. These features discriminated the winners:</p>"
            f"<table><tr><th>feature</th><th>winners</th><th>controls</th><th>effect</th><th>reading</th></tr>{rows}</table>"
            f"<p class=sub style='margin-top:10px'>{hl} In plain words: <b style='color:var(--goldlt)'>deeply beaten, "
            f"near multi-year lows, volatile/contested, and being trashed in the news — bought at peak pessimism.</b></p>")


TIER_COLOR = {"EMERGING": TEAL, "SPECULATIVE_EARLY": AMBER}
FIT_COLOR = {"STRONG": GREEN, "MODERATE": AMBER, "WEAK": RED}


def disc_tiles(discoveries):
    tiles = ""
    for x in discoveries:
        tc = TIER_COLOR.get(x.get("tier"), GOLD); fc = FIT_COLOR.get(x.get("early_fit"), GOLD)
        mc = x.get("market_cap_b"); na = x.get("n_analysts")
        tiles += f"""<a class=tile href="disc_{esc(x['ticker'])}.html">
          <div class=vrib style="background:{fc}">{esc(x.get('early_fit',''))}</div>
          <div class=tk>{esc(x['ticker'])}</div>
          <div class=se><span style="color:{tc}">{esc(x.get('tier','').replace('_',' '))}</span>
            · {('$'+str(mc)+'B') if mc else '?'} · {na if na is not None else '?'} analysts</div>
          <div class=up style="color:{tc}">early-fit score {x.get('early_score','?')}</div>
          <div class=th>{esc((x.get('thesis','') or '')[:115])}…</div></a>"""
    return tiles


def disc_page(x, disclaimer):
    t = x["ticker"]; tc = TIER_COLOR.get(x.get("tier"), GOLD); fc = FIT_COLOR.get(x.get("early_fit"), GOLD)
    sigs = "".join(f'<span class="sc hit" style="display:inline-block;margin:3px">{esc(s)}</span>'
                   for s in x.get("early_signals", []))
    r = x.get("ratings", {}) or {}
    ratings = f"{r.get('buy',0)} buy · {r.get('hold',0)} hold · {r.get('sell',0)} sell" if r else "thin coverage"
    body = f"""
    <div class=nav><a href="index.html">← All ideas</a></div>
    <div class=phead>
      <div><div class=pname>{esc(t)}</div><div class=pcompany>{esc(x.get('name',''))}</div></div>
      <div class=ppx><div class=v>${x.get('price','—')}</div>
        <div style="margin-top:8px">
          <span class=badge style="border-color:{tc};color:{tc}">{esc(x.get('tier','').replace('_',' '))}</span>
          <span class=badge style="border-color:{fc};color:{fc}">{esc(x.get('early_fit',''))} EARLY-FIT</span></div></div>
    </div>
    <div class=rule></div>
    <div class=score>
      <div class="sc hit"><div class=l>early-fit score</div><div class=v>{x.get('early_score','—')}/100</div></div>
      <div class="sc {'hit' if (x.get('n_analysts') or 99)<=6 else 'miss'}"><div class=l>analyst coverage</div><div class=v>{x.get('n_analysts','—')} analysts</div></div>
      <div class="sc {'hit' if x.get('cheap') else 'miss'}"><div class=l>valuation</div><div class=v>{esc(x.get('valuation','—'))}</div></div>
      <div class="sc {'hit' if x.get('eps_rising') else 'miss'}"><div class=l>estimates</div><div class=v>{'rising ▲' if x.get('eps_rising') else 'flat/falling ▼'}</div></div>
      <div class="sc {'hit' if (x.get('earnings_surprise_pct') or 0)>2 else 'miss'}"><div class=l>earnings surprise</div><div class=v>{x.get('earnings_surprise_pct','—')}%</div></div>
      <div class=sc><div class=l>market cap</div><div class=v>${x.get('market_cap_b','—')}B</div></div>
    </div>
    <div class=sec-h>Why it's early</div>
    <div class=prose>{sigs or '<span class=sub>signals pending</span>'}</div>
    <div class=sec-h>Thesis</div>
    <div class=prose><b>Thesis.</b> {esc(x.get('thesis',''))}</div>
    <div class="prose risk" style="margin-top:10px"><b>Key risk.</b> {esc(x.get('key_risk',''))}</div>
    <div class=meta><span><b>Analysts:</b> {ratings}</span><span><b>Rev growth:</b> {x.get('rev_growth_pct','—')}%</span>
      <span><b>Margin:</b> {x.get('margin_pct','—')}%</span><span><b>Base position:</b> {x.get('base_pos_pct','—')}% of range</span></div>
    <div class=sec-h>News evidence</div>
    {news_list(x.get('news'))}
    <div class=disc>{esc(disclaimer)}</div>
    <div class=foot>BREAKOUT SCOUT · EARLY DISCOVERY · pre-hype small/mid-cap · forward research, not a backtested edge</div>
    """
    return page(f"{t} — Early Discovery", body)


def index_page(ideas, asof, source, disclaimer, discoveries=None, disc_disc=""):
    order = {"CONFIRM": 0, "CAUTION": 1, "VETO": 2}
    ideas = sorted(ideas, key=lambda i: order.get(i.get("verdict"), 3))
    tiles = ""
    for i in ideas:
        v = i.get("verdict", ""); vc = VCOLOR.get(v, GOLD); up = i.get("upside_pct")
        tiles += f"""<a class=tile href="{esc(i['ticker'])}.html">
          <div class=vrib style="background:{vc}">{esc(v)}</div>
          <div class=tk>{esc(i['ticker'])}</div>
          <div class=se>{SEC_LABEL.get(i.get('sector',''),i.get('sector',''))} · {i.get('stage','')}</div>
          {('<div class=up>▲ +'+str(up)+'% upside</div>') if up else ''}
          <div class=th>{esc((i.get('thesis','') or '')[:115])}…</div></a>"""
    disc_section = ""
    if discoveries:
        disc_section = f"""
    <div class=sec-h>★ Early Discovery · pre-hype small &amp; mid-caps · found before the crowd</div>
    <p class=sub>The other mode: under-covered, cheap, quietly-improving small/mid-caps caught <i>early</i> —
      the “MU in early-2025 / NVDA-2023 / PLTR-at-$6” moment, before the hype. Tiered
      <span style="color:{TEAL}">EMERGING</span> (mid-cap, steadier) vs
      <span style="color:{AMBER}">SPECULATIVE EARLY</span> (micro/IPO, lottery-risk).</p>
    <div class=disc style="border-color:{AMBER}">{esc(disc_disc)}</div>
    <div class=grid>{disc_tiles(discoveries)}</div>"""
    body = f"""
    <div class=kicker>Quant Bot · Equity Research</div>
    <h1 class=mast>Breakout Scout</h1>
    <div class=sub>generated {esc(asof)} · {len(ideas)} large-cap ideas · {len(discoveries or [])} early discoveries</div>
    <div class=disc>{esc(disclaimer)}</div>
    {disc_section}
    <div class=sec-h>Large-cap mode · deeply-beaten fingerprint · ranked confirm → caution → veto</div>
    <p class=sub>The first mode: famous large-caps that fell hard and resemble past winners at their bottoms.
      Each opens a full dossier with a same-sector cohort comparison.</p>
    {learned_table()}
    <div class=grid>{tiles}</div>
    <div class=foot>Two modes — early small-cap discovery + large-cap beaten-down fingerprint.<br>
      Conviction research, not a backtested edge. You decide.</div>
    """
    return page("Breakout Scout — research", body)


def main():
    have_ideas = os.path.exists(IDEAS)
    have_disc = os.path.exists("breakout_discover.json")
    if not have_ideas and not have_disc:
        print(f"no {IDEAS} or breakout_discover.json — run breakout_scout.py + Stage B and/or the discovery sweep first."); return
    d = json.load(open(IDEAS)) if have_ideas else {}
    ideas = d.get("ideas", [])
    asof = d.get("_asof", ""); source = d.get("_source", ""); disc = d.get("_disclaimer", "")
    # early-discovery mode (second mode)
    discoveries, disc_disc = [], ""
    if have_disc:
        from breakout_discover import ranked_discoveries
        discoveries, dd = ranked_discoveries()
        disc_disc = (dd or {}).get("_disclaimer", "")
        if not asof: asof = (dd or {}).get("_asof", "")
    cohorts = {}
    if os.path.exists(SHORT):
        cohorts = json.load(open(SHORT)).get("cohorts", {})
    # attach sector to each idea if missing (from cohorts or pattern)
    for i in ideas:
        if not i.get("sector"):
            i["sector"] = next((s for s, lst in cohorts.items() if any(c["ticker"] == i["ticker"] for c in lst)), "")
    os.makedirs(OUTDIR, exist_ok=True)
    open(os.path.join(OUTDIR, "index.html"), "w").write(
        index_page(ideas, asof, source, disc, discoveries=discoveries, disc_disc=disc_disc))
    for i in ideas:
        open(os.path.join(OUTDIR, f"{i['ticker']}.html"), "w").write(stock_page(i, cohorts, disc))
    for x in discoveries:
        open(os.path.join(OUTDIR, f"disc_{x['ticker']}.html"), "w").write(disc_page(x, disc_disc))
    confirms = [i["ticker"] for i in ideas if i.get("verdict") == "CONFIRM"]
    strong = [x["ticker"] for x in discoveries if x.get("early_fit") == "STRONG"]
    print(f"wrote {OUTDIR}/index.html + {len(ideas)} large-cap pages (CONFIRM: {', '.join(confirms) or 'none'}) "
          f"+ {len(discoveries)} discovery pages (STRONG: {', '.join(strong) or 'none'})")


if __name__ == "__main__":
    main()
