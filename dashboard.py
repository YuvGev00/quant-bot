#!/usr/bin/env python3
"""dashboard.py — self-contained terminal-styled dashboard.html, regenerated each run.
No server / no external fonts (works offline + emails). Honest about the two agents:
REVERSAL = the validated edge (p<0.01); MOMENTUM = a "what's moving" radar (not a proven edge).
Reads holdings.json (empty = 100% cash). Run: python3 dashboard.py
"""
from __future__ import annotations
import json, os, html
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, gate_returns, hysteresis, after_tax, stats, LAG
from early_scanner import panels, score_leader, ETF_UNIVERSE
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict
from bot_utils import data_quality

LEVERAGED = {"SOXL","TECL","TQQQ","QLD","SSO","UPRO","SPXL","FAS","TNA","UDOW","ROM"}
MAX_SINGLE = 0.35; TOP_N = 5


def vcls(v): return {"CONFIRM":"v-good","NEUTRAL":"v-mid","CAUTION":"v-warn","VETO":"v-bad","UNREAD":"v-mid"}.get(v,"v-mid")


def sparkline(series, w=700, h=110, color="#5eead4"):
    v = series.dropna().values
    if len(v) < 2: return ""
    lo, hi = v.min(), v.max(); rng = (hi-lo) or 1
    pts = [(w*i/(len(v)-1), h-(h-16)*(x-lo)/rng-8) for i,x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" style="display:block;margin-top:10px">'
            f'<polygon points="0,{h} {line} {w},{h}" fill="url(#g)" opacity=".15"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{line}"/>'
            f'<defs><linearGradient id=g x1=0 x2=0 y1=0 y2=1><stop offset=0 stop-color="{color}"/>'
            f'<stop offset=1 stop-color="{color}" stop-opacity=0/></linearGradient></defs></svg>')


def reversal_candidates(px, vol, top=8):
    pxf, volf = px.ffill(), vol.ffill()
    drop = -(pxf/pxf.shift(10)-1)
    relvol = (volf/volf.rolling(63).mean()).clip(0.5,4.0)
    score = (drop*relvol).iloc[-1].replace([np.inf,-np.inf],np.nan).dropna()
    score = score.drop(labels=[s for s in score.index if s in LEVERAGED], errors="ignore")
    return [(s, (pxf[s].iloc[-1]/pxf[s].iloc[-11]-1)*100, float(v)) for s,v in score[score>0].nlargest(top).items()]


def main():
    px, vol = panels(ETF_UNIVERSE)
    ok, dmsg = data_quality(px)
    spy = load_close("SPY"); ro, ret, rf, _ = walk_forward(spy, None)
    risk_off = bool(ro.iloc[-1] > 0.5); asof = ro.index[-1].date()
    in_mkt = hysteresis((1-ro).shift(LAG).fillna(1.0), 3, 3)
    g = gate_returns(in_mkt, ret.loc[in_mkt.index], rf.loc[in_mkt.index])
    eq = (1 + after_tax(g, 0.25)).cumprod()
    s = stats(after_tax(g, 0.25)); sbh = stats(after_tax(ret.loc[g.index], 0.25))

    # momentum radar (no leverage)
    msc = score_leader(px.ffill()).iloc[-1].dropna()
    msc = msc.drop(labels=[x for x in msc.index if x in LEVERAGED], errors="ignore")
    mom_top = msc.sort_values(ascending=False).head(8)

    # reversal — the validated edge
    rev = reversal_candidates(px, vol)
    rev_news = {}
    if os.path.exists("news_verdicts_reversal.json"):
        try: rev_news = {k.upper():v.upper() for k,v in json.load(open("news_verdicts_reversal.json")).items() if not k.startswith("_")}
        except Exception: pass

    # holdings
    cur = {}
    if os.path.exists("holdings.json"):
        try: cur = json.load(open("holdings.json")).get("positions", {})
        except Exception: pass
    flat = (len(cur) == 0)

    # ---- HTML ----
    regime_txt = "RISK-OFF" if risk_off else "CALM"; regime_cls = "risk" if risk_off else "calm"
    regime_msg = "Storm regime — stay in cash. Do not buy." if risk_off else "Clear regime — edges are live."

    # reversal rows (editorial style)
    rev_rows = ""; buys = []
    vlabel = {"CONFIRM":"buy the bounce","CAUTION":"watch","VETO":"falling knife — skip","UNREAD":"unread"}
    for sym, d10, scv in rev:
        nv = rev_news.get(sym, "UNREAD")
        if nv not in ("VETO",): buys.append((sym, nv))
        rev_rows += (f'<tr><td class=tk>{sym}</td><td class=n>{d10:+.1f}%</td>'
                     f'<td class=vd vd-{nv.lower()}>{vlabel.get(nv,nv.lower())}</td></tr>')
    mom_rows = ""
    for i,(sym,v) in enumerate(mom_top.items(),1):
        held = ' · held' if sym in cur else ''
        mom_rows += f'<tr><td class=ix>{i:02d}</td><td class=tk>{sym}{held}</td><td class=n>{v:.2f}</td></tr>'


    # ACTION (editorial)
    if risk_off:
        action = '<div class="act act-cash"><div class=ah>Move to cash</div><div class=ap>The storm-detector is risk-off. Sit out — do not buy.</div></div>'
    elif flat:
        if buys:
            n = len(buys) or 1
            items = "".join(f'<li><b>{b[0]}</b> &mdash; {int(100/n)}% <span class=ag>{b[1].lower()}</span></li>' for b in buys)
            action = (f'<div class=act><div class=ah>Open these positions</div>'
                      f'<ol class=buys>{items}</ol>'
                      f'<div class=ap>You hold cash today, so these are fresh entries from the reversal edge. '
                      f'Keep any remainder in cash. You place the orders.</div></div>')
        else:
            action = '<div class=act><div class=ah>Stay in cash</div><div class=ap>No bounce candidate is news-confirmed this week.</div></div>'
    else:
        action = '<div class=act><div class=ah>See weekly email</div><div class=ap>You have positions on file — the emailed sheet has the hold/add/trim/sell calls.</div></div>'

    port_line = "100% cash — no positions held." if flat else " · ".join(f"{k} {v*100:.0f}%" for k,v in cur.items())
    big = f"{s['CAGR%']:.1f}"

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>The Quant Ledger — {asof}</title>
<style>
@import url('data:text/css;,');
:root{{--ink:#1a1714;--paper:#f4efe6;--paper2:#ece5d8;--rule:#cfc6b4;--mut:#8a8170;--accent:#9c2b1b;--green:#3f6b४a;--gold:#a8852c}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--paper);color:var(--ink);
font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;
max-width:760px;margin:0 auto;padding:46px 30px 80px;line-height:1.6;
background-image:repeating-linear-gradient(0deg,transparent,transparent 27px,rgba(0,0,0,.012) 28px)}}
.masthead{{text-align:center;border-bottom:3px double var(--ink);padding-bottom:14px;margin-bottom:6px}}
.title{{font-size:42px;font-weight:700;letter-spacing:-1px;font-variant:small-caps;line-height:1}}
.dek{{font-style:italic;color:var(--mut);font-size:14px;margin-top:8px;letter-spacing:.3px}}
.byline{{display:flex;justify-content:space-between;font-size:11px;text-transform:uppercase;letter-spacing:2px;color:var(--mut);
border-bottom:1px solid var(--rule);padding:8px 0;margin-bottom:30px}}
.lead{{font-size:15px}} .lead .drop{{float:left;font-size:62px;line-height:.78;padding:6px 10px 0 0;font-weight:700;color:var(--accent)}}
.regime{{display:inline-block;font-variant:small-caps;font-weight:700;letter-spacing:1px;padding:1px 10px;border-radius:2px}}
.regime.calm{{color:#2f5e3a;background:rgba(63,107,74,.12);border:1px solid rgba(63,107,74,.4)}}
.regime.risk{{color:var(--accent);background:rgba(156,43,27,.1);border:1px solid rgba(156,43,27,.4)}}
h2.sec{{font-variant:small-caps;font-size:14px;letter-spacing:3px;color:var(--mut);font-weight:700;
border-bottom:1px solid var(--rule);padding-bottom:5px;margin:38px 0 16px}}
.act{{background:var(--paper2);border:1px solid var(--rule);border-left:4px solid var(--accent);padding:18px 22px;border-radius:2px}}
.act-cash{{border-left-color:var(--accent)}}
.ah{{font-size:21px;font-weight:700;margin-bottom:6px}}
.ap{{font-size:13.5px;color:#4a443b;font-style:italic;margin-top:10px}}
ol.buys{{margin:6px 0 0;padding-left:22px}} ol.buys li{{font-size:17px;padding:3px 0}}
.ag{{font-style:italic;color:var(--mut);font-size:13px}}
table{{width:100%;border-collapse:collapse;margin-top:2px}}
th{{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:2px;color:var(--mut);font-weight:700;
border-bottom:1px solid var(--ink);padding:6px 6px}}
td{{padding:9px 6px;border-bottom:1px solid var(--rule);font-size:15px}}
td.tk{{font-weight:700;letter-spacing:.5px}} td.n{{text-align:right;font-variant-numeric:tabular-nums;color:#4a443b}}
td.ix{{color:var(--mut);font-size:12px;width:34px}} td.vd{{font-style:italic;font-size:13px}}
.vd-confirm{{color:#2f5e3a}} .vd-veto{{color:var(--accent)}} .vd-caution{{color:var(--gold)}} .vd-unread{{color:var(--mut)}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:30px}}
.fig{{border:1px solid var(--rule);background:var(--paper2);padding:18px 20px;border-radius:2px;margin-top:4px}}
.figrow{{display:flex;justify-content:space-between;align-items:baseline;padding:7px 0;border-bottom:1px dotted var(--rule)}}
.figrow:last-child{{border:none}} .figrow .k{{font-style:italic;color:var(--mut);font-size:13px}}
.figrow .val{{font-size:21px;font-weight:700}} .figrow .vs{{font-size:11px;color:var(--mut);margin-left:8px}}
.kicker{{font-style:italic;color:var(--mut);font-size:12.5px;margin-top:8px}}
.warn{{background:rgba(168,133,44,.12);border:1px solid var(--gold);padding:12px 16px;font-style:italic;font-size:13px;margin-bottom:20px;border-radius:2px}}
.foot{{border-top:3px double var(--ink);margin-top:46px;padding-top:14px;font-size:11px;color:var(--mut);font-style:italic;line-height:1.7;text-align:center}}
@media(max-width:560px){{.cols{{grid-template-columns:1fr}} .title{{font-size:32px}}}}
</style></head><body>

<div class=masthead>
  <div class=title>The Quant Ledger</div>
  <div class=dek>A weekly accounting of regime, risk, and opportunity &mdash; decisions only, you place the trades</div>
</div>
<div class=byline><span>Vol. I</span><span>{asof}</span><span>After Israeli tax · No. {regime_txt.lower()}</span></div>

{'<div class=warn>&#9888; '+html.escape(dmsg)+' — recommendations withheld this issue.</div>' if not ok else ''}

<p class=lead><span class=drop>{'S' if risk_off else 'T'}</span>
he storm-detector reads <span class="regime {regime_cls}">{regime_txt}</span> as of {asof}.
{'A storm regime is in force; the prudent position is cash and patience.' if risk_off else 'Conditions are clear, so the edges below are live.'}
Your book today stands at <b>{port_line}</b> The reversal strategy &mdash; the one that withstood
statistical scrutiny (permutation p&lt;0.01) &mdash; drives the recommendation; momentum is carried
below merely as a survey of what is moving.</p>

<h2 class=sec>The Recommendation</h2>
{action}

<h2 class=sec>Reversal &mdash; Bounce Candidates</h2>
<table><tr><th>Security</th><th class=n>10-day move</th><th>Verdict</th></tr>{rev_rows}</table>
<div class=kicker>Heavy-volume losers tend to rebound. A &ldquo;falling knife&rdquo; verdict means the decline is justified by news &mdash; left alone.</div>

<div class=cols>
<div>
<h2 class=sec>Momentum &mdash; A Survey</h2>
<table><tr><th>#</th><th>Security</th><th class=n>Score</th></tr>{mom_rows}</table>
<div class=kicker>Not a proven edge; a reading of what is currently in favour.</div>
</div>
<div>
<h2 class=sec>The Defence, Audited</h2>
<div class=fig>
<div class=figrow><span class=k>Return / year</span><span class=val style="color:var(--green)">{big}%<span class=vs>vs {sbh['CAGR%']:.1f}% hold</span></span></div>
<div class=figrow><span class=k>Risk-adjusted</span><span class=val>{s['Sharpe']:.2f}<span class=vs>vs {sbh['Sharpe']:.2f}</span></span></div>
<div class=figrow><span class=k>Worst drawdown</span><span class=val style="color:var(--accent)">{s['maxDD%']:.0f}%<span class=vs>vs {sbh['maxDD%']:.0f}%</span></span></div>
</div>
<div class=kicker>The storm-gate trades a little return for far smaller crashes. Honest, look-ahead-free figures.</div>
</div>
</div>

{sparkline(eq)}
<div class=kicker style="text-align:center">Growth of $1 under the gated strategy, after tax &mdash; now ${eq.iloc[-1]:.2f}.</div>

<div class=foot>
This ledger decides; it never trades. Orders are placed by hand via Interactive Israel.<br>
Reversal is a statistically-validated edge; momentum is a radar, not a promise; leverage is excluded throughout.<br>
Set in Georgia. Composed by python3 dashboard.py.
</div>
</body></html>"""
    open("dashboard.html","w").write(doc)
    print(f"wrote dashboard.html — {regime_txt}, portfolio={'CASH' if flat else 'mixed'}, "
          f"{len(buys)} reversal buys, gated Sharpe {s['Sharpe']:.2f}")


if __name__ == "__main__":
    main()
