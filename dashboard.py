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

    # reversal action (the proven edge drives the recommendation)
    rev_rows = ""; buys = []
    for sym, d10, scv in rev:
        nv = rev_news.get(sym, "UNREAD")
        if nv not in ("VETO",): buys.append((sym, nv))
        rev_rows += (f'<tr><td class=t>{sym}</td><td class=num red>{d10:+.1f}%</td><td class=num>{scv:.2f}</td>'
                     f'<td><span class="v {vcls(nv)}">{nv}</span></td></tr>')
    mom_rows = ""
    for i,(sym,v) in enumerate(mom_top.items(),1):
        held = '<span class=chip>held</span>' if sym in cur else ''
        mom_rows += f'<tr><td class=rk>{i}</td><td class=t>{sym}{held}</td><td class=num>{v:.2f}</td><td class=bar><span style="width:{min(v/mom_top.iloc[0]*100,100):.0f}%"></span></td></tr>'

    # ACTION block
    if risk_off:
        action = '<div class="ord sell"><span class=oact>MOVE TO CASH</span><span class=osym>storm regime</span></div>'
    elif flat:
        if buys:
            n = len([b for b in buys if b[1] != "VETO"]) or 1
            action = "".join(f'<div class="ord buy"><span class=oact>BUY {100//n}%</span><span class=osym>{b[0]} · reversal {b[1].lower()}</span></div>' for b in buys)
            action += f'<div class="cashline">portfolio is empty → these are fresh buys (reversal edge). Keep any remainder in cash.</div>'
        else:
            action = '<div class="ord muted">No confirmed bounce candidates — stay in cash.</div>'
    else:
        action = '<div class="ord muted">(holdings on file — see weekly email for HOLD/ADD/TRIM/SELL)</div>'

    port_line = "100% CASH — you hold nothing right now." if flat else ", ".join(f"{k} {v*100:.0f}%" for k,v in cur.items())

    doc = f"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>QUANTBOT · {regime_txt}</title>
<style>
:root{{--bg:#080b11;--p:#0f1620;--p2:#0b1119;--ln:#1c2735;--tx:#cdd9e5;--dim:#6a7889;
--grn:#34d399;--red:#fb7185;--amb:#fbbf24;--cy:#38bdf8;--mono:'SF Mono',ui-monospace,Menlo,monospace}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--tx);font:13.5px/1.55 var(--mono);max-width:820px;margin:0 auto;padding:26px 18px 70px;
background-image:radial-gradient(1200px 400px at 15% -5%,rgba(56,189,248,.07),transparent),radial-gradient(900px 360px at 95% 0,rgba(52,211,153,.05),transparent);-webkit-font-smoothing:antialiased}}
.hd{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ln);padding-bottom:13px}}
.lg{{font-size:13px;letter-spacing:3px;color:var(--cy)}} .lg b{{color:#fff}} .as{{color:var(--dim);font-size:11.5px;letter-spacing:1px}}
.flow{{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px;font-size:10.5px;color:var(--dim)}}
.flow span{{background:var(--p2);border:1px solid var(--ln);border-radius:14px;padding:3px 10px}} .flow b{{color:var(--cy)}}
.tag{{color:var(--dim);font-size:10.5px;letter-spacing:1.5px;text-transform:uppercase;margin:24px 0 9px;display:flex;align-items:center;gap:9px}}
.tag::after{{content:"";flex:1;height:1px;background:var(--ln)}}
.rg{{margin-top:18px;border:1px solid var(--ln);border-radius:14px;padding:22px 24px;background:var(--p);position:relative;overflow:hidden}}
.rg.calm{{border-color:rgba(52,211,153,.4);box-shadow:inset 0 0 60px -26px var(--grn)}}
.rg.risk{{border-color:rgba(251,113,133,.45);box-shadow:inset 0 0 60px -22px var(--red)}}
.rst{{font-size:38px;font-weight:700;letter-spacing:1px;line-height:1}} .calm .rst{{color:var(--grn)}} .risk .rst{{color:var(--red)}}
.dot{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:13px;vertical-align:middle}}
.calm .dot{{background:var(--grn);box-shadow:0 0 14px var(--grn);animation:pl 2.4s infinite}}
.risk .dot{{background:var(--red);box-shadow:0 0 14px var(--red);animation:pl 1.1s infinite}}
@keyframes pl{{0%,100%{{opacity:1}}50%{{opacity:.3}}}} .rmsg{{color:var(--dim);margin-top:9px;font-size:12.5px}}
.ord{{display:flex;align-items:center;gap:14px;background:var(--p);border:1px solid var(--ln);border-left-width:3px;border-radius:9px;padding:12px 16px;margin-bottom:7px}}
.ord .oact{{font-weight:600;min-width:96px}} .ord .osym{{color:var(--dim);font-size:12px}}
.ord.buy{{border-left-color:var(--grn)}} .ord.buy .oact{{color:var(--grn)}}
.ord.sell{{border-left-color:var(--red)}} .ord.sell .oact{{color:var(--red)}}
.ord.muted{{border-left-color:var(--ln);color:var(--dim);justify-content:center}}
.cashline{{color:var(--dim);font-size:11.5px;margin-top:4px}}
.pnl{{background:var(--p);border:1px solid var(--ln);border-radius:13px;padding:6px 4px}}
table{{width:100%;border-collapse:collapse}}
th{{color:var(--dim);font-size:9.5px;letter-spacing:1.5px;text-transform:uppercase;font-weight:500;text-align:left;padding:9px 13px;border-bottom:1px solid var(--ln)}}
td{{padding:9px 13px;border-bottom:1px solid rgba(28,39,53,.5)}} tr:last-child td{{border-bottom:none}}
td.t{{font-weight:600;color:#eaf2fb;letter-spacing:.4px}} td.num{{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}}
td.num.red{{color:var(--red)}} td.rk{{color:var(--dim);width:26px}}
td.bar{{width:32%}} td.bar span{{display:block;height:6px;background:linear-gradient(90deg,var(--cy),var(--grn));border-radius:3px}}
.v{{font-size:10.5px;font-weight:600;letter-spacing:.4px;padding:2px 8px;border-radius:5px}}
.v-good{{background:rgba(52,211,153,.15);color:var(--grn)}} .v-mid{{background:rgba(106,120,137,.18);color:var(--dim)}}
.v-warn{{background:rgba(251,191,36,.15);color:var(--amb)}} .v-bad{{background:rgba(251,113,133,.16);color:var(--red)}}
.chip{{font-size:9px;background:rgba(56,189,248,.13);color:var(--cy);padding:2px 6px;border-radius:4px;margin-left:7px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--ln);border:1px solid var(--ln);border-radius:13px;overflow:hidden}}
.stat{{background:var(--p);padding:17px 18px}} .stat .l{{color:var(--dim);font-size:9.5px;letter-spacing:1.5px;text-transform:uppercase}}
.stat .v2{{font-size:25px;font-weight:700;margin-top:5px;font-variant-numeric:tabular-nums}} .stat .c{{color:var(--dim);font-size:10.5px;margin-top:3px}}
.note{{color:var(--dim);font-size:11px;margin-top:7px;line-height:1.6}}
.warn{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.4);color:var(--amb);border-radius:10px;padding:13px 17px;margin-top:16px;font-size:12.5px}}
.ft{{color:var(--dim);font-size:10.5px;margin-top:34px;border-top:1px solid var(--ln);padding-top:14px;line-height:1.7}}
</style>
<div class=hd><div class=lg>QUANT<b>BOT</b></div><div class=as>{asof} · after IL tax 25%</div></div>
<div class=flow><span>1·<b>detect</b> regime</span><span>2·<b>scan</b> reversal+momentum</span><span>3·<b>read</b> news+fundamentals</span><span>4·<b>size</b> w/ guardrails</span><span>5·<b>you</b> trade</span></div>

{'<div class=warn>⚠ '+html.escape(dmsg)+' — decisions suppressed.</div>' if not ok else ''}

<div class="rg {regime_cls}"><div class=rst><span class=dot></span>{regime_txt}</div><div class=rmsg>{regime_msg}</div></div>

<div class=tag>▶ Your portfolio</div>
<div class=ord muted style="justify-content:flex-start;color:var(--tx)"><span class=oact style="color:var(--cy)">{'CASH' if flat else 'MIXED'}</span><span class=osym>{port_line}</span></div>

<div class=tag>▶ Suggested action — reversal edge (the validated one)</div>
{action}

<div class=tag>📉 Reversal agent · bounce candidates (heavy-volume losers, no leverage)</div>
<div class=pnl><table><tr><th>etf</th><th class=num>10d drop</th><th class=num>capit.</th><th>news · falling-knife?</th></tr>{rev_rows}</table></div>
<div class=note>The proven edge: oversold names that fell on heavy volume tend to bounce. News VETO = a justified drop (don't catch the knife).</div>

<div class=tag>🔥 Momentum radar · what's hot (not a proven edge — just info)</div>
<div class=pnl><table><tr><th>#</th><th>etf</th><th class=num>score</th><th>strength</th></tr>{mom_rows}</table></div>

<div class=tag>Storm-detector track record · honest (causal, hysteresis)</div>
<div class=stats>
<div class=stat><div class=l>Return / yr</div><div class=v2 style="color:var(--grn)">{s['CAGR%']:.1f}%</div><div class=c>buy&amp;hold {sbh['CAGR%']:.1f}%</div></div>
<div class=stat><div class=l>Smoothness</div><div class=v2>{s['Sharpe']:.2f}</div><div class=c>buy&amp;hold {sbh['Sharpe']:.2f}</div></div>
<div class=stat><div class=l>Worst crash</div><div class=v2 style="color:var(--amb)">{s['maxDD%']:.0f}%</div><div class=c>buy&amp;hold {sbh['maxDD%']:.0f}%</div></div>
</div>
{sparkline(eq)}
<div class=note>$1 → ${eq.iloc[-1]:.2f} (gated, after-tax). The gate trades return for crash protection — its job is defense.</div>

<div class=ft>Decision tool — never places trades; you trade manually via Interactive Israel.<br>
Reversal = statistically-validated edge (permutation p&lt;0.01). Momentum = a what's-moving radar, NOT proven. Leverage excluded. Backtest estimates, not promises.<br>
Refresh: <code>python3 dashboard.py</code></div>
</html>"""
    open("dashboard.html","w").write(doc)
    print(f"wrote dashboard.html — {regime_txt}, portfolio={'CASH' if flat else 'mixed'}, "
          f"{len([b for b in (rev_news or {}) if rev_news.get(b)!='VETO'])} reversal buys, gated Sharpe {s['Sharpe']:.2f}")


if __name__ == "__main__":
    main()
