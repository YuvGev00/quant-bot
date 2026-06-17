#!/usr/bin/env python3
"""dashboard.py — regenerate a self-contained, terminal-styled dashboard.html each run.
No server, no external fonts/CDN (works offline + emails cleanly). Leads with the ACTION
(what to do this month), then the evidence: regime, picks w/ fundamentals+news verdicts,
holdings, and the honest causal backtest. Reflects the 100%-satellite config.

Run:  python3 dashboard.py
"""
from __future__ import annotations
import json, os, html
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, gate_returns, hysteresis, after_tax, stats, LAG
from early_scanner import panels, score_leader, ETF_UNIVERSE
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict
from bot_utils import data_quality

LEVERAGED = {"SOXL", "TECL", "TQQQ", "QLD", "SSO", "UPRO", "SPXL", "FAS", "TNA", "UDOW", "ROM"}
MAX_SINGLE = 0.35
LEV_CAP = 0.20
TOP_N = 5


def sparkline(series, w=680, h=120, color="#4ade80"):
    v = series.dropna().values
    if len(v) < 2: return ""
    lo, hi = v.min(), v.max(); rng = (hi - lo) or 1
    pts = [(w * i / (len(v) - 1), h - (h - 14) * (x - lo) / rng - 7) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" class="spark">'
            f'<polygon points="{area}" fill="url(#g)" opacity="0.18"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{line}"/>'
            f'<defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1">'
            f'<stop offset="0" stop-color="{color}"/><stop offset="1" stop-color="{color}" stop-opacity="0"/>'
            f'</linearGradient></defs></svg>')


def main():
    px, _ = panels(ETF_UNIVERSE)
    ok, dmsg = data_quality(px)

    spy = load_close("SPY"); ro, ret, rf, lams = walk_forward(spy, None)
    risk_off = bool(ro.iloc[-1] > 0.5); asof = ro.index[-1].date()
    in_mkt = hysteresis((1 - ro).shift(LAG).fillna(1.0), 3, 3)
    g = gate_returns(in_mkt, ret.loc[in_mkt.index], rf.loc[in_mkt.index])
    eq = (1 + after_tax(g, 0.25)).cumprod()
    s = stats(after_tax(g, 0.25)); sbh = stats(after_tax(ret.loc[g.index], 0.25))

    sc = score_leader(px).iloc[-1].dropna()
    top10 = sc.sort_values(ascending=False).head(10)
    picks = sc[sc > 0].nlargest(TOP_N)
    cur = {}
    if os.path.exists("holdings.json"):
        try: cur = json.load(open("holdings.json")).get("positions", {})
        except Exception: pass
    news = {}
    if os.path.exists("news_verdicts.json"):
        try: news = {k.upper(): v.upper() for k, v in json.load(open("news_verdicts.json")).items() if not k.startswith("_")}
        except Exception: pass

    # fundamentals + combined verdict per pick → target weights (the ACTION)
    rows_pick = []
    target = {}
    for sym, mom in picks.items():
        fscore = health_score(fetch_fundamentals(sym))["score"]
        fv = fundamental_verdict(fscore)
        nv = news.get(sym, "UNREAD")
        order = {"VETO": 0, "CAUTION": 1, "UNREAD": 2, "NEUTRAL": 2, "CONFIRM": 3}
        cv = {0: "VETO", 1: "CAUTION", 2: "NEUTRAL", 3: "CONFIRM"}[min(order.get(fv, 2), order.get(nv, 2))]
        w = {"CONFIRM": 1.0, "NEUTRAL": 1.0, "CAUTION": 0.5, "VETO": 0.0}[cv]
        rows_pick.append({"sym": sym, "mom": float(mom), "fund": fv, "fscore": fscore, "news": nv, "verdict": cv, "w": w})
    if not risk_off:
        live = [r for r in rows_pick if r["w"] > 0]
        tot = sum(r["w"] for r in live) or 1
        for r in live:
            r["wt"] = min(r["w"] / tot, MAX_SINGLE)
        # leverage cap
        levw = sum(r.get("wt", 0) for r in live if r["sym"] in LEVERAGED)
        if levw > LEV_CAP and any(r["sym"] not in LEVERAGED for r in live):
            for r in live:
                if r["sym"] in LEVERAGED: r["wt"] *= LEV_CAP / levw
        target = {r["sym"]: r.get("wt", 0) for r in live}
        tt = sum(target.values()) or 1
        target = {k: v / tt * (1 - max(0, 1 - sum(r.get("wt",0) for r in live))) if False else v for k, v in target.items()}

    # build orders (action) — compare target vs current
    orders = []
    allt = sorted(set(cur) | set(target))
    for t in allt:
        c = cur.get(t, 0.0); tg = target.get(t, 0.0); d = tg - c
        if risk_off and c > 0: act, cls = "SELL → cash", "sell"
        elif abs(d) <= 0.05: act, cls = ("HOLD", "hold") if c > 0 else ("—", "muted")
        elif c == 0: act, cls = f"BUY {tg*100:.0f}%", "buy"
        elif tg == 0: act, cls = "SELL all", "sell"
        elif d > 0: act, cls = f"ADD → {tg*100:.0f}%", "buy"
        else: act, cls = f"TRIM → {tg*100:.0f}%", "sell"
        if act not in ("—",):
            orders.append((t, c, tg, act, cls))

    # ---- render ----
    regime_txt = "RISK-OFF" if risk_off else "CALM"
    regime_cls = "risk" if risk_off else "calm"
    regime_msg = "Storm detected — go to cash. Ignore the picks below." if risk_off else "Clear weather — momentum picks are live."

    pick_rows = ""
    vmap = {"CONFIRM": "v-good", "NEUTRAL": "v-mid", "CAUTION": "v-warn", "VETO": "v-bad", "UNREAD": "v-mid"}
    for r in sorted(rows_pick, key=lambda x: -x["mom"]):
        lev = '<span class="chip lev">3×</span>' if r["sym"] in LEVERAGED else ""
        fs = f'{r["fscore"]:.0f}' if r["fscore"] is not None else '—'
        pick_rows += (f'<tr><td class=t>{r["sym"]}{lev}</td><td class=num>{r["mom"]:.2f}</td>'
                      f'<td class=num>{fs}</td><td><span class="v {vmap[r["fund"]]}">{r["fund"]}</span></td>'
                      f'<td><span class="v {vmap[r["news"]]}">{r["news"]}</span></td>'
                      f'<td><span class="v {vmap[r["verdict"]]}">{r["verdict"]}</span></td></tr>')

    top_rows = ""
    for i, (sym, val) in enumerate(top10.items(), 1):
        cls = "star" if sym in picks.index else ""
        lev = '<span class="chip lev">3×</span>' if sym in LEVERAGED else ""
        held = '<span class="chip held">held</span>' if sym in cur else ""
        top_rows += f'<tr class="{cls}"><td class=rk>{i}</td><td class=t>{sym}{lev}{held}</td><td class=num>{val:.2f}</td><td class=bar><span style="width:{min(val/top10.iloc[0]*100,100):.0f}%"></span></td></tr>'

    order_html = ""
    if not orders:
        order_html = '<div class="ord muted">No changes — hold current positions.</div>'
    else:
        for t, c, tg, act, cls in orders:
            order_html += f'<div class="ord {cls}"><span class="oact">{act}</span><span class="osym">{t}</span></div>'
    cash_tgt = max(0, 100 - sum(target.values()) * 100) if not risk_off else 100

    doc = f"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>QUANT BOT // {regime_txt}</title>
<style>
:root{{--bg:#0a0e14;--panel:#111722;--panel2:#0d131c;--line:#1e2a3a;--txt:#c8d3e0;--dim:#6b7a8d;
--green:#4ade80;--red:#f87171;--amber:#fbbf24;--accent:#38bdf8;--mono:'SF Mono',ui-monospace,'Cascadia Code',Menlo,monospace}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--txt);font:14px/1.55 var(--mono);
background-image:radial-gradient(circle at 20% -10%,rgba(56,189,248,.06),transparent 40%),radial-gradient(circle at 90% 0%,rgba(74,222,128,.04),transparent 35%);
max-width:840px;margin:0 auto;padding:28px 20px 60px;-webkit-font-smoothing:antialiased}}
.head{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:4px}}
.logo{{font-size:13px;letter-spacing:3px;color:var(--accent);font-weight:600}}
.logo b{{color:var(--txt)}} .asof{{color:var(--dim);font-size:12px;letter-spacing:1px}}
.tag{{color:var(--dim);font-size:11px;letter-spacing:1.5px;text-transform:uppercase;margin:18px 0 8px;display:flex;align-items:center;gap:8px}}
.tag::after{{content:"";flex:1;height:1px;background:var(--line)}}
.regime{{margin-top:20px;border:1px solid var(--line);border-radius:12px;padding:20px 22px;position:relative;overflow:hidden;background:var(--panel)}}
.regime.calm{{border-color:rgba(74,222,128,.35);box-shadow:0 0 40px -18px var(--green) inset}}
.regime.risk{{border-color:rgba(248,113,113,.4);box-shadow:0 0 40px -16px var(--red) inset}}
.rstate{{font-size:34px;font-weight:700;letter-spacing:1px;line-height:1}}
.calm .rstate{{color:var(--green)}} .risk .rstate{{color:var(--red)}}
.rdot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:12px;vertical-align:middle}}
.calm .rdot{{background:var(--green);box-shadow:0 0 12px var(--green);animation:p 2.4s infinite}}
.risk .rdot{{background:var(--red);box-shadow:0 0 12px var(--red);animation:p 1.1s infinite}}
@keyframes p{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}
.rmsg{{color:var(--dim);margin-top:8px;font-size:13px}}
.orders{{display:flex;flex-direction:column;gap:7px;margin-top:6px}}
.ord{{display:flex;align-items:center;gap:14px;background:var(--panel);border:1px solid var(--line);border-left-width:3px;border-radius:8px;padding:11px 16px}}
.ord .oact{{font-weight:600;min-width:130px}} .ord .osym{{color:var(--dim);letter-spacing:1px}}
.ord.buy{{border-left-color:var(--green)}} .ord.buy .oact{{color:var(--green)}}
.ord.sell{{border-left-color:var(--red)}} .ord.sell .oact{{color:var(--red)}}
.ord.hold{{border-left-color:var(--accent)}} .ord.hold .oact{{color:var(--accent)}}
.ord.muted{{border-left-color:var(--line);color:var(--dim);justify-content:center}}
.cashline{{color:var(--dim);font-size:12px;margin-top:8px;text-align:right}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 4px;margin-top:6px}}
table{{width:100%;border-collapse:collapse}}
th{{color:var(--dim);font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-weight:500;text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}}
td{{padding:9px 12px;border-bottom:1px solid rgba(30,42,58,.5)}}
tr:last-child td{{border-bottom:none}}
td.t{{font-weight:600;color:#e6eef8;letter-spacing:.5px}} td.num{{text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}}
td.rk{{color:var(--dim);width:28px}} tr.star td.t{{color:var(--accent)}}
td.bar{{width:30%}} td.bar span{{display:block;height:6px;background:linear-gradient(90deg,var(--accent),var(--green));border-radius:3px}}
.chip{{font-size:9px;letter-spacing:.5px;padding:2px 5px;border-radius:4px;margin-left:7px;vertical-align:middle}}
.chip.lev{{background:rgba(251,191,36,.15);color:var(--amber);border:1px solid rgba(251,191,36,.3)}}
.chip.held{{background:rgba(56,189,248,.12);color:var(--accent)}}
.v{{font-size:11px;font-weight:600;letter-spacing:.5px;padding:2px 8px;border-radius:5px}}
.v-good{{background:rgba(74,222,128,.14);color:var(--green)}} .v-mid{{background:rgba(107,122,141,.18);color:var(--dim)}}
.v-warn{{background:rgba(251,191,36,.14);color:var(--amber)}} .v-bad{{background:rgba(248,113,113,.16);color:var(--red)}}
.stats{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:6px}}
.stat{{background:var(--panel);padding:16px 18px}}
.stat .l{{color:var(--dim);font-size:10px;letter-spacing:1.5px;text-transform:uppercase}}
.stat .v2{{font-size:24px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}}
.stat .c{{color:var(--dim);font-size:11px;margin-top:2px}}
.spark{{margin-top:14px;display:block}}
.warn{{background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.4);color:var(--amber);border-radius:10px;padding:14px 18px;margin-top:18px;font-size:13px}}
.foot{{color:var(--dim);font-size:11px;margin-top:34px;border-top:1px solid var(--line);padding-top:14px;line-height:1.7}}
.flow{{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;font-size:11px;color:var(--dim)}}
.flow span{{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:4px 11px}}
.flow b{{color:var(--accent)}}
</style>

<div class="head">
  <div class="logo">QUANT<b>BOT</b> // 100% SATELLITE</div>
  <div class="asof">{asof} · after IL tax 25%</div>
</div>
<div class="flow"><span>1 · <b>detect</b> regime</span><span>2 · <b>scan</b> momentum</span><span>3 · <b>read</b> fundamentals+news</span><span>4 · <b>decide</b> + size</span><span>5 · <b>you</b> trade</span></div>

{'<div class="warn">⚠ '+html.escape(dmsg)+' — decisions suppressed this run.</div>' if not ok else ''}

<div class="regime {regime_cls}">
  <div class="rstate"><span class="rdot"></span>{regime_txt}</div>
  <div class="rmsg">{regime_msg}</div>
</div>

<div class="tag">▶ Action this month</div>
<div class="orders">{order_html}</div>
<div class="cashline">target cash: {cash_tgt:.0f}%</div>

<div class="tag">Picks · fundamentals + news verdict</div>
<div class="panel"><table>
<tr><th>etf</th><th class=num>mom</th><th class=num>health</th><th>fundamentals</th><th>news</th><th>final</th></tr>
{pick_rows}
</table></div>

<div class="tag">Top 10 momentum · scanned {len(sc)} ETFs</div>
<div class="panel"><table>
<tr><th>#</th><th>etf</th><th class=num>score</th><th>strength</th></tr>
{top_rows}
</table></div>

<div class="tag">Storm-detector track record · honest (causal, hysteresis)</div>
<div class="stats">
  <div class="stat"><div class=l>Return / yr</div><div class=v2 style="color:var(--green)">{s['CAGR%']:.1f}%</div><div class=c>buy&hold {sbh['CAGR%']:.1f}%</div></div>
  <div class="stat"><div class=l>Smoothness</div><div class=v2>{s['Sharpe']:.2f}</div><div class=c>buy&hold {sbh['Sharpe']:.2f}</div></div>
  <div class="stat"><div class=l>Worst crash</div><div class=v2 style="color:var(--amber)">{s['maxDD%']:.0f}%</div><div class=c>buy&hold {sbh['maxDD%']:.0f}%</div></div>
</div>
{sparkline(eq)}

<div class="foot">
  This is a <b>decision tool</b> — it never places trades. You place orders manually via Interactive Israel.<br>
  Storm-detector gates equity exposure; momentum ranks ~80 ETFs; fundamentals + a live news read veto froth;
  guardrails cap leverage at {int(LEV_CAP*100)}% &amp; any single position at {int(MAX_SINGLE*100)}%.<br>
  Numbers are after-tax backtest estimates, not promises. Re-run <code>python3 dashboard.py</code> to refresh.
</div>
</html>"""
    open("dashboard.html", "w").write(doc)
    print(f"wrote dashboard.html — regime={regime_txt}, {len(orders)} orders, top pick {top10.index[0]}, gated Sharpe {s['Sharpe']:.2f}")


if __name__ == "__main__":
    main()
