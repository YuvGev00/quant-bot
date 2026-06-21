#!/usr/bin/env python3
"""dashboard.py — FUTURISTIC finance HUD. Self-contained dashboard.html (no server/CDN, emails fine).
Expanded universe (188 names, diversity-capped), real news verdicts, honest causal numbers.
Run: python3 dashboard.py
"""
from __future__ import annotations
import json, os, html
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, gate_returns, hysteresis, after_tax, stats, LAG
from early_scanner import panels, score_leader
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict
from bot_utils import data_quality
from universe import all_symbols, diversify, SECTOR, LEVERAGED

TOP_N = 5; MAX_PER_SECTOR = 2


def sparkline(series, w=720, h=130, color="#00e5ff"):
    v = series.dropna().values
    if len(v) < 2: return ""
    lo, hi = v.min(), v.max(); rng = (hi-lo) or 1
    pts = [(w*i/(len(v)-1), h-(h-18)*(x-lo)/rng-9) for i,x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" class=spark>'
            f'<polygon points="0,{h} {line} {w},{h}" fill="url(#gg)"/>'
            f'<polyline class=pline fill=none stroke="{color}" stroke-width=2 points="{line}"/>'
            f'<defs><linearGradient id=gg x1=0 x2=0 y1=0 y2=1><stop offset=0 stop-color="{color}" stop-opacity=.35/>'
            f'<stop offset=1 stop-color="{color}" stop-opacity=0/></linearGradient></defs></svg>')


def revscore(px, vol):
    pxf, volf = px.ffill(), vol.ffill()
    drop = -(pxf/pxf.shift(10)-1); relvol = (volf/volf.rolling(63).mean()).clip(.5,4.0)
    return (drop*relvol).iloc[-1].replace([np.inf,-np.inf],np.nan).dropna(), pxf


def main():
    syms = all_symbols(include_stocks=True)
    px, vol = panels(syms)
    ok, dmsg = data_quality(px)
    spy = load_close("SPY"); ro, ret, rf, _ = walk_forward(spy, None)
    risk_off = bool(ro.iloc[-1] > 0.5); asof = ro.index[-1].date()
    gen_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")   # when this dashboard was generated
    in_mkt = hysteresis((1-ro).shift(LAG).fillna(1.0),3,3)
    g = gate_returns(in_mkt, ret.loc[in_mkt.index], rf.loc[in_mkt.index])
    eq = (1+after_tax(g,0.25)).cumprod(); st = stats(after_tax(g,0.25)); sbh = stats(after_tax(ret.loc[g.index],0.25))

    rsc, pxf = revscore(px, vol)
    ranked = rsc[rsc>0].sort_values(ascending=False)
    rev_pick = diversify(list(ranked.index), TOP_N, MAX_PER_SECTOR)
    rev_top = ranked.head(10)
    rev_news = {}; news_date = "—"
    if os.path.exists("news_verdicts_reversal.json"):
        try:
            raw = json.load(open("news_verdicts_reversal.json"))
            import re as _re
            m = _re.search(r"(\d{4}-\d{2}-\d{2})", raw.get("_comment",""))
            if m: news_date = m.group(1)
            rev_news = {k.upper():v.upper() for k,v in raw.items() if not k.startswith("_")}
        except Exception: pass

    msc = score_leader(px.ffill()).iloc[-1].dropna()
    mom_top = msc.sort_values(ascending=False).head(8)

    # EPS-revision signal (forward experiment) — read the panel if it exists
    eps_rows = ""; eps_snaps = 0; eps_status = "no data yet"
    try:
        from eps_revision import load_panel, revision_signal
        ep = load_panel()
        if not ep.empty:
            eps_snaps = ep["date"].nunique()
            sig = revision_signal()
            if not sig.empty:
                top = sig.head(4); bot = sig.tail(3)
                for _, r in pd.concat([top, bot]).drop_duplicates("ticker").iterrows():
                    col = "var(--gr)" if r["revision%"] > 0 else ("var(--mg)" if r["revision%"] < 0 else "var(--dim)")
                    eps_rows += (f'<tr><td class=tk>{r["ticker"]}</td>'
                                 f'<td class=neg style="color:{col}">{r["revision%"]:+.1f}%</td>'
                                 f'<td class=sec>{int(r["n_analysts"])} analysts</td></tr>')
                eps_status = f"{eps_snaps} weekly snapshots — {'LIVE signal' if eps_snaps>=2 else 'collecting'}"
            else:
                eps_status = f"{eps_snaps} snapshot(s) — need 2+ for a revision"
    except Exception: pass

    # value-agent ideas (cheap+quality), if the agent wrote a snapshot; else compute a light version
    val_rows = ""
    try:
        from value_agent import value_score
        from universe import STOCK_SECTOR
        vcand = []
        for t in list(STOCK_SECTOR)[:0]:  # skip heavy live fetch in dashboard; read snapshot instead
            pass
        if os.path.exists("value_ideas.json"):
            vi = json.load(open("value_ideas.json"))
            for r in vi.get("ideas", [])[:6]:
                an = r.get("analyst","—"); acl = "v-good" if "buy" in an else ("v-bad" if "sell" in an else "v-caution" if an=="hold" else "v-mid")
                val_rows += (f'<tr><td class=tk>{r["ticker"]}</td><td class=sec>{r.get("sector","")}</td>'
                             f'<td class=neg style="color:var(--gr)">{r.get("score","")}</td>'
                             f'<td><span class="vd {acl}">{an}</span></td>'
                             f'<td class=sec>{r.get("thesis","")[:40]}</td></tr>')
    except Exception:
        pass

    # Breakout Scout — slim launcher (full multi-page research site lives in breakout_site/)
    bo_launch = ""
    if os.path.exists("breakout_ideas.json"):
        try:
            bo = json.load(open("breakout_ideas.json"))
            bideas = bo.get("ideas", [])
            vrank = {"CONFIRM": 0, "CAUTION": 1, "VETO": 2}
            bvc = {"CONFIRM": "var(--gr)", "CAUTION": "var(--am)", "VETO": "var(--mg)"}
            bordered = sorted(bideas, key=lambda i: vrank.get(i.get("verdict"), 9))
            lead = next((i for i in bordered if i.get("verdict") == "CONFIRM"), bordered[0] if bordered else None)
            nC = sum(1 for i in bideas if i.get("verdict") == "CONFIRM")
            nW = sum(1 for i in bideas if i.get("verdict") == "CAUTION")
            nV = sum(1 for i in bideas if i.get("verdict") == "VETO")
            chips = "".join(
                f'<a class="bo-chip" href="breakout_site/{i["ticker"]}.html" '
                f'style="border-color:{bvc.get(i.get("verdict"),"var(--dim)")};color:{bvc.get(i.get("verdict"),"var(--dim)")}">'
                f'{i["ticker"]} {i.get("upside_pct",0):+.0f}%</a>' for i in bordered)
            leadtxt = (f'<b style="color:var(--gr)">{lead["ticker"]}</b> — {html.escape(lead.get("thesis","")[:150])}…'
                       if lead else "run breakout_agent.py")
            bo_launch = (
                f'<div class=sec-h>Breakout Scout :: beaten-down blow-up candidates :: SPECULATIVE (not a backtested edge)'
                f'<a class=refresh href="breakout_site/index.html">⤢ OPEN RESEARCH SITE</a></div>'
                f'<div class="glass" style="padding:18px 22px">'
                f'<div style="font-size:11px;letter-spacing:1px;color:var(--dim);margin-bottom:10px">'
                f'{nC} CONFIRM · {nW} CAUTION · {nV} VETO · as of {html.escape(bo.get("_asof",""))}</div>'
                f'<div style="font-size:13px;line-height:1.6;margin-bottom:12px">{leadtxt}</div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap">{chips}</div>'
                f'<div class=note style="margin-top:12px">Full dossiers (charts, fingerprint scorecard, analyst panel, news, '
                f'same-sector cohort) in <b>breakout_site/index.html</b>. Conviction research → leads, not signals.</div></div>')
        except Exception:
            pass

    cur = {}
    if os.path.exists("holdings.json"):
        try: cur = json.load(open("holdings.json")).get("positions",{})
        except Exception: pass
    flat = (len(cur)==0)

    vtxt = {"CONFIRM":"BOUNCE","CAUTION":"WATCH","VETO":"KNIFE","UNREAD":"—"}
    buys = [s for s in rev_pick if rev_news.get(s,"UNREAD")!="VETO"]

    rev_rows=""
    for s in rev_top.index:
        d10=(pxf[s].iloc[-1]/pxf[s].iloc[-11]-1)*100; nv=rev_news.get(s,"UNREAD"); sec=SECTOR.get(s,"other")
        pk=" data-pick=1" if s in rev_pick else ""
        rev_rows+=(f'<tr{pk}><td class=tk>{s}</td><td class=sec>{sec}</td><td class=neg>{d10:+.1f}%</td>'
                   f'<td><span class="vd v-{nv.lower()}">{vtxt.get(nv,nv)}</span></td></tr>')
    mom_rows=""
    for i,(s,v) in enumerate(mom_top.items(),1):
        sec=SECTOR.get(s,"other")
        mom_rows+=(f'<tr><td class=ix>{i:02d}</td><td class=tk>{s}</td><td class=sec>{sec}</td>'
                   f'<td class=barcell><span class=bar style="width:{min(v/mom_top.iloc[0]*100,100):.0f}%"></span></td></tr>')

    regime_txt="RISK-OFF" if risk_off else "CALM"; rc="risk" if risk_off else "calm"
    if risk_off:
        act='<div class=act><div class=acth>◢ DEFENSIVE PROTOCOL</div><div class=actb>Storm regime engaged — capital to cash. No entries.</div></div>'
    elif flat and buys:
        n=len(buys)
        li="".join(f'<div class=buyrow><span class=bsym>{b}</span><span class=balloc>{int(100/n)}%</span><span class=bsec>{SECTOR.get(b,"")}</span></div>' for b in buys)
        act=f'<div class=act><div class=acth>◢ ACQUIRE // reversal edge</div>{li}<div class=actb>Portfolio empty → fresh entries. Remainder to cash. Execution: manual.</div></div>'
    elif flat:
        act='<div class=act><div class=acth>◢ HOLD CASH</div><div class=actb>No news-confirmed bounce candidate this cycle.</div></div>'
    else:
        act='<div class=act><div class=acth>◢ SEE WEEKLY UPLINK</div><div class=actb>Positions on file — manage per emailed sheet.</div></div>'
    port=f"100% CASH" if flat else " · ".join(f"{k} {v*100:.0f}%" for k,v in cur.items())
    warn_block = '<div class="glass warn">/!\\ ' + html.escape(dmsg) + ' — DECISIONS SUPPRESSED</div>' if not ok else ''

    doc=f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>QUANTBOT :: {regime_txt}</title>
<style>
@keyframes scan{{0%{{transform:translateY(-100%)}}100%{{transform:translateY(100vh)}}}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}
@keyframes flow{{to{{stroke-dashoffset:-1000}}}}
@keyframes glow{{0%,100%{{filter:drop-shadow(0 0 6px var(--cy))}}50%{{filter:drop-shadow(0 0 18px var(--cy))}}}}
:root{{--bg:#04070f;--gl:rgba(12,22,40,.55);--ln:rgba(0,229,255,.18);--cy:#00e5ff;--mg:#ff2d92;--gr:#1dffb0;--am:#ffcf3a;--tx:#bfe9ff;--dim:#5a7390}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--tx);font:13px/1.5 'SF Mono',ui-monospace,Menlo,monospace;max-width:860px;margin:0 auto;padding:30px 20px 80px;position:relative;overflow-x:hidden;
background-image:linear-gradient(rgba(0,229,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.045) 1px,transparent 1px),radial-gradient(1000px 500px at 50% -10%,rgba(0,229,255,.1),transparent),radial-gradient(800px 400px at 100% 8%,rgba(255,45,146,.07),transparent);
background-size:42px 42px,42px 42px,100% 100%,100% 100%}}
body::before{{content:"";position:fixed;left:0;right:0;top:0;height:140px;background:linear-gradient(180deg,transparent,rgba(0,229,255,.05),transparent);animation:scan 8s linear infinite;pointer-events:none;z-index:99}}
.hd{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--ln);padding-bottom:14px}}
.lg{{font-size:16px;letter-spacing:6px;font-weight:700;color:#fff;text-shadow:0 0 14px var(--cy)}} .lg span{{color:var(--cy)}}
.as{{font-size:10px;letter-spacing:2px;color:var(--dim)}}
.glass{{background:var(--gl);border:1px solid var(--ln);border-radius:14px;backdrop-filter:blur(8px);box-shadow:0 0 34px -14px var(--cy),inset 0 1px 0 rgba(255,255,255,.05)}}
.sec-h{{font-size:10px;letter-spacing:4px;color:var(--cy);text-transform:uppercase;margin:30px 0 12px;display:flex;align-items:center;gap:10px;text-shadow:0 0 8px var(--cy)}}
.sec-h::before{{content:"//"}} .sec-h::after{{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--ln),transparent)}}
.refresh{{color:var(--cy);border:1px solid var(--cy);border-radius:20px;padding:4px 12px;text-decoration:none;font-size:10px;letter-spacing:1px;transition:.2s;text-shadow:0 0 6px var(--cy)}}
.refresh:hover{{background:var(--cy);color:#04070f;box-shadow:0 0 18px var(--cy)}}
.newsstamp{{color:var(--dim);font-size:10.5px;letter-spacing:.5px;margin:-4px 0 10px}}
.reg{{margin-top:22px;padding:26px 28px;position:relative;overflow:hidden}}
.reg .ring{{position:absolute;right:-46px;top:-46px;width:170px;height:170px;border-radius:50%;border:2px solid var(--cy);opacity:.22;animation:glow 3s infinite}}
.reg.risk .ring{{border-color:var(--mg)}}
.rl{{font-size:9px;letter-spacing:4px;color:var(--dim)}}
.rv{{font-size:48px;font-weight:700;letter-spacing:2px;line-height:1;margin-top:6px}}
.calm .rv{{color:var(--gr);text-shadow:0 0 26px var(--gr)}} .risk .rv{{color:var(--mg);text-shadow:0 0 26px var(--mg)}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:15px;vertical-align:middle;animation:pulse 2s infinite}}
.calm .dot{{background:var(--gr);box-shadow:0 0 16px var(--gr)}} .risk .dot{{background:var(--mg);box-shadow:0 0 16px var(--mg);animation:pulse 1s infinite}}
.rm{{color:var(--dim);margin-top:10px;font-size:12px;letter-spacing:.5px}}
.act{{padding:20px 24px}} .acth{{color:var(--cy);font-size:13px;letter-spacing:2px;margin-bottom:10px;text-shadow:0 0 10px var(--cy)}}
.actb{{color:var(--dim);font-size:11.5px;margin-top:12px;letter-spacing:.4px}}
.buyrow{{display:flex;align-items:center;gap:16px;padding:10px 0;border-bottom:1px solid rgba(0,229,255,.08)}} .buyrow:last-of-type{{border:none}}
.bsym{{font-size:19px;font-weight:700;color:#fff;min-width:74px;letter-spacing:1px}} .balloc{{color:var(--gr);font-weight:700;text-shadow:0 0 10px var(--gr)}} .bsec{{color:var(--dim);font-size:11px;font-style:italic}}
.port{{padding:15px 24px;display:flex;align-items:center;gap:18px}} .port .pl{{font-size:9px;letter-spacing:3px;color:var(--dim)}} .port .pv{{font-size:18px;color:var(--cy);font-weight:700;letter-spacing:1px;text-shadow:0 0 10px var(--cy)}}
table{{width:100%;border-collapse:collapse}} .tbl{{padding:4px 6px}}
th{{font-size:9px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;text-align:left;padding:10px 12px;border-bottom:1px solid var(--ln);font-weight:500}}
td{{padding:10px 12px;border-bottom:1px solid rgba(0,229,255,.06);font-size:13px}}
tr:last-child td{{border:none}} tr[data-pick] td{{background:linear-gradient(90deg,rgba(0,229,255,.1),transparent)}}
td.tk{{font-weight:700;color:#fff;letter-spacing:1px}} td.sec{{color:var(--dim);font-size:11px;font-style:italic}}
td.neg{{color:var(--mg);text-align:right;font-variant-numeric:tabular-nums}} td.ix{{color:var(--dim);width:30px}}
.vd{{font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 10px;border-radius:20px;border:1px solid}}
.v-confirm{{color:var(--gr);border-color:var(--gr);box-shadow:0 0 12px -3px var(--gr)}}
.v-veto{{color:var(--mg);border-color:var(--mg)}} .v-caution{{color:var(--am);border-color:var(--am)}} .v-unread{{color:var(--dim);border-color:var(--dim)}}
.barcell{{width:42%}} .bar{{display:block;height:7px;border-radius:4px;background:linear-gradient(90deg,var(--mg),var(--cy));box-shadow:0 0 12px -2px var(--cy)}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.stat{{padding:18px 16px;text-align:center}} .stat .sl{{font-size:9px;letter-spacing:2px;color:var(--dim);text-transform:uppercase}}
.stat .sv{{font-size:30px;font-weight:700;margin-top:8px;font-variant-numeric:tabular-nums}} .stat .sc{{font-size:10px;color:var(--dim);margin-top:4px}}
.spark .pline{{stroke-dasharray:1000;animation:flow 20s linear infinite;filter:drop-shadow(0 0 6px var(--cy))}}
.note{{color:var(--dim);font-size:11px;margin-top:9px;letter-spacing:.3px;line-height:1.6}}
.warn{{padding:14px 18px;margin-top:16px;border-color:var(--am);color:var(--am)}}
.bo-chip{{font-size:11px;font-weight:700;letter-spacing:.5px;padding:5px 11px;border-radius:16px;border:1px solid;text-decoration:none;transition:.2s}}
.bo-chip:hover{{background:rgba(255,255,255,.06);box-shadow:0 0 12px -3px currentColor}}
.ft{{margin-top:42px;border-top:1px solid var(--ln);padding-top:16px;color:var(--dim);font-size:10px;letter-spacing:1px;line-height:1.8;text-align:center}}
</style></head><body>
<div class=hd><div class=lg>QUANT<span>::</span>BOT</div><div class=as>GENERATED {gen_time} :: DATA {asof} :: {len(syms)} ASSETS</div></div>

{warn_block}

<div class="glass reg {rc}"><div class=ring></div>
  <div class=rl>MARKET REGIME</div>
  <div class=rv><span class=dot></span>{regime_txt}</div>
  <div class=rm>{'Storm protocol engaged — defensive posture.' if risk_off else 'Systems nominal — edges online.'}</div>
</div>

<div class="glass port" style="margin-top:14px"><span class=pl>PORTFOLIO</span><span class=pv>{port}</span></div>

<div class=sec-h>Directive</div>
<div class="glass">{act}</div>

{bo_launch}

<div class=sec-h>Reversal Array :: validated edge :: diversified &le;{MAX_PER_SECTOR}/sector
  <a class=refresh href="https://claude.ai/code/routines" target=_blank title="Trigger the cloud agent to web-read fresh news (local can't browse)">⟳ REFRESH NEWS</a></div>
<div class=newsstamp>news scan as of <b>{news_date}</b> · click ⟳ to run the cloud agent for a live re-read</div>
<div class="glass tbl"><table><tr><th>ASSET</th><th>SECTOR</th><th style=text-align:right>10D &Delta;</th><th>NEWS SCAN</th></tr>{rev_rows}</table></div>
<div class=note>Oversold names that capitulated on heavy volume tend to rebound. <b>KNIFE</b> = decline justified by news → excluded. Glowing rows = selected.</div>

<div class=sec-h>Value Lab :: cheap + quality :: &le;$100/share :: SPECULATIVE ideas (not a backtested edge)</div>
<div class="glass tbl"><table><tr><th>ASSET</th><th>SECTOR</th><th>SCORE</th><th>ANALYSTS</th><th>THESIS</th></tr>{val_rows or '<tr><td colspan=5 class=sec>run value_agent.py to populate</td></tr>'}</table></div>
<div class=note>Cheap stocks that are also profitable/growing (not value traps). Each is a STARTING thesis to investigate — discretionary research, do your own diligence.</div>

<div class=sec-h>EPS-Revision Lab :: forward experiment :: {eps_status}</div>
<div class="glass tbl"><table><tr><th>ASSET</th><th>EPS REVISION</th><th>COVERAGE</th></tr>{eps_rows or '<tr><td colspan=3 class=sec>collecting weekly snapshots — needs 2+ to show revisions (the cloud agent grows this)</td></tr>'}</table></div>
<div class=note>Rising forward EPS estimates (analysts upgrading) tend to predict outperformance. UNVALIDATED forward test — needs ~12 weekly snapshots before permutation-testing. Green=upgrade, magenta=downgrade.</div>

<div class=sec-h>Momentum Radar :: unverified :: informational</div>
<div class="glass tbl"><table><tr><th>#</th><th>ASSET</th><th>SECTOR</th><th>STRENGTH</th></tr>{mom_rows}</table></div>

<div class=sec-h>Defense Telemetry :: honest backtest</div>
<div class=stats>
<div class="glass stat"><div class=sl>RETURN/YR</div><div class=sv style="color:var(--gr);text-shadow:0 0 16px var(--gr)">{st['CAGR%']:.1f}%</div><div class=sc>hold {sbh['CAGR%']:.1f}%</div></div>
<div class="glass stat"><div class=sl>SHARPE</div><div class=sv style="color:var(--cy);text-shadow:0 0 16px var(--cy)">{st['Sharpe']:.2f}</div><div class=sc>hold {sbh['Sharpe']:.2f}</div></div>
<div class="glass stat"><div class=sl>MAX DRAWDOWN</div><div class=sv style="color:var(--mg);text-shadow:0 0 16px var(--mg)">{st['maxDD%']:.0f}%</div><div class=sc>hold {sbh['maxDD%']:.0f}%</div></div>
</div>
<div class="glass" style="margin-top:14px;padding:14px 16px 8px">{sparkline(eq)}<div class=note style="text-align:center">CAPITAL TRAJECTORY :: $1 &rarr; ${eq.iloc[-1]:.2f} :: gated / after-tax</div></div>

<div class=ft>QUANTBOT DECISION CORE :: NEVER AUTO-EXECUTES :: MANUAL FILL VIA INTERACTIVE ISRAEL<br>
REVERSAL = STAT-VALIDATED (p&lt;0.01) :: MOMENTUM = RADAR ONLY :: LEVERAGE EXCLUDED :: BACKTEST &ne; PROMISE</div>
</body></html>"""
    open("dashboard.html","w").write(doc)
    print(f"wrote dashboard.html — {regime_txt}, {len(syms)} assets, portfolio={'CASH' if flat else 'mixed'}, "
          f"reversal picks {rev_pick}, buys {buys}, Sharpe {st['Sharpe']:.2f}")


if __name__ == "__main__":
    main()
