#!/usr/bin/env python3
"""Persona probe dashboard — self-contained interactive HTML from episode JSONL logs.

Three panels:
  1. Trait time-series  — how projections evolve across 10-token chunks, averaged
                          over all completions for the selected agent
  2. Agent heatmap      — per-agent mean trait projections, all agents at once
  3. Delta bars         — steered-vs-control trait shifts (requires --control)

Usage
─────
python scripts/textarena/probe_dashboard.py \\
    --steered  logs/debate/debate_activation_p1_charismatic_2p \\
    --control  logs/debate/debate_noop_probe_2p \\
    --out      reports/debate_charismatic_dashboard.html
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Data loading ──────────────────────────────────────────────────────────────

def _extract_probe(rec: dict) -> Tuple[Dict, List]:
    """Return (mean_dict, chunks) handling old flat-dict and new {mean,chunks} format."""
    probe = rec.get("persona_probe") or {}
    if "mean" in probe:
        return probe["mean"], probe.get("chunks", [])
    # old flat format: treat whole dict as mean, no chunk data
    return {k: v for k, v in probe.items() if isinstance(v, (int, float))}, []


def load_records(run_dir: str) -> List[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "episode_*.jsonl"), recursive=True)):
        try:
            for line in open(path, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mean, _ = _extract_probe(rec)
                    if mean:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(records: List[dict], top_k_ts: int = 6, top_k_hm: int = 22) -> dict:
    mean_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mean_n: Dict[str, int] = defaultdict(int)
    # chunk_scores[agent][trait][chunk_idx] = [score, ...]
    chunk_scores: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:
        aid = rec["agent_id"]
        mean_dict, chunks = _extract_probe(rec)
        for trait, score in mean_dict.items():
            mean_sum[aid][trait] += score
        mean_n[aid] += 1
        for chunk in chunks:
            scores = chunk.get("scores", {})
            # token is the cumulative token count at end of chunk; chunk index = token/window - 1
            token = chunk.get("token", 0)
            cidx = max(0, (token // 10) - 1)
            for trait, score in scores.items():
                lst = chunk_scores[aid][trait]
                while len(lst) <= cidx:
                    lst.append([])
                lst[cidx].append(score)

    agents = sorted(mean_n.keys(), key=lambda a: int("".join(filter(str.isdigit, a)) or "0"))

    means: Dict[str, Dict[str, float]] = {
        aid: {t: s / mean_n[aid] for t, s in mean_sum[aid].items()}
        for aid in agents
    }

    all_traits = sorted({t for m in means.values() for t in m})

    # Heatmap traits: top by max |mean| across agents
    hm_traits = sorted(
        all_traits,
        key=lambda t: -max(abs(means[a].get(t, 0.0)) for a in agents),
    )[:top_k_hm]

    # Time-series: find max chunk depth we actually have
    max_chunks = min(
        max(
            (len(v) for a in chunk_scores.values() for v in a.values()),
            default=1,
        ),
        35,
    )

    # TS traits: most variance across chunk positions (most dynamic)
    def ts_interest(trait: str) -> float:
        vals = [
            s
            for aid in agents
            for cidx in range(max_chunks)
            for s in (
                chunk_scores[aid].get(trait, [])[cidx]
                if cidx < len(chunk_scores[aid].get(trait, []))
                else []
            )
        ]
        if len(vals) < 2:
            return abs(means.get(agents[0] if agents else "", {}).get(trait, 0.0))
        mu = sum(vals) / len(vals)
        return sum((v - mu) ** 2 for v in vals) / len(vals)

    ts_traits = sorted(all_traits, key=lambda t: -ts_interest(t))[:top_k_ts]

    # Average chunk trajectories per agent per ts_trait
    ts: Dict[str, Dict[str, List]] = {}
    for aid in agents:
        ts[aid] = {}
        for trait in ts_traits:
            row: List = []
            for cidx in range(max_chunks):
                bucket = chunk_scores[aid].get(trait, [])
                vals = bucket[cidx] if cidx < len(bucket) else []
                row.append(round(sum(vals) / len(vals), 5) if vals else None)
            ts[aid][trait] = row

    return {
        "agents": agents,
        "means": {a: {t: round(v, 5) for t, v in m.items()} for a, m in means.items()},
        "ts": ts,
        "ts_traits": ts_traits,
        "ts_max_chunks": max_chunks,
        "hm_traits": hm_traits,
    }


def compute_deltas(
    steered_agg: dict, control_agg: Optional[dict]
) -> Optional[Dict[str, Dict[str, float]]]:
    if control_agg is None:
        return None
    deltas: Dict[str, Dict[str, float]] = {}
    for aid in steered_agg["agents"]:
        s = steered_agg["means"].get(aid, {})
        c = control_agg["means"].get(aid, {})
        deltas[aid] = {
            t: round(s.get(t, 0.0) - c.get(t, 0.0), 5)
            for t in set(s) | set(c)
        }
    return deltas


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML = r"""<!doctype html>
<title>{title}</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
/* ── tokens ──────────────────────────────────────────────────────────────── */
:root {{
  --bg:          #f7f6f2;
  --surface:     #ffffff;
  --border:      #e2e1da;
  --text:        #111110;
  --text-2:      #56544e;
  --text-3:      #908e86;
  --grid:        #ece9e1;
  /* categorical: validated adjacent-pairs, slots 1–6 */
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a;
  --c4: #eda100; --c5: #e87ba4; --c6: #008300;
  /* diverging poles */
  --pos: #2a78d6;
  --neg: #e34948;
  --mid: #f0efec;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:      #111110;
    --surface: #1a1a18;
    --border:  #2a2a28;
    --text:    #f0efe8;
    --text-2:  #b8b6ae;
    --text-3:  #7a7872;
    --grid:    #242420;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70;
    --c4: #c98500; --c5: #d55181; --c6: #008300;
    --pos: #3987e5;
    --neg: #e66767;
    --mid: #383835;
  }}
}}
:root[data-theme="dark"] {{
  --bg:      #111110;
  --surface: #1a1a18;
  --border:  #2a2a28;
  --text:    #f0efe8;
  --text-2:  #b8b6ae;
  --text-3:  #7a7872;
  --grid:    #242420;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70;
  --c4: #c98500; --c5: #d55181; --c6: #008300;
  --pos: #3987e5;
  --neg: #e66767;
  --mid: #383835;
}}

/* ── base ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0 0 48px;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  font-variant-numeric: tabular-nums;
}}
h1 {{ font-size: 18px; font-weight: 600; margin: 0; }}
h2 {{ font-size: 13px; font-weight: 600; letter-spacing: .04em;
      text-transform: uppercase; color: var(--text-2); margin: 0 0 4px; }}
p.sub {{ font-size: 12px; color: var(--text-3); margin: 2px 0 12px; }}

.wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 24px; }}
header {{ padding: 28px 24px 20px; max-width: 1120px; margin: 0 auto; }}
.run-badge {{
  display: inline-block; font-size: 11px; padding: 2px 8px;
  border-radius: 4px; background: var(--border); color: var(--text-2);
  margin-top: 6px; margin-right: 6px;
}}

.panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 16px;
}}

.controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }}
label {{ font-size: 12px; color: var(--text-2); margin-right: 4px; }}
select {{
  font: inherit; font-size: 12px;
  background: var(--bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px;
  padding: 3px 8px; cursor: pointer;
}}

/* chart scroll containers */
.chart-scroll {{ overflow-x: auto; }}
svg {{ display: block; overflow: visible; }}

/* tooltip */
#tip {{
  position: fixed; pointer-events: none; display: none;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 11px;
  font-size: 12px; line-height: 1.6;
  box-shadow: 0 2px 8px rgba(0,0,0,.15);
  max-width: 260px; z-index: 99;
}}
#tip strong {{ font-weight: 600; display: block; margin-bottom: 4px;
               font-size: 11px; color: var(--text-2); }}

/* theme toggle */
.theme-btn {{
  position: fixed; top: 12px; right: 16px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 10px;
  font: 12px system-ui; cursor: pointer; color: var(--text-2);
  z-index: 10;
}}
</style>

<button class="theme-btn" onclick="toggleTheme()">◑ theme</button>
<div id="tip"></div>

<header>
  <h1>{title}</h1>
  <div>
    <span class="run-badge">steered: {steered_name}</span>
    {control_badge}
  </div>
</header>

<div class="wrap">
  <!-- Panel 1: Time Series -->
  <div class="panel" id="p-ts">
    <h2>Trait projections over token chunks</h2>
    <p class="sub">Each line is the average cosine projection at that 10-token window across all completions for the selected agent.</p>
    <div class="controls">
      <div><label>Agent</label><select id="ts-agent"></select></div>
    </div>
    <div class="chart-scroll"><svg id="ts-svg"></svg></div>
  </div>

  <!-- Panel 2: Agent heatmap -->
  <div class="panel" id="p-hm">
    <h2>Per-agent mean projections</h2>
    <p class="sub">Mean cosine projection across all turns and episodes. Blue = aligned with trait direction, red = opposite.</p>
    <div class="chart-scroll"><svg id="hm-svg"></svg></div>
  </div>

  <!-- Panel 3: Delta bars -->
  <div class="panel" id="p-delta" style="display:none">
    <h2>Steered vs control — trait shift</h2>
    <p class="sub">Δ = steered mean − control mean, per agent. ★ marks |Δ| ≥ 0.05.</p>
    <div class="controls">
      <div><label>Agent</label><select id="delta-agent"></select></div>
    </div>
    <div class="chart-scroll"><svg id="delta-svg"></svg></div>
  </div>
</div>

<script>
// ── embedded data ─────────────────────────────────────────────────────────────
const DATA = {data_json};

// ── palette ───────────────────────────────────────────────────────────────────
function cssVar(name) {{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}}

const CAT = ['--c1','--c2','--c3','--c4','--c5','--c6'];

function divColor(v, lo, hi) {{
  // diverging: lo=most negative, hi=most positive, 0 = mid
  const abs_max = Math.max(Math.abs(lo), Math.abs(hi), 0.001);
  const t = Math.max(-1, Math.min(1, v / abs_max));
  const pos = cssVar('--pos');
  const neg = cssVar('--neg');
  const mid = cssVar('--mid');
  return t >= 0 ? lerpHex(mid, pos, t) : lerpHex(mid, neg, -t);
}}

function hexToRgb(h) {{
  h = h.replace('#','');
  if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}}
function rgbToHex([r,g,b]) {{
  return '#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
}}
function lerpHex(a, b, t) {{
  const ra = hexToRgb(a), rb = hexToRgb(b);
  return rgbToHex(ra.map((v,i) => v + (rb[i]-v)*t));
}}

// ── SVG helpers ───────────────────────────────────────────────────────────────
function svgEl(tag, attrs={{}}) {{
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}}
function svgText(txt, attrs={{}}) {{
  const el = svgEl('text', attrs);
  el.textContent = txt;
  return el;
}}

// ── tooltip ───────────────────────────────────────────────────────────────────
const tip = document.getElementById('tip');
function showTip(html, x, y) {{
  tip.innerHTML = html;
  tip.style.display = 'block';
  const W = window.innerWidth, H = window.innerHeight;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = (x + 14 + tw > W ? x - tw - 10 : x + 14) + 'px';
  tip.style.top  = (y + th + 10 > H ? y - th - 6  : y + 10) + 'px';
}}
function hideTip() {{ tip.style.display = 'none'; }}

// ── Panel 1: Time series ──────────────────────────────────────────────────────
const TS_W = 700, TS_H = 280;
const TS_ML = 58, TS_MR = 170, TS_MT = 16, TS_MB = 44;
const TS_PW = TS_W - TS_ML - TS_MR, TS_PH = TS_H - TS_MT - TS_MB;

function buildTs(agentId) {{
  const svg = document.getElementById('ts-svg');
  svg.setAttribute('viewBox', `0 0 ${{TS_W}} ${{TS_H}}`);
  svg.setAttribute('width', TS_W); svg.setAttribute('height', TS_H);
  svg.innerHTML = '';

  const traits = DATA.ts_traits;
  const maxC = DATA.ts_max_chunks;
  const agData = DATA.ts[agentId] || {{}};

  // y domain
  let yLo = Infinity, yHi = -Infinity;
  for (const tr of traits) {{
    for (const v of (agData[tr] || [])) {{
      if (v !== null) {{ yLo = Math.min(yLo, v); yHi = Math.max(yHi, v); }}
    }}
  }}
  if (!isFinite(yLo)) {{ yLo = -0.3; yHi = 0.3; }}
  const pad = (yHi - yLo) * 0.12 || 0.05;
  yLo -= pad; yHi += pad;

  function xp(i) {{ return TS_ML + (i / (maxC - 1 || 1)) * TS_PW; }}
  function yp(v) {{ return TS_MT + (1 - (v - yLo) / (yHi - yLo)) * TS_PH; }}

  const g = cssVar('--grid'), t2 = cssVar('--text-2'), t3 = cssVar('--text-3');
  const textClr = cssVar('--text');

  // gridlines + y-axis ticks
  const nY = 5;
  for (let i = 0; i <= nY; i++) {{
    const v = yLo + (yHi - yLo) * i / nY;
    const y = yp(v);
    svg.appendChild(svgEl('line', {{x1: TS_ML, y1: y, x2: TS_ML + TS_PW, y2: y,
      stroke: g, 'stroke-width': 1}}));
    svg.appendChild(svgText(v.toFixed(3), {{x: TS_ML - 6, y: y + 4,
      'text-anchor': 'end', 'font-size': 10, fill: t3}}));
  }}

  // zero line
  if (yLo < 0 && yHi > 0) {{
    const y0 = yp(0);
    svg.appendChild(svgEl('line', {{x1: TS_ML, y1: y0, x2: TS_ML + TS_PW, y2: y0,
      stroke: t3, 'stroke-width': 1, 'stroke-dasharray': '3,3'}}));
  }}

  // x-axis ticks
  const xStep = Math.max(1, Math.floor(maxC / 8));
  for (let i = 0; i < maxC; i += xStep) {{
    const x = xp(i);
    svg.appendChild(svgEl('line', {{x1: x, y1: TS_MT + TS_PH, x2: x,
      y2: TS_MT + TS_PH + 4, stroke: t3, 'stroke-width': 1}}));
    svg.appendChild(svgText(`${{(i+1)*10}}`, {{x, y: TS_MT + TS_PH + 16,
      'text-anchor': 'middle', 'font-size': 10, fill: t3}}));
  }}

  // axis labels
  svg.appendChild(svgText('tokens', {{x: TS_ML + TS_PW/2, y: TS_H - 4,
    'text-anchor': 'middle', 'font-size': 11, fill: t2}}));
  const yLabel = svgText('projection', {{x: 12, y: TS_MT + TS_PH/2,
    'text-anchor': 'middle', 'font-size': 11, fill: t2,
    transform: `rotate(-90,12,${{TS_MT + TS_PH/2}})`}});
  svg.appendChild(yLabel);

  // axis border
  svg.appendChild(svgEl('line', {{x1: TS_ML, y1: TS_MT, x2: TS_ML,
    y2: TS_MT + TS_PH, stroke: cssVar('--border'), 'stroke-width': 1}}));
  svg.appendChild(svgEl('line', {{x1: TS_ML, y1: TS_MT + TS_PH,
    x2: TS_ML + TS_PW, y2: TS_MT + TS_PH,
    stroke: cssVar('--border'), 'stroke-width': 1}}));

  // hover crosshair
  const xhair = svgEl('line', {{y1: TS_MT, y2: TS_MT + TS_PH,
    stroke: t3, 'stroke-width': 1, 'stroke-dasharray': '2,2', display: 'none'}});
  svg.appendChild(xhair);

  // lines
  const colors = CAT.map(v => cssVar(v));
  for (let ti = 0; ti < traits.length; ti++) {{
    const trait = traits[ti];
    const vals = agData[trait] || [];
    const col = colors[ti % colors.length];
    let d = ''; let prev = null;
    for (let i = 0; i < maxC; i++) {{
      const v = vals[i];
      if (v === null || v === undefined) {{ prev = null; continue; }}
      const x = xp(i), y = yp(v);
      d += (prev === null ? `M${{x}},${{y}}` : `L${{x}},${{y}}`);
      prev = [x, y];
    }}
    if (d) svg.appendChild(svgEl('path', {{d, fill: 'none', stroke: col,
      'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}}));
  }}

  // dot markers + hover overlay
  for (let i = 0; i < maxC; i++) {{
    const x = xp(i);
    const overlay = svgEl('rect', {{
      x: x - (TS_PW / maxC / 2), y: TS_MT,
      width: TS_PW / maxC, height: TS_PH,
      fill: 'transparent', cursor: 'crosshair'
    }});
    overlay.addEventListener('mousemove', e => {{
      xhair.setAttribute('x1', x); xhair.setAttribute('x2', x);
      xhair.removeAttribute('display');
      let html = `<strong>token ${{(i+1)*10}}</strong>`;
      for (let ti = 0; ti < traits.length; ti++) {{
        const v = (agData[traits[ti]] || [])[i];
        const col = colors[ti % colors.length];
        const vStr = v !== null && v !== undefined ? v.toFixed(4) : '—';
        html += `<div style="color:${{col}}">${{traits[ti]}}: ${{vStr}}</div>`;
      }}
      showTip(html, e.clientX, e.clientY);
    }});
    overlay.addEventListener('mouseleave', () => {{
      xhair.setAttribute('display', 'none'); hideTip();
    }});
    svg.appendChild(overlay);
  }}

  // legend
  for (let ti = 0; ti < traits.length; ti++) {{
    const col = colors[ti % colors.length];
    const lx = TS_ML + TS_PW + 14, ly = TS_MT + 8 + ti * 18;
    svg.appendChild(svgEl('line', {{x1: lx, y1: ly + 5, x2: lx + 20, y2: ly + 5,
      stroke: col, 'stroke-width': 2, 'stroke-linecap': 'round'}}));
    svg.appendChild(svgText(traits[ti], {{x: lx + 24, y: ly + 9,
      'font-size': 11, fill: cssVar('--text-2')}}));
  }}
}}

// ── Panel 2: Heatmap ──────────────────────────────────────────────────────────
function buildHeatmap() {{
  const svg = document.getElementById('hm-svg');
  svg.innerHTML = '';

  const agents = DATA.agents;
  const traits = DATA.hm_traits;
  const means  = DATA.means;

  const ROW = 26, COL = 48, ML = 68, MT = 70, MB = 8;
  const W = ML + traits.length * COL + 2;
  const H = MT + agents.length * ROW + MB;

  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);

  // global range for color scaling
  let lo = Infinity, hi = -Infinity;
  for (const a of agents) for (const t of traits) {{
    const v = (means[a] || {{}})[t] || 0;
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }}

  const t2 = cssVar('--text-2'), t3 = cssVar('--text-3');

  // column headers (rotated)
  for (let ci = 0; ci < traits.length; ci++) {{
    const x = ML + ci * COL + COL / 2;
    const lbl = svgText(traits[ci], {{
      x: 0, y: 0,
      'font-size': 10, fill: t2, 'text-anchor': 'start',
      transform: `translate(${{x}},${{MT - 4}}) rotate(-45)`
    }});
    svg.appendChild(lbl);
  }}

  // rows
  for (let ri = 0; ri < agents.length; ri++) {{
    const aid = agents[ri];
    const y = MT + ri * ROW;
    // row label
    svg.appendChild(svgText(aid, {{x: ML - 6, y: y + ROW/2 + 4,
      'text-anchor': 'end', 'font-size': 11, fill: cssVar('--text')}}));

    for (let ci = 0; ci < traits.length; ci++) {{
      const trait = traits[ci];
      const v = (means[aid] || {{}})[trait] || 0;
      const col = divColor(v, lo, hi);
      const x = ML + ci * COL;
      const cell = svgEl('rect', {{x: x+1, y: y+1, width: COL-2, height: ROW-2,
        fill: col, rx: 2}});
      cell.addEventListener('mousemove', e => {{
        showTip(`<strong>${{aid}} · ${{trait}}</strong>${{v.toFixed(4)}}`, e.clientX, e.clientY);
      }});
      cell.addEventListener('mouseleave', hideTip);
      svg.appendChild(cell);

      // value label for larger cells
      if (COL >= 40) {{
        const lum = hexToRgb(col).reduce((a,v,i)=>a+v*[0.299,0.587,0.114][i],0);
        svg.appendChild(svgText(v.toFixed(2), {{
          x: x + COL/2, y: y + ROW/2 + 4,
          'text-anchor': 'middle', 'font-size': 9,
          fill: lum > 160 ? '#111' : '#eee'
        }}));
      }}
    }}
  }}

  // diverging legend bar
  const LX = ML, LY = H - MB + 4, LW = Math.min(200, traits.length * COL);
  const steps = 40;
  for (let i = 0; i < steps; i++) {{
    const v = lo + (hi - lo) * i / steps;
    svg.appendChild(svgEl('rect', {{
      x: LX + i * (LW/steps), y: LY, width: LW/steps + 0.5, height: 6,
      fill: divColor(v, lo, hi)
    }}));
  }}
  svg.appendChild(svgText(lo.toFixed(2), {{x: LX, y: LY+16, 'font-size': 9, fill: t3}}));
  svg.appendChild(svgText(hi.toFixed(2), {{x: LX+LW, y: LY+16, 'font-size': 9,
    fill: t3, 'text-anchor': 'end'}}));
  svg.setAttribute('height', H + 20);
}}

// ── Panel 3: Delta bars ───────────────────────────────────────────────────────
function buildDelta(agentId) {{
  const svg = document.getElementById('delta-svg');
  svg.innerHTML = '';
  if (!DATA.deltas) return;

  const deltas = DATA.deltas[agentId] || {{}};
  const ranked = Object.entries(deltas)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 30);

  const ROW = 20, ML = 180, MR = 60, MT = 16, MB = 10;
  const W = 720, H = MT + ranked.length * ROW + MB;
  const PW = W - ML - MR;

  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);

  const maxAbs = Math.max(...ranked.map(([,v]) => Math.abs(v)), 0.01);
  const cx = ML + PW / 2;

  // center zero line
  svg.appendChild(svgEl('line', {{x1: cx, y1: MT, x2: cx, y2: MT + ranked.length * ROW,
    stroke: cssVar('--border'), 'stroke-width': 1}}));

  // ±0.05 reference lines
  const ref5 = (0.05 / maxAbs) * (PW / 2);
  for (const sign of [1, -1]) {{
    svg.appendChild(svgEl('line', {{
      x1: cx + sign*ref5, y1: MT,
      x2: cx + sign*ref5, y2: MT + ranked.length * ROW,
      stroke: cssVar('--text-3'), 'stroke-width': 1, 'stroke-dasharray': '3,3'
    }}));
  }}

  for (let i = 0; i < ranked.length; i++) {{
    const [trait, delta] = ranked[i];
    const y = MT + i * ROW;
    const bw = (Math.abs(delta) / maxAbs) * (PW / 2);
    const col = delta >= 0 ? cssVar('--pos') : cssVar('--neg');
    const x0 = delta >= 0 ? cx : cx - bw;

    const bar = svgEl('rect', {{x: x0, y: y + 3, width: bw, height: ROW - 6,
      fill: col, rx: 2, opacity: 0.85}});
    bar.addEventListener('mousemove', e => {{
      showTip(`<strong>${{trait}}</strong>Δ = ${{delta >= 0 ? '+' : ''}}${{delta.toFixed(4)}}`, e.clientX, e.clientY);
    }});
    bar.addEventListener('mouseleave', hideTip);
    svg.appendChild(bar);

    // trait label
    svg.appendChild(svgText(trait + (Math.abs(delta) >= 0.05 ? ' ★' : ''), {{
      x: ML - 5, y: y + ROW/2 + 4,
      'text-anchor': 'end', 'font-size': 11,
      fill: Math.abs(delta) >= 0.05 ? cssVar('--text') : cssVar('--text-2')
    }}));

    // value label
    const sign = delta >= 0 ? '+' : '';
    svg.appendChild(svgText(`${{sign}}${{delta.toFixed(3)}}`, {{
      x: delta >= 0 ? cx + bw + 4 : cx - bw - 4, y: y + ROW/2 + 4,
      'text-anchor': delta >= 0 ? 'start' : 'end',
      'font-size': 10, fill: cssVar('--text-3')
    }}));
  }}

  // x-axis label
  svg.appendChild(svgText('← more control                more steered →', {{
    x: cx, y: H - 1, 'text-anchor': 'middle', 'font-size': 10,
    fill: cssVar('--text-3')
  }}));
  svg.setAttribute('height', H + 4);
}}

// ── Wire up ───────────────────────────────────────────────────────────────────
const agents = DATA.agents;

// TS agent selector
const tsSel = document.getElementById('ts-agent');
agents.forEach(a => {{ const o = document.createElement('option'); o.value = a; o.textContent = a; tsSel.appendChild(o); }});
tsSel.addEventListener('change', () => buildTs(tsSel.value));

// Delta agent selector
if (DATA.deltas) {{
  document.getElementById('p-delta').style.display = '';
  const dSel = document.getElementById('delta-agent');
  agents.forEach(a => {{ const o = document.createElement('option'); o.value = a; o.textContent = a; dSel.appendChild(o); }});
  dSel.addEventListener('change', () => buildDelta(dSel.value));
}}

function render() {{
  buildTs(tsSel.value || agents[0]);
  buildHeatmap();
  if (DATA.deltas) buildDelta(document.getElementById('delta-agent').value || agents[0]);
}}

function toggleTheme() {{
  const r = document.documentElement;
  const cur = r.getAttribute('data-theme');
  const isDark = cur === 'dark' || (!cur && window.matchMedia('(prefers-color-scheme:dark)').matches);
  r.setAttribute('data-theme', isDark ? 'light' : 'dark');
  render();
}}

render();
</script>
"""


def build_html(
    steered_agg: dict,
    control_agg: Optional[dict],
    deltas: Optional[dict],
    steered_name: str,
    control_name: Optional[str],
) -> str:
    payload = {
        "agents": steered_agg["agents"],
        "means": steered_agg["means"],
        "ts": steered_agg["ts"],
        "ts_traits": steered_agg["ts_traits"],
        "ts_max_chunks": steered_agg["ts_max_chunks"],
        "hm_traits": steered_agg["hm_traits"],
        "deltas": deltas,
    }
    title = steered_name
    if control_name:
        title += f" vs {control_name}"
    control_badge = (
        f'<span class="run-badge">control: {control_name}</span>'
        if control_name
        else ""
    )
    return _HTML.format(
        title=title,
        steered_name=steered_name,
        control_badge=control_badge,
        data_json=json.dumps(payload, separators=(",", ":")),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate an interactive persona probe dashboard HTML file."
    )
    ap.add_argument("--steered", required=True, help="Steered run log directory.")
    ap.add_argument("--control", default=None, help="Control run log directory (optional).")
    ap.add_argument("--out", required=True, help="Output HTML path.")
    ap.add_argument("--top-k-ts", type=int, default=6, help="Traits shown in time series (default 6).")
    ap.add_argument("--top-k-hm", type=int, default=22, help="Traits shown in heatmap (default 22).")
    args = ap.parse_args(argv)

    print(f"Loading steered run: {args.steered}")
    steered_records = load_records(args.steered)
    if not steered_records:
        print("ERROR: no probe data found in steered run.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(steered_records)} probe records across "
          f"{len(set(r['agent_id'] for r in steered_records))} agents")

    steered_agg = aggregate(steered_records, args.top_k_ts, args.top_k_hm)

    control_agg = None
    control_name = None
    if args.control:
        print(f"Loading control run: {args.control}")
        control_records = load_records(args.control)
        if control_records:
            print(f"  {len(control_records)} probe records")
            control_agg = aggregate(control_records, args.top_k_ts, args.top_k_hm)
            control_name = os.path.basename(args.control.rstrip("/\\"))
        else:
            print("  WARNING: no probe data in control run, skipping delta panel.")

    deltas = compute_deltas(steered_agg, control_agg)

    html = build_html(
        steered_agg, control_agg, deltas,
        steered_name=os.path.basename(args.steered.rstrip("/\\")),
        control_name=control_name,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written: {args.out}")


if __name__ == "__main__":
    main()
