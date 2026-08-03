#!/usr/bin/env python3
"""Static research report: all steering experiments compared to noop baseline.

All figures are pre-rendered in a single scrollable HTML page — no interactions,
no dropdowns, just plots.

Debate section
  1. On-target bar chart  — did the target trait increase in player_1?
  2. Player_1 (steered) delta heatmap vs noop
  3. Player_0 (unsteered) delta heatmap vs noop — contagion check
  4. Target-trait chunk time-series — how does alignment evolve mid-completion?

Mafia section
  5. On-target bar chart  — wolf-agent target trait vs noop wolf
  6. Wolf delta heatmap vs noop wolf
  7. Villager (non-wolf) delta heatmap vs noop — contagion check

Usage
-----
python scripts/textarena/probe_report.py --logs-dir logs --out report.html
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────

DEBATE_NOOP_ID = "debate_noop_probe_2p"
MAFIA_NOOP_ID  = "mafia_noop_probe_8p"

DEBATE_STEER_PATTERN = "debate_activation_p1_*_2p"
MAFIA_STEER_PATTERN  = "mafia_activation_wolf_*_8p"

DEBATE_STEERED_AGENT = "player_1"

TRAIT_RE = re.compile(r"debate_activation_p1_(.+)_2p|mafia_activation_wolf_(.+)_8p")


def parse_target_trait(run_id: str) -> Optional[str]:
    m = TRAIT_RE.match(run_id)
    if m:
        return m.group(1) or m.group(2)
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _extract_probe(rec: dict) -> Tuple[Dict, List]:
    probe = rec.get("persona_probe") or {}
    if "mean" in probe and isinstance(probe["mean"], dict):
        return probe["mean"], probe.get("chunks", [])
    flat = {k: v for k, v in probe.items() if isinstance(v, (int, float))}
    return flat, []


def load_records(run_dir: str) -> List[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl"))):
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


def load_wolf_map(run_dir: str) -> Dict[int, str]:
    """Return {episode: wolf_agent_id} by reading summary files."""
    wolf_map: Dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
        m = re.search(r"episode_(\d+)\.summary\.json$", path)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            s = json.load(open(path, encoding="utf-8"))
            close_info = s.get("final_info", {}).get("close_info", {})
            for idx, info in close_info.items():
                if info.get("role") == "Mafia":
                    wolf_map[ep] = f"player_{idx}"
                    break
        except (OSError, json.JSONDecodeError):
            continue
    return wolf_map


# ── Aggregation ───────────────────────────────────────────────────────────────

def mean_by_agent(records: List[dict]) -> Dict[str, Dict[str, float]]:
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    ns:   Dict[str, int] = defaultdict(int)
    for rec in records:
        aid = rec["agent_id"]
        mean_dict, _ = _extract_probe(rec)
        for t, v in mean_dict.items():
            sums[aid][t] += v
        ns[aid] += 1
    return {
        a: {t: round(s / ns[a], 5) for t, s in sums[a].items()}
        for a in ns
    }


def mean_filtered(records: List[dict], keep_agents: set) -> Dict[str, float]:
    """Mean over records whose agent_id is in keep_agents (single pooled mean)."""
    trait_sum: Dict[str, float] = defaultdict(float)
    n = 0
    for rec in records:
        if rec["agent_id"] not in keep_agents:
            continue
        mean_dict, _ = _extract_probe(rec)
        for t, v in mean_dict.items():
            trait_sum[t] += v
        n += 1
    if n == 0:
        return {}
    return {t: round(s / n, 5) for t, s in trait_sum.items()}


def wolf_filtered_mean(records: List[dict], wolf_map: Dict[int, str]
                       ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (wolf_mean, villager_mean) using per-episode wolf identity."""
    w_sum: Dict[str, float] = defaultdict(float)
    v_sum: Dict[str, float] = defaultdict(float)
    wn = 0; vn = 0
    for rec in records:
        ep  = rec.get("episode", -1)
        aid = rec["agent_id"]
        wolf = wolf_map.get(ep)
        mean_dict, _ = _extract_probe(rec)
        if aid == wolf:
            for t, v in mean_dict.items():
                w_sum[t] += v
            wn += 1
        else:
            for t, v in mean_dict.items():
                v_sum[t] += v
            vn += 1
    wolf_m    = {t: round(s / wn, 5) for t, s in w_sum.items()} if wn else {}
    villager_m = {t: round(s / vn, 5) for t, s in v_sum.items()} if vn else {}
    return wolf_m, villager_m


def chunk_ts_by_agent(records: List[dict], agent_id: str,
                      trait: str, max_chunks: int = 30) -> List[Optional[float]]:
    """Average chunk trajectory for one agent / one trait."""
    buckets: List[List[float]] = [[] for _ in range(max_chunks)]
    for rec in records:
        if rec["agent_id"] != agent_id:
            continue
        _, chunks = _extract_probe(rec)
        for chunk in chunks:
            token = chunk.get("token", 0)
            cidx = max(0, (token // 10) - 1)
            if cidx < max_chunks:
                score = chunk.get("scores", {}).get(trait)
                if score is not None:
                    buckets[cidx].append(score)
    return [
        round(sum(b) / len(b), 5) if b else None
        for b in buckets
    ]


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_data(logs_dir: str) -> dict:
    debate_dir = os.path.join(logs_dir, "debate")
    mafia_dir  = os.path.join(logs_dir, "mafia")

    # ── Debate ────────────────────────────────────────────────────────────────
    debate_noop_path = os.path.join(debate_dir, DEBATE_NOOP_ID)
    debate_noop_recs = load_records(debate_noop_path)
    debate_noop_by_agent = mean_by_agent(debate_noop_recs)

    debate_exps = []
    for run_dir in sorted(glob.glob(os.path.join(debate_dir, "debate_activation_p1_*"))):
        run_id = os.path.basename(run_dir)
        trait  = parse_target_trait(run_id)
        if not trait:
            continue
        recs = load_records(run_dir)
        if not recs:
            continue
        by_agent = mean_by_agent(recs)
        has_chunks = any(
            len(_extract_probe(r)[1]) > 0 for r in recs[:20]
        )
        ts = (
            chunk_ts_by_agent(recs, DEBATE_STEERED_AGENT, trait)
            if has_chunks else []
        )
        debate_exps.append({
            "run_id": run_id,
            "trait":  trait,
            "by_agent": by_agent,
            "ts_steered": ts,
            "n": len(recs),
        })

    # sort experiments consistently
    debate_exps.sort(key=lambda x: x["trait"])

    # ── Mafia ─────────────────────────────────────────────────────────────────
    mafia_noop_path = os.path.join(mafia_dir, MAFIA_NOOP_ID)
    mafia_noop_recs  = load_records(mafia_noop_path)
    mafia_noop_wolf_map = load_wolf_map(mafia_noop_path)
    mafia_noop_wolf_m, mafia_noop_vil_m = wolf_filtered_mean(
        mafia_noop_recs, mafia_noop_wolf_map
    )

    mafia_exps = []
    for run_dir in sorted(glob.glob(os.path.join(mafia_dir, "mafia_activation_wolf_*"))):
        run_id = os.path.basename(run_dir)
        trait  = parse_target_trait(run_id)
        if not trait:
            continue
        recs = load_records(run_dir)
        if not recs:
            continue
        wolf_map = load_wolf_map(run_dir)
        wolf_m, vil_m = wolf_filtered_mean(recs, wolf_map)
        mafia_exps.append({
            "run_id":   run_id,
            "trait":    trait,
            "wolf_mean":    wolf_m,
            "vil_mean":     vil_m,
            "n": len(recs),
        })

    mafia_exps.sort(key=lambda x: x["trait"])

    # ── Choose top heatmap traits ──────────────────────────────────────────────
    debate_p1_noop = debate_noop_by_agent.get(DEBATE_STEERED_AGENT, {})
    debate_p0_noop = debate_noop_by_agent.get("player_0", {})

    def top_delta_traits(pairs, noop_m, n=22):
        """pairs = list of {trait: float} dicts (steered means). Returns top-n traits by max |delta|."""
        trait_max: Dict[str, float] = defaultdict(float)
        for s_m in pairs:
            for t in set(s_m) | set(noop_m):
                d = abs(s_m.get(t, 0.0) - noop_m.get(t, 0.0))
                trait_max[t] = max(trait_max[t], d)
        return sorted(trait_max, key=lambda t: -trait_max[t])[:n]

    debate_hm_p1 = top_delta_traits(
        [e["by_agent"].get(DEBATE_STEERED_AGENT, {}) for e in debate_exps],
        debate_p1_noop,
    )
    debate_hm_p0 = top_delta_traits(
        [e["by_agent"].get("player_0", {}) for e in debate_exps],
        debate_p0_noop,
    )
    debate_hm_traits = sorted(
        set(debate_hm_p1) | set(debate_hm_p0),
        key=lambda t: -max(
            abs(e["by_agent"].get(DEBATE_STEERED_AGENT, {}).get(t, 0) - debate_p1_noop.get(t, 0))
            for e in debate_exps
        )
    )[:22]

    mafia_hm_traits = sorted(
        set(top_delta_traits([e["wolf_mean"] for e in mafia_exps], mafia_noop_wolf_m)) |
        set(top_delta_traits([e["vil_mean"]  for e in mafia_exps], mafia_noop_vil_m)),
        key=lambda t: -max(
            abs(e["wolf_mean"].get(t, 0) - mafia_noop_wolf_m.get(t, 0))
            for e in mafia_exps
        )
    )[:22]

    return {
        "debate": {
            "exps": debate_exps,
            "noop_p1": debate_p1_noop,
            "noop_p0": debate_p0_noop,
            "hm_traits": debate_hm_traits,
        },
        "mafia": {
            "exps": mafia_exps,
            "noop_wolf": mafia_noop_wolf_m,
            "noop_vil":  mafia_noop_vil_m,
            "hm_traits": mafia_hm_traits,
        },
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box}
body{margin:0;padding:0 0 64px;background:var(--bg);color:var(--text);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums}
:root{
  --bg:#f5f4f0;--surface:#ffffff;--border:#e0dfd6;
  --text:#0e0e0d;--text2:#5a5850;--text3:#95938b;
  --grid:#e8e6de;--axis:#c8c6be;
  --pos:#2a78d6;--neg:#e34948;--mid:#f0efec;
  --noop:#9a9890;
  --c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;
  --c4:#eda100;--c5:#e87ba4;--c6:#008300;
  --c7:#4a3aa7;--c8:#e34948;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#0f0f0e;--surface:#181815;--border:#2b2b28;
  --text:#eeede6;--text2:#b0ae a6;--text3:#78766e;
  --grid:#222220;--axis:#3a3835;
  --mid:#383835;--noop:#706e66;
  --c1:#3987e5;--c2:#d95926;--c3:#199e70;
  --c4:#c98500;--c5:#d55181;--c6:#008300;
  --c7:#9085e9;--c8:#e66767;
}}
:root[data-theme=dark]{
  --bg:#0f0f0e;--surface:#181815;--border:#2b2b28;
  --text:#eeede6;--text2:#b0aea6;--text3:#78766e;
  --grid:#222220;--axis:#3a3835;
  --mid:#383835;--noop:#706e66;
  --c1:#3987e5;--c2:#d95926;--c3:#199e70;
  --c4:#c98500;--c5:#d55181;--c6:#008300;
  --c7:#9085e9;--c8:#e66767;
}
.wrap{max-width:960px;margin:0 auto;padding:0 28px}
header{padding:32px 28px 24px;max-width:960px;margin:0 auto;border-bottom:1px solid var(--border)}
h1{font-size:20px;font-weight:700;margin:0 0 4px}
.subtitle{font-size:13px;color:var(--text2);margin:0}
.section-hd{
  font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
  color:var(--text3);margin:40px 0 4px;padding-bottom:8px;
  border-bottom:1px solid var(--border)
}
.fig{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:20px 24px 16px;margin-bottom:20px}
.fig-title{font-size:13px;font-weight:600;margin:0 0 2px}
.fig-cap{font-size:11px;color:var(--text3);margin:0 0 14px}
.scroll{overflow-x:auto}
svg{display:block;overflow:visible}
#tip{position:fixed;pointer-events:none;display:none;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;padding:7px 10px;font-size:12px;
  line-height:1.6;box-shadow:0 2px 8px rgba(0,0,0,.15);z-index:99;max-width:220px}
#tip b{display:block;font-size:11px;color:var(--text2);margin-bottom:2px}
.theme-btn{position:fixed;top:12px;right:16px;background:var(--surface);
  border:1px solid var(--border);border-radius:6px;padding:5px 10px;
  font:12px system-ui;cursor:pointer;color:var(--text2);z-index:10}
"""

_JS_HEAD = r"""
const TIP = document.getElementById('tip');
function tip(html,x,y){
  TIP.innerHTML=html; TIP.style.display='block';
  const W=window.innerWidth,H=window.innerHeight;
  const tw=TIP.offsetWidth,th=TIP.offsetHeight;
  TIP.style.left=(x+14+tw>W?x-tw-10:x+14)+'px';
  TIP.style.top=(y+th+10>H?y-th-6:y+10)+'px';
}
function hideTip(){TIP.style.display='none'}

function sv(tag,a={}){
  const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const[k,v]of Object.entries(a))e.setAttribute(k,v);
  return e;
}
function stxt(t,a={}){const e=sv('text',a);e.textContent=t;return e;}

function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}

function hexRgb(h){
  h=h.replace('#','');
  if(h.length===3)h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];
}
function rgbHex([r,g,b]){return'#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('')}
function lerp(a,b,t){const ra=hexRgb(a),rb=hexRgb(b);return rgbHex(ra.map((v,i)=>v+(rb[i]-v)*t))}

function divColor(v,lo,hi){
  const absMax=Math.max(Math.abs(lo),Math.abs(hi),1e-4);
  const t=Math.max(-1,Math.min(1,v/absMax));
  return t>=0?lerp(css('--mid'),css('--pos'),t):lerp(css('--mid'),css('--neg'),-t);
}

const CATS=['--c1','--c2','--c3','--c4','--c5','--c6','--c7','--c8'];
function cat(i){return css(CATS[i%CATS.length])}

function toggleTheme(){
  const r=document.documentElement;
  const isDark=r.getAttribute('data-theme')==='dark'||
    (!r.getAttribute('data-theme')&&window.matchMedia('(prefers-color-scheme:dark)').matches);
  r.setAttribute('data-theme',isDark?'light':'dark');
  renderAll();
}
"""

# ── individual chart renderers (JS) ──────────────────────────────────────────

_JS_HBAR = r"""
/* grouped horizontal bar chart (on-target effect) */
function drawOnTarget(svgId, exps, noopByTrait, label){
  const svg=document.getElementById(svgId);
  svg.innerHTML='';
  const n=exps.length;
  const ML=170,MR=80,MT=16,MB=24;
  const ROW=36, BH=12, GAP=4;
  const H=MT+n*ROW+MB;
  const W=680;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('width',W);svg.setAttribute('height',H);
  const PW=W-ML-MR;

  // x range
  let lo=0,hi=0;
  for(const exp of exps){
    const sv2=(exp.steered||0),nv=(noopByTrait[exp.trait]||0);
    lo=Math.min(lo,sv2,nv); hi=Math.max(hi,sv2,nv);
  }
  const pad=(hi-lo)*0.12||0.05; lo-=pad; hi+=pad;

  function xp(v){return ML+(v-lo)/(hi-lo)*PW}
  const x0=xp(0);

  // grid + x-axis
  const g=css('--grid'),t3=css('--text3'),t2=css('--text2');
  const nTicks=5;
  for(let i=0;i<=nTicks;i++){
    const v=lo+(hi-lo)*i/nTicks;
    const x=xp(v);
    svg.appendChild(sv('line',{x1:x,y1:MT,x2:x,y2:MT+n*ROW,stroke:g,'stroke-width':1}));
    svg.appendChild(stxt(v.toFixed(2),{x,y:MT+n*ROW+14,'text-anchor':'middle','font-size':10,fill:t3}));
  }
  // zero line
  svg.appendChild(sv('line',{x1:x0,y1:MT,x2:x0,y2:MT+n*ROW,stroke:css('--axis'),'stroke-width':1}));

  for(let i=0;i<exps.length;i++){
    const exp=exps[i];
    const y=MT+i*ROW;
    const sv_val=exp.steered||0;
    const nv=noopByTrait[exp.trait]||0;
    const col=cat(i);

    // experiment label
    svg.appendChild(stxt(exp.trait,{x:ML-6,y:y+ROW/2+4,'text-anchor':'end','font-size':12,fill:css('--text')}));

    // noop bar (gray, behind)
    const noopBar=sv('rect',{
      x:Math.min(x0,xp(nv))+1,y:y+4,
      width:Math.abs(xp(nv)-x0)-1,height:ROW-8,
      fill:css('--noop'),rx:2,opacity:.6
    });
    svg.appendChild(noopBar);

    // steered bar
    const steerBar=sv('rect',{
      x:Math.min(x0,xp(sv_val))+1,y:y+4+(BH+GAP)/2,
      width:Math.abs(xp(sv_val)-x0)-1,height:BH,
      fill:col,rx:2,opacity:.9
    });
    steerBar.addEventListener('mousemove',e=>{
      tip(`<b>${exp.trait}</b>steered: ${sv_val.toFixed(4)}<br>noop: ${nv.toFixed(4)}<br>Δ: ${(sv_val-nv).toFixed(4)}`,e.clientX,e.clientY);
    });
    steerBar.addEventListener('mouseleave',hideTip);
    svg.appendChild(steerBar);
  }

  // legend
  const lx=ML+PW+8, ly=MT+4;
  svg.appendChild(sv('rect',{x:lx,y:ly,width:14,height:8,fill:css('--noop'),rx:2,opacity:.6}));
  svg.appendChild(stxt('noop',{x:lx+17,y:ly+8,'font-size':10,fill:t2}));
  svg.appendChild(sv('rect',{x:lx,y:ly+16,width:14,height:8,fill:css('--c1'),rx:2}));
  svg.appendChild(stxt(label,{x:lx+17,y:ly+24,'font-size':10,fill:t2}));
}
"""

_JS_HEATMAP = r"""
/* delta heatmap: rows=experiments, cols=traits */
function drawHeatmap(svgId, exps, noop, traits, getVal){
  const svg=document.getElementById(svgId);
  svg.innerHTML='';

  const rows=exps.length, cols=traits.length;
  const CW=36, RH=28, ML=160, MT=80, MB=12;
  const W=ML+cols*CW+8;
  const H=MT+rows*RH+MB;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('width',W);svg.setAttribute('height',H);

  // global delta range
  let lo=0,hi=0;
  for(const exp of exps)for(const t of traits){
    const d=getVal(exp,t)-(noop[t]||0);
    lo=Math.min(lo,d);hi=Math.max(hi,d);
  }

  const t2=css('--text2'),t3=css('--text3');

  // col headers (rotated)
  for(let ci=0;ci<cols;ci++){
    const x=ML+ci*CW+CW/2;
    const e=stxt(traits[ci],{x:0,y:0,'font-size':10,fill:t2,'text-anchor':'start',
      transform:`translate(${x},${MT-4}) rotate(-45)`});
    svg.appendChild(e);
  }

  // rows
  for(let ri=0;ri<rows;ri++){
    const exp=exps[ri];
    const y=MT+ri*RH;
    svg.appendChild(stxt(exp.trait,{x:ML-6,y:y+RH/2+4,'text-anchor':'end','font-size':11,
      fill:exp.trait===exp.trait?css('--text'):t2}));
    for(let ci=0;ci<cols;ci++){
      const trait=traits[ci];
      const delta=getVal(exp,trait)-(noop[trait]||0);
      const col=divColor(delta,lo,hi);
      const x=ML+ci*CW;
      const cell=sv('rect',{x:x+1,y:y+1,width:CW-2,height:RH-2,fill:col,rx:2});
      const dStr=(delta>=0?'+':'')+delta.toFixed(3);
      const isTarget=trait===exp.trait;
      // outline target cell
      if(isTarget)svg.appendChild(sv('rect',{x:x+1,y:y+1,width:CW-2,height:RH-2,
        fill:'none',stroke:css('--text'),rx:2,'stroke-width':1.5}));
      cell.addEventListener('mousemove',e=>{
        tip(`<b>${exp.trait} → ${trait}</b>Δ ${dStr}${isTarget?' ★ target':''}`,e.clientX,e.clientY);
      });
      cell.addEventListener('mouseleave',hideTip);
      svg.appendChild(cell);
    }
  }

  // diverging legend
  const LX=ML,LY=H-MB+4,LW=Math.min(220,cols*CW);
  const steps=40;
  for(let i=0;i<steps;i++){
    const v=lo+(hi-lo)*i/steps;
    svg.appendChild(sv('rect',{x:LX+i*(LW/steps),y:LY,width:LW/steps+.5,height:6,fill:divColor(v,lo,hi)}));
  }
  svg.appendChild(stxt(lo.toFixed(3),{x:LX,y:LY+16,'font-size':9,fill:t3}));
  svg.appendChild(stxt('0',{x:LX+LW/2,y:LY+16,'font-size':9,fill:t3,'text-anchor':'middle'}));
  svg.appendChild(stxt(hi.toFixed(3),{x:LX+LW,y:LY+16,'font-size':9,fill:t3,'text-anchor':'end'}));
  svg.appendChild(stxt('★ = target trait',{x:LX+LW+6,y:LY+6,'font-size':9,fill:t3}));
  svg.setAttribute('height',H+20);
}
"""

_JS_TS = r"""
/* target-trait chunk time-series */
function drawTS(svgId, exps, noopMean){
  const svg=document.getElementById(svgId);
  svg.innerHTML='';

  const W=740,H=260,ML=56,MR=170,MT=12,MB=40;
  const PW=W-ML-MR,PH=H-MT-MB;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  svg.setAttribute('width',W);svg.setAttribute('height',H);

  // max chunks across all experiments
  let maxC=exps.reduce((m,e)=>Math.max(m,(e.ts_steered||[]).length),0);
  if(maxC===0){svg.appendChild(stxt('No chunk data available for this comparison.',{x:W/2,y:H/2,'text-anchor':'middle','font-size':12,fill:css('--text3')}));return;}
  maxC=Math.min(maxC,30);

  // y domain
  let lo=Infinity,hi=-Infinity;
  for(const exp of exps)for(const v of (exp.ts_steered||[])){
    if(v!==null){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  }
  // also include noop means as horizontal refs
  for(const exp of exps){const n=noopMean[exp.trait]||0;lo=Math.min(lo,n);hi=Math.max(hi,n);}
  const pad=(hi-lo)*0.12||0.05;lo-=pad;hi+=pad;

  function xp(i){return ML+i/(maxC-1||1)*PW}
  function yp(v){return MT+(1-(v-lo)/(hi-lo))*PH}

  const g=css('--grid'),t3=css('--text3'),t2=css('--text2');

  // gridlines
  for(let i=0;i<=5;i++){
    const v=lo+(hi-lo)*i/5,y=yp(v);
    svg.appendChild(sv('line',{x1:ML,y1:y,x2:ML+PW,y2:y,stroke:g,'stroke-width':1}));
    svg.appendChild(stxt(v.toFixed(3),{x:ML-5,y:y+4,'text-anchor':'end','font-size':9,fill:t3}));
  }
  if(lo<0&&hi>0){
    const y0=yp(0);
    svg.appendChild(sv('line',{x1:ML,y1:y0,x2:ML+PW,y2:y0,stroke:t3,'stroke-width':1,'stroke-dasharray':'3,3'}));
  }
  // x ticks
  const xStep=Math.max(1,Math.floor(maxC/7));
  for(let i=0;i<maxC;i+=xStep){
    svg.appendChild(sv('line',{x1:xp(i),y1:MT+PH,x2:xp(i),y2:MT+PH+4,stroke:t3,'stroke-width':1}));
    svg.appendChild(stxt(`${(i+1)*10}`,{x:xp(i),y:MT+PH+16,'text-anchor':'middle','font-size':9,fill:t3}));
  }
  svg.appendChild(stxt('tokens (chunk end)',{x:ML+PW/2,y:H-4,'text-anchor':'middle','font-size':11,fill:t2}));

  // draw each experiment's steered trajectory + noop reference
  for(let ei=0;ei<exps.length;ei++){
    const exp=exps[ei];
    const col=cat(ei);
    const ts=exp.ts_steered||[];
    // noop horizontal reference (dashed)
    const nv=noopMean[exp.trait];
    if(nv!=null){
      const yn=yp(nv);
      svg.appendChild(sv('line',{x1:ML,y1:yn,x2:ML+PW,y2:yn,stroke:col,'stroke-width':1,'stroke-dasharray':'4,4',opacity:.5}));
    }
    // steered line
    let d='';let prev=null;
    for(let i=0;i<Math.min(ts.length,maxC);i++){
      if(ts[i]===null){prev=null;continue;}
      const x=xp(i),y=yp(ts[i]);
      d+=prev===null?`M${x},${y}`:`L${x},${y}`;
      prev=1;
    }
    if(d)svg.appendChild(sv('path',{d,fill:'none',stroke:col,'stroke-width':2,
      'stroke-linejoin':'round','stroke-linecap':'round'}));
  }

  // borders
  svg.appendChild(sv('line',{x1:ML,y1:MT,x2:ML,y2:MT+PH,stroke:css('--axis'),'stroke-width':1}));
  svg.appendChild(sv('line',{x1:ML,y1:MT+PH,x2:ML+PW,y2:MT+PH,stroke:css('--axis'),'stroke-width':1}));

  // legend
  for(let ei=0;ei<exps.length;ei++){
    const col=cat(ei);
    const lx=ML+PW+12,ly=MT+4+ei*20;
    svg.appendChild(sv('line',{x1:lx,y1:ly+5,x2:lx+18,y2:ly+5,stroke:col,'stroke-width':2,'stroke-linecap':'round'}));
    svg.appendChild(stxt(exps[ei].trait,{x:lx+22,y:ly+9,'font-size':11,fill:css('--text2')}));
  }
  svg.appendChild(stxt('— steered  - - noop ref',{x:ML+PW+12,y:MT+4+exps.length*20+12,'font-size':9,fill:t3}));
}
"""

_JS_RENDER = r"""
function renderAll(){
  const D=window.__DATA__;

  // ── Debate ────────────────────────────────────────────────────────────────
  // Fig 1: on-target
  drawOnTarget('fig1-svg',
    D.debate.exps.map(e=>({trait:e.trait,steered:e.by_agent.player_1?.[e.trait]||0})),
    D.debate.noop_p1,
    'steered (p1)'
  );

  // Fig 2: player_1 delta heatmap
  drawHeatmap('fig2-svg',D.debate.exps,D.debate.noop_p1,D.debate.hm_traits,
    (e,t)=>e.by_agent.player_1?.[t]||0);

  // Fig 3: player_0 delta heatmap
  drawHeatmap('fig3-svg',D.debate.exps,D.debate.noop_p0,D.debate.hm_traits,
    (e,t)=>e.by_agent.player_0?.[t]||0);

  // Fig 4: time series
  drawTS('fig4-svg',D.debate.exps,D.debate.noop_p1);

  // ── Mafia ─────────────────────────────────────────────────────────────────
  // Fig 5: on-target (wolf)
  drawOnTarget('fig5-svg',
    D.mafia.exps.map(e=>({trait:e.trait,steered:e.wolf_mean?.[e.trait]||0})),
    D.mafia.noop_wolf,
    'steered wolf'
  );

  // Fig 6: wolf delta heatmap
  drawHeatmap('fig6-svg',D.mafia.exps,D.mafia.noop_wolf,D.mafia.hm_traits,
    (e,t)=>e.wolf_mean?.[t]||0);

  // Fig 7: villager delta heatmap
  drawHeatmap('fig7-svg',D.mafia.exps,D.mafia.noop_vil,D.mafia.hm_traits,
    (e,t)=>e.vil_mean?.[t]||0);
}

renderAll();
"""


def build_html(data: dict) -> str:
    data_json = json.dumps(data, separators=(",", ":"))
    return f"""<!doctype html>
<title>Persona probe report</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{_CSS}</style>
<button class="theme-btn" onclick="toggleTheme()">&#9680; theme</button>
<div id="tip"></div>

<header>
  <h1>Persona probe report</h1>
  <p class="subtitle">
    Cosine projections onto 53 PersVecGen trait vectors &mdash;
    steered experiments vs noop baseline. &#9733; marks the target trait in heatmaps.
  </p>
</header>

<div class="wrap">

<p class="section-hd">Debate &mdash; player_1 (AGAINST) steered, player_0 unsteered control</p>

<div class="fig">
  <p class="fig-title">Fig 1 &mdash; On-target effect (debate player_1)</p>
  <p class="fig-cap">Mean projection on the intended trait for the steered agent (player_1) vs the same agent in the noop run. Wider bar = stronger alignment.</p>
  <div class="scroll"><svg id="fig1-svg"></svg></div>
</div>

<div class="fig">
  <p class="fig-title">Fig 2 &mdash; Steered agent (player_1) full trait delta</p>
  <p class="fig-cap">Steered mean &minus; noop mean for player_1 across all experiments. Blue = steering increased this trait, red = decreased. &#9733; outlines the target trait for each row.</p>
  <div class="scroll"><svg id="fig2-svg"></svg></div>
</div>

<div class="fig">
  <p class="fig-title">Fig 3 &mdash; Unsteered agent (player_0) trait delta &mdash; contagion check</p>
  <p class="fig-cap">Same delta calculation but for player_0, who receives no steering. Non-zero deltas indicate social contagion from the steered opponent.</p>
  <div class="scroll"><svg id="fig3-svg"></svg></div>
</div>

<div class="fig">
  <p class="fig-title">Fig 4 &mdash; Target-trait chunk time-series (player_1)</p>
  <p class="fig-cap">Mean cosine projection on each experiment's target trait, averaged across all player_1 completions at each 10-token chunk. Solid lines = steered; dashed = noop mean reference for that trait.</p>
  <div class="scroll"><svg id="fig4-svg"></svg></div>
</div>

<p class="section-hd">Mafia &mdash; wolf agent steered (role-aware, varies per episode), villagers unsteered</p>

<div class="fig">
  <p class="fig-title">Fig 5 &mdash; On-target effect (mafia wolf)</p>
  <p class="fig-cap">Mean projection on the target trait for whoever was the wolf that episode, averaged across all episodes. Compared to the noop wolf baseline.</p>
  <div class="scroll"><svg id="fig5-svg"></svg></div>
</div>

<div class="fig">
  <p class="fig-title">Fig 6 &mdash; Wolf trait delta heatmap</p>
  <p class="fig-cap">Wolf-agent mean minus noop-wolf mean per trait, across all mafia experiments.</p>
  <div class="scroll"><svg id="fig6-svg"></svg></div>
</div>

<div class="fig">
  <p class="fig-title">Fig 7 &mdash; Villager trait delta heatmap &mdash; contagion check</p>
  <p class="fig-cap">Unsteered villager mean minus noop villager mean. Significant deltas suggest the wolf's steering affected other players' language.</p>
  <div class="scroll"><svg id="fig7-svg"></svg></div>
</div>

</div>

<script>
window.__DATA__ = {data_json};
{_JS_HEAD}
{_JS_HBAR}
{_JS_HEATMAP}
{_JS_TS}
{_JS_RENDER}
</script>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate static persona probe research report.")
    ap.add_argument("--logs-dir", default="logs", help="Root logs directory (default: logs)")
    ap.add_argument("--out", required=True, help="Output HTML path")
    args = ap.parse_args(argv)

    print("Building data...")
    data = build_data(args.logs_dir)
    print(f"  debate: {len(data['debate']['exps'])} experiments")
    print(f"  mafia:  {len(data['mafia']['exps'])} experiments")

    html = build_html(data)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written: {args.out}")


if __name__ == "__main__":
    main()
