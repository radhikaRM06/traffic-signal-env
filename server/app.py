import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from server.traffic_environment import TrafficSignalEnvironment
    from models import TrafficAction
except Exception:
    try:
        sys.path.insert(0, '/app')
        from server.traffic_environment import TrafficSignalEnvironment
        from models import TrafficAction
    except Exception as e:
        print(f"Import error: {e}")

_env = TrafficSignalEnvironment()

app = FastAPI(title="Traffic Signal Control - OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ResetRequest(BaseModel):
    task_id: Optional[str] = "single_intersection_easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    phase_assignments: dict = {}
    duration: int = 5
    task_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=DASHBOARD)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "traffic-signal-env"}

@app.get("/metadata")
async def metadata():
    return {"name":"traffic-signal-control","version":"1.0.0","description":"Real-world traffic signal control RL environment","tasks":["single_intersection_easy","arterial_corridor_medium","urban_grid_hard"]}

@app.get("/schema")
async def schema():
    return {"action":{"phase_assignments":"Dict[str,int]","duration":"int(1-10)","task_id":"str"},"observation":{"intersections":"List[IntersectionObs]","total_vehicles_waiting":"int","network_avg_wait_time":"float","network_throughput":"int","done":"bool","reward":"float"}}

@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    obs = _env.reset(task_id=request.task_id, seed=request.seed)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False, "info": {"task_id": request.task_id}}

@app.post("/step")
async def step(request: StepRequest):
    action = TrafficAction(phase_assignments=request.phase_assignments, duration=request.duration, task_id=request.task_id or _env._task_id)
    obs = _env.step(action)
    info = {"step_number": obs.step_number, "network_throughput": obs.network_throughput, "network_avg_wait_time": obs.network_avg_wait_time}
    if obs.done and "final_score" in obs.metadata:
        info["final_score"] = obs.metadata["final_score"]
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done, "info": info}

@app.get("/state")
async def state():
    return _env.state.model_dump()

@app.post("/mcp")
async def mcp(body: dict = {}):
    return {"status": "ok", "tool": body.get("tool", ""), "result": "MCP endpoint active"}

@app.get("/tasks")
async def tasks():
    return {"tasks": [
        {"id":"single_intersection_easy","difficulty":"easy","max_steps":100,"description":"Control 1 intersection, moderate demand"},
        {"id":"arterial_corridor_medium","difficulty":"medium","max_steps":150,"description":"Coordinate 3 sequential intersections"},
        {"id":"urban_grid_hard","difficulty":"hard","max_steps":200,"description":"Manage 2x2 grid, rush-hour + emergencies"},
    ]}

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrafficAI — OpenEnv</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;color:#e0e6f0;min-height:100vh;display:flex;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px}

/* ── SIDEBAR ── */
.sidebar{width:210px;min-width:210px;background:#0d1120;border-right:1px solid #1e2640;display:flex;flex-direction:column;min-height:100vh}
.logo{padding:18px 16px 14px;border-bottom:1px solid #1e2640}
.logo-title{font-size:18px;font-weight:700;color:#e0e6f0}
.logo-title span{color:#00e5a0}
.logo-badge{margin-top:6px;display:inline-flex;align-items:center;gap:5px;background:#00e5a012;border:1px solid #00e5a030;border-radius:12px;padding:3px 9px;font-size:9px;color:#00e5a0;letter-spacing:1px}
.logo-badge::before{content:'';width:6px;height:6px;border-radius:50%;background:#00e5a0;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.stats-block{padding:12px 16px;border-bottom:1px solid #1e2640}
.stats-label{font-size:9px;color:#4a5580;letter-spacing:2px;margin-bottom:8px}
.stat-row{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;color:#8892b0}
.stat-val{color:#00e5a0;font-weight:600}
.stat-val.amber{color:#f59e0b}
.nav{padding:10px 0;flex:1}
.nav-item{display:flex;align-items:center;gap:9px;padding:9px 16px;font-size:12px;color:#4a5580;cursor:pointer;transition:all .15s;border-left:2px solid transparent;user-select:none}
.nav-item:hover{color:#8892b0;background:#111827}
.nav-item.active{color:#00e5a0;background:#0f1f30;border-left-color:#00e5a0;font-weight:600}
.nav-icon{width:14px;text-align:center;flex-shrink:0}

/* ── MAIN ── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.topbar{padding:12px 20px;border-bottom:1px solid #1e2640;display:flex;align-items:center;gap:10px;background:#0a0e1a;flex-shrink:0;flex-wrap:wrap}
.topbar-title{font-size:15px;font-weight:700}
.topbar-title span{color:#00e5a0}
.tag{padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600;white-space:nowrap}
.tag-green{background:#00e5a022;color:#00e5a0;border:1px solid #00e5a044}
.tag-amber{background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44}
.tag-red{background:#ef444422;color:#ef4444;border:1px solid #ef444444}
.tag-blue{background:#3b82f622;color:#3b82f6;border:1px solid #3b82f644}
.tag-purple{background:#a855f722;color:#a855f7;border:1px solid #a855f744}
.topbar-btns{margin-left:auto;display:flex;gap:8px}
.btn-p{background:#00e5a0;color:#0a0e1a;border:none;padding:7px 18px;border-radius:7px;font-size:12px;font-weight:700;cursor:pointer;transition:opacity .15s}
.btn-p:hover{opacity:.85}
.btn-s{background:transparent;color:#e0e6f0;border:1px solid #2a3550;padding:7px 16px;border-radius:7px;font-size:12px;cursor:pointer;transition:all .15s}
.btn-s:hover{border-color:#3b82f6;color:#3b82f6}

/* ── PAGES ── */
.content{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:14px}
.page{display:none;flex-direction:column;gap:14px}
.page.active{display:flex}
.page-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:2px}
.page-header h2{font-size:16px;font-weight:700}

/* ── HERO ── */
.hero{background:#0f1f30;border:1px solid #1e3050;border-radius:12px;padding:22px 26px}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:#00e5a012;border:1px solid #00e5a030;border-radius:20px;padding:4px 12px;font-size:10px;color:#00e5a0;letter-spacing:1px;margin-bottom:12px}
.hero h1{font-size:24px;font-weight:700;margin-bottom:8px}
.hero h1 span{color:#00e5a0}
.hero p{font-size:12px;color:#4a5580;line-height:1.7;max-width:520px;margin-bottom:16px}

/* ── METRIC CARDS ── */
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.mc{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;transition:border-color .2s}
.mc:hover{border-color:#2a3a5a}
.mc-label{font-size:10px;color:#4a5580;margin-bottom:6px;letter-spacing:.5px;text-transform:uppercase}
.mc-val{font-size:22px;font-weight:700;line-height:1;transition:all .3s}
.mc-sub{font-size:10px;color:#4a5580;margin-top:4px}
.mc-trend{position:absolute;top:12px;right:12px;font-size:10px;padding:2px 7px;border-radius:4px;font-weight:600}
.trend-up{background:#00e5a015;color:#00e5a0}
.trend-dn{background:#ef444415;color:#ef4444}
@keyframes pop{0%{transform:scale(.92);opacity:.5}60%{transform:scale(1.05)}100%{transform:scale(1);opacity:1}}
.mc.popping .mc-val{animation:pop .35s ease}

/* ── GRAPHS ── */
.graphs-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.graph-card{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:12px 14px}
.graph-title{font-size:10px;color:#4a5580;letter-spacing:.5px;text-transform:uppercase;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.graph-val{font-size:12px;font-weight:600}

/* ── TWO-COL LAYOUT ── */
.two-col{display:grid;grid-template-columns:1.3fr .9fr;gap:14px}
.panel{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:14px 16px}
.panel-title{font-size:10px;font-weight:600;color:#8892b0;margin-bottom:12px;display:flex;align-items:center;gap:8px;letter-spacing:1px;text-transform:uppercase}
.pulse{width:7px;height:7px;border-radius:50%;background:#00e5a0;animation:blink 2s infinite;display:inline-block;flex-shrink:0}

/* ── TASK TABS ── */
.task-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.task-btn{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:5px 12px;border-radius:6px;font-size:11px;cursor:pointer;transition:all .15s}
.task-btn.active{background:#0f2a1a;border-color:#00e5a0;color:#00e5a0}

/* ── SCENARIO BAR ── */
.scenario-bar{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
.scenario-label{font-size:10px;color:#4a5580;white-space:nowrap}
.sc-btn{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:4px 11px;border-radius:5px;font-size:11px;cursor:pointer;transition:all .15s}
.sc-btn:hover{border-color:#f59e0b;color:#f59e0b}
.sc-btn.sc-active{border-color:#ef4444;background:#1a0f0f;color:#ef4444}
.sc-btn.sc-ped{border-color:#a855f7;background:#1a0f1f;color:#a855f7}
.sc-btn.sc-rush{border-color:#f59e0b;background:#1a160f;color:#f59e0b}

/* ── INTERSECTION GRID ── */
.int-container{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;min-height:140px;align-items:center}
.int-box{background:#111827;border:1px solid #1e2640;border-radius:8px;padding:10px;text-align:center}
.int-id{font-size:10px;color:#4a5580;margin-bottom:8px;font-weight:600;letter-spacing:1px}
.sig-grid{display:grid;grid-template-columns:36px 36px 36px;grid-template-rows:36px 36px 36px;gap:2px;margin:0 auto}
.sg{border-radius:4px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:1px}
.sg-road{background:#0a0e1a}
.sg-center{background:#161e30;font-size:8px;color:#4a5580;line-height:1.3}
.sig-dot{width:11px;height:11px;border-radius:50%;transition:background .3s}
.s-green{background:#00e5a0;box-shadow:0 0 6px #00e5a080}
.s-red{background:#ef4444;box-shadow:0 0 6px #ef444480}
.s-yellow{background:#f59e0b;box-shadow:0 0 6px #f59e0b80}
.qnum{font-size:9px;color:#8892b0;font-weight:600}
.qb{height:3px;background:#1e2640;border-radius:2px;margin-top:2px;width:28px}
.qbf{height:100%;background:#00e5a0;border-radius:2px;transition:width .4s}
.ew-row{display:flex;justify-content:space-between;font-size:9px;color:#4a5580;margin-top:4px}

/* ── CONTROLS ── */
.ctrl{display:flex;gap:8px;margin-top:10px;align-items:center;flex-wrap:wrap}
.cbtn{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:5px 14px;border-radius:6px;font-size:11px;cursor:pointer;transition:all .15s;white-space:nowrap}
.cbtn:hover{border-color:#4a5580;color:#e0e6f0}
.cbtn.running{background:#0f2a1a;border-color:#00e5a0;color:#00e5a0}
.ep-info{font-size:10px;color:#4a5580;margin-left:auto}
.pbar{height:4px;background:#1e2640;border-radius:2px;overflow:hidden;margin-top:4px}
.pbf{height:100%;background:#00e5a0;border-radius:2px;transition:width .4s}
.sbf{height:100%;background:#f59e0b;border-radius:2px;transition:width .5s}
.replay-row{display:flex;gap:8px;margin-top:10px}
.rbtn-blue{background:#0f1a2a;border:1px solid #3b82f644;color:#3b82f6;padding:5px 14px;border-radius:6px;font-size:11px;cursor:pointer}
.rbtn-gray{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:5px 14px;border-radius:6px;font-size:11px;cursor:pointer}

/* ── WHY PANEL ── */
.why-card{background:#0d1120;border:1px solid #00e5a040;border-radius:10px;padding:14px 16px}
.why-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.why-badge{font-size:9px;background:#00e5a012;color:#00e5a0;border:1px solid #00e5a030;border-radius:4px;padding:2px 8px;letter-spacing:.5px}
.decision-box{background:#080c14;border-radius:6px;padding:10px 12px;border-left:2px solid #00e5a0;margin-bottom:10px}
.decision-label{font-size:9px;color:#4a5580;letter-spacing:.5px;margin-bottom:4px;text-transform:uppercase}
.decision-val{font-size:13px;color:#e0e6f0;font-weight:600}
.reason-list{list-style:none;display:flex;flex-direction:column;gap:1px}
.reason-list li{display:flex;align-items:flex-start;gap:8px;font-size:11px;color:#8892b0;padding:5px 0;border-bottom:1px solid #111827}
.reason-list li:last-child{border:none}
.rdot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:3px}
.rdot-g{background:#00e5a0}
.rdot-r{background:#ef4444}
.rdot-b{background:#3b82f6}
.rdot-a{background:#f59e0b}

/* ── REWARD ENGINE ── */
.rr{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.rr-top{display:flex;justify-content:space-between;font-size:11px}
.rbar{height:4px;background:#1e2640;border-radius:2px;overflow:hidden}
.rbf-g{height:100%;background:#00e5a0;border-radius:2px;transition:width .3s}
.rbf-r{height:100%;background:#ef4444;border-radius:2px;transition:width .3s}
.net-rew{border-top:1px solid #1e2640;padding-top:10px;display:flex;justify-content:space-between;font-size:14px;font-weight:700;margin-top:4px}

/* ── AGENT LOG ── */
.log-box{background:#080c14;border:1px solid #1e2640;border-radius:6px;padding:10px;height:110px;overflow-y:auto;font-family:'Courier New',monospace;font-size:10px}
.ll{padding:1px 0;color:#4a5580}
.ll-s{color:#00e5a0}
.ll-st{color:#3b82f6}
.ll-e{color:#f59e0b;font-weight:600}
.ll-err{color:#ef4444}
.ll-sc{color:#a855f7}

/* ── PERF SUMMARY ── */
.perf-card{background:linear-gradient(135deg,#0f2a1a,#0a1a30);border:1px solid #00e5a040;border-radius:10px;padding:14px 16px}
.perf-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.perf-item{background:#0a0e1a;border-radius:6px;padding:10px 12px;border:1px solid #1e2640}
.pi-label{font-size:9px;color:#4a5580;letter-spacing:.5px;text-transform:uppercase;margin-bottom:4px}
.pi-value{font-size:18px;font-weight:700}

/* ── RL TASKS PAGE ── */
.tasks-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.tc{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:18px;transition:border-color .2s;cursor:default}
.tc:hover{border-color:#2a3a5a}
.tc-badge{display:inline-block;font-size:9px;border-radius:4px;padding:2px 8px;margin-bottom:10px;font-weight:700;letter-spacing:.5px}
.badge-easy{background:#00e5a015;color:#00e5a0;border:1px solid #00e5a040}
.badge-med{background:#f59e0b15;color:#f59e0b;border:1px solid #f59e0b40}
.badge-hard{background:#ef444415;color:#ef4444;border:1px solid #ef444440}
.tc h3{font-size:14px;font-weight:700;margin-bottom:6px}
.tc p{font-size:11px;color:#4a5580;line-height:1.6;margin-bottom:12px}
.tc-stats{display:flex;gap:14px;font-size:10px;color:#4a5580;margin-bottom:12px}
.tc-stats b{color:#8892b0}
.tc-actions{display:flex;gap:8px}

/* ── API PAGE ── */
.api-table{width:100%;border-collapse:collapse}
.api-table th{font-size:9px;letter-spacing:1.5px;color:#4a5580;text-align:left;padding:8px 14px;border-bottom:1px solid #1e2640;text-transform:uppercase}
.api-table td{padding:10px 14px;font-size:12px;border-bottom:1px solid #111827;vertical-align:middle}
.api-table tr:last-child td{border:none}
.api-table tr:hover td{background:#0f1525}
.m-get{background:#3b82f622;color:#3b82f6;border:1px solid #3b82f644;font-size:9px;padding:2px 8px;border-radius:3px;font-weight:700;display:inline-block;min-width:40px;text-align:center;letter-spacing:.5px}
.m-post{background:#00e5a022;color:#00e5a0;border:1px solid #00e5a044;font-size:9px;padding:2px 8px;border-radius:3px;font-weight:700;display:inline-block;min-width:40px;text-align:center;letter-spacing:.5px}
.ep-path{font-family:'Courier New',monospace;color:#e0e6f0;font-size:13px}
.ep-desc{color:#4a5580;font-size:11px}
.ptag{font-size:9px;padding:3px 9px;border-radius:10px;border:1px solid #1e2640;color:#4a5580;display:inline-block;margin:2px;background:#111827}

/* ── REWARD ENGINE PAGE ── */
.re-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.re-component{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:18px}
.re-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #111827;font-size:12px}
.re-row:last-child{border:none}
.re-label{color:#8892b0}
.re-val-pos{color:#00e5a0;font-weight:700}
.re-val-neg{color:#ef4444;font-weight:700}

/* ── SIGNAL SIM PAGE ── */
.sim-grid{display:grid;grid-template-columns:1fr 300px;gap:14px}
.sim-canvas{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:18px;display:flex;flex-direction:column;align-items:center;min-height:360px}
.big-int{display:grid;grid-template-columns:repeat(3,90px);grid-template-rows:repeat(3,90px);gap:6px;margin:16px 0}
.bi{background:#111827;border:1px solid #1e2640;border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;font-size:10px;color:#4a5580}
.bi.road{background:#0d1525}
.bi.center{background:#0a0e1a;border-color:#2a3550;font-size:12px;font-weight:700;color:#e0e6f0}
.big-dot{width:20px;height:20px;border-radius:50%;transition:all .3s}
.big-dot.g{background:#00e5a0;box-shadow:0 0 12px #00e5a080}
.big-dot.r{background:#ef4444;box-shadow:0 0 12px #ef444480}
.big-dot.y{background:#f59e0b;box-shadow:0 0 12px #f59e0b80}
</style>
</head>
<body>

<!-- ═══ SIDEBAR ═══ -->
<div class="sidebar">
  <div class="logo">
    <div class="logo-title">Traffic<span>AI</span></div>
    <div class="logo-badge">INTELLIGENCE ON OPENENV</div>
  </div>
  <div class="stats-block">
    <div class="stats-label">QUICK STATS</div>
    <div class="stat-row"><span>Intersections</span><span class="stat-val" id="sb-ints">1</span></div>
    <div class="stat-row"><span>Tasks</span><span class="stat-val">3</span></div>
    <div class="stat-row"><span>Mode</span><span class="stat-val amber">Heuristic</span></div>
    <div class="stat-row"><span>Status</span><span class="stat-val" id="sb-status">Ready</span></div>
  </div>
  <nav class="nav">
    <div class="nav-item active" data-page="home"><span class="nav-icon">⌂</span> Home &amp; Overview</div>
    <div class="nav-item" data-page="signal"><span class="nav-icon">◯</span> Signal Sim</div>
    <div class="nav-item" data-page="rl-tasks"><span class="nav-icon">▣</span> RL Tasks</div>
    <div class="nav-item" data-page="api"><span class="nav-icon">⚡</span> API Explorer</div>
    <div class="nav-item" data-page="reward"><span class="nav-icon">◈</span> Reward Engine</div>
  </nav>
</div>

<!-- ═══ MAIN ═══ -->
<div class="main">
  <div class="topbar">
    <div class="topbar-title">Traffic<span>AI</span> Platform</div>
    <span class="tag tag-green" id="status-tag">● Running</span>
    <span class="tag tag-blue">OpenEnv v1.0</span>
    <span class="tag tag-amber" id="step-tag">step 0</span>
    <div class="topbar-btns">
      <button class="btn-p" id="main-start-btn" onclick="doReset()">▶ Start Simulation</button>
      <button class="btn-s" onclick="navigate('rl-tasks')">◯ View RL Tasks</button>
    </div>
  </div>

  <div class="content">

    <!-- ══════════════════════════════════════ HOME PAGE -->
    <div class="page active" id="page-home">

      <!-- HERO -->
      <div class="hero">
        <div class="hero-badge">★ INTELLIGENCE LAYERED ON OPENENV</div>
        <h1>Traffic<span>AI</span> Platform</h1>
        <p>Next-generation reinforcement learning environment for urban traffic signal control. AI agents minimize vehicle wait times, handle emergencies, and coordinate green waves across city intersections.</p>
        <div class="btn-row" style="display:flex;gap:10px">
          <button class="btn-p" onclick="doReset()">▶ Start Simulation</button>
          <button class="btn-s" onclick="navigate('rl-tasks')">◯ View RL Tasks</button>
        </div>
      </div>

      <!-- METRIC CARDS -->
      <div class="metrics">
        <div class="mc" id="mc-wait">
          <div class="mc-label">Vehicles Waiting</div>
          <div class="mc-val" id="m-w" style="color:#00e5a0">0</div>
          <div class="mc-sub">across network</div>
          <div class="mc-trend trend-up" id="m-w-trend">—</div>
        </div>
        <div class="mc" id="mc-awt">
          <div class="mc-label">Avg Wait Time</div>
          <div class="mc-val" id="m-awt" style="color:#3b82f6">0.0s</div>
          <div class="mc-sub">seconds</div>
        </div>
        <div class="mc" id="mc-tp">
          <div class="mc-label">Throughput</div>
          <div class="mc-val" id="m-tp" style="color:#f59e0b">0</div>
          <div class="mc-sub">vehicles / tick</div>
        </div>
        <div class="mc" id="mc-rew">
          <div class="mc-label">Reward</div>
          <div class="mc-val" id="m-rew" style="color:#00e5a0">0.000</div>
          <div class="mc-sub">this step</div>
        </div>
      </div>

      <!-- LIVE GRAPHS -->
      <div class="graphs-grid">
        <div class="graph-card">
          <div class="graph-title">
            <span>Avg Wait Time</span>
            <span class="graph-val" id="gv-awt" style="color:#3b82f6">0.0s</span>
          </div>
          <div style="position:relative;height:100px">
            <canvas id="chartAWT" role="img" aria-label="Average wait time over steps">Wait time history chart.</canvas>
          </div>
        </div>
        <div class="graph-card">
          <div class="graph-title">
            <span>Reward per Step</span>
            <span class="graph-val" id="gv-rew" style="color:#00e5a0">0.000</span>
          </div>
          <div style="position:relative;height:100px">
            <canvas id="chartREW" role="img" aria-label="Reward per step">Reward history chart.</canvas>
          </div>
        </div>
        <div class="graph-card">
          <div class="graph-title">
            <span>Throughput</span>
            <span class="graph-val" id="gv-tp" style="color:#f59e0b">0</span>
          </div>
          <div style="position:relative;height:100px">
            <canvas id="chartTP" role="img" aria-label="Throughput per step">Throughput history chart.</canvas>
          </div>
        </div>
      </div>

      <!-- MAIN TWO-COL -->
      <div class="two-col">

        <!-- LEFT: intersection + controls -->
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="panel">
            <div class="panel-title"><span class="pulse"></span>Live Intersection View</div>

            <!-- Task tabs -->
            <div class="task-row">
              <button class="task-btn active" onclick="pickTask('single_intersection_easy',this)">Single <span class="tag tag-green" style="font-size:9px;padding:1px 6px">Easy</span></button>
              <button class="task-btn" onclick="pickTask('arterial_corridor_medium',this)">Arterial <span class="tag tag-amber" style="font-size:9px;padding:1px 6px">Med</span></button>
              <button class="task-btn" onclick="pickTask('urban_grid_hard',this)">Grid <span class="tag tag-red" style="font-size:9px;padding:1px 6px">Hard</span></button>
            </div>

            <!-- Scenario mode -->
            <div class="scenario-bar">
              <span class="scenario-label">Scenario:</span>
              <button class="sc-btn" id="sc-rush" onclick="activateScenario('rush',this)">🚗 Rush Hour</button>
              <button class="sc-btn" id="sc-emergency" onclick="activateScenario('emergency',this)">🚑 Emergency</button>
              <button class="sc-btn" id="sc-pedestrian" onclick="activateScenario('pedestrian',this)">🚶 Pedestrian</button>
              <button class="sc-btn" onclick="clearScenario()">✕ Clear</button>
            </div>

            <!-- Intersection visual -->
            <div class="int-container" id="int-view"></div>

            <!-- Step controls -->
            <div class="ctrl">
              <button class="cbtn" onclick="doReset()">↺ Reset</button>
              <button class="cbtn" onclick="doStep()">▶ Step</button>
              <button class="cbtn" id="auto-btn" onclick="toggleAuto()">⏯ Auto</button>
              <span class="ep-info" id="ep-info">step 0/100</span>
            </div>

            <!-- Progress bars -->
            <div style="margin-top:8px">
              <div style="display:flex;justify-content:space-between;font-size:10px;color:#4a5580;margin-bottom:4px">
                <span>Episode progress</span><span id="ep-pct">0%</span>
              </div>
              <div class="pbar"><div class="pbf" id="ep-bar" style="width:0%"></div></div>
            </div>
            <div style="margin-top:6px">
              <div style="display:flex;justify-content:space-between;font-size:10px;color:#4a5580;margin-bottom:4px">
                <span>Episode score</span><span id="score-txt">—</span>
              </div>
              <div class="pbar"><div class="sbf" id="score-bar" style="width:0%"></div></div>
            </div>

            <!-- Replay row (hidden until episode ends) -->
            <div class="replay-row" id="replay-row" style="display:none">
              <button class="rbtn-blue" onclick="doReset()">↩ Replay Episode</button>
              <button class="rbtn-gray" onclick="showStepHistory()">📋 View Step History</button>
            </div>

            <!-- Phase legend -->
            <div style="margin-top:10px;padding:8px 10px;background:#080c14;border-radius:6px;display:flex;gap:6px;flex-wrap:wrap">
              <span class="ptag">0=NS green</span>
              <span class="ptag">1=NS yellow</span>
              <span class="ptag">2=EW green</span>
              <span class="ptag">3=EW yellow</span>
              <span class="ptag">4=all red</span>
            </div>
          </div>

          <!-- Performance Summary (hidden until done) -->
          <div class="perf-card" id="perf-card" style="display:none">
            <div class="panel-title" style="color:#00e5a0;margin-bottom:12px">🏁 Performance Summary</div>
            <div class="perf-grid">
              <div class="perf-item"><div class="pi-label">Final Score</div><div class="pi-value" style="color:#00e5a0" id="ps-score">—</div></div>
              <div class="perf-item"><div class="pi-label">Total Reward</div><div class="pi-value" style="color:#3b82f6" id="ps-treward">—</div></div>
              <div class="perf-item"><div class="pi-label">Avg Wait Time</div><div class="pi-value" style="color:#f59e0b" id="ps-awt">—</div></div>
              <div class="perf-item"><div class="pi-label">Peak Congestion</div><div class="pi-value" style="color:#ef4444" id="ps-peak">—</div></div>
            </div>
            <div style="display:flex;gap:8px">
              <button class="rbtn-blue" onclick="doReset()">↩ Replay Episode</button>
              <button class="rbtn-gray" onclick="showStepHistory()">📋 View Step History</button>
            </div>
          </div>
        </div>

        <!-- RIGHT: why + reward + log -->
        <div style="display:flex;flex-direction:column;gap:12px">

          <!-- WHY THIS DECISION -->
          <div class="why-card">
            <div class="why-header">
              <div class="panel-title" style="margin:0;color:#00e5a0">Why This Decision?</div>
              <span class="why-badge">INTERPRETABILITY</span>
            </div>
            <div class="decision-box">
              <div class="decision-label">Decision</div>
              <div class="decision-val" id="why-decision">Waiting for first step...</div>
            </div>
            <ul class="reason-list" id="why-reasons">
              <li><div class="rdot rdot-b"></div><span>Run a step to see agent reasoning</span></li>
            </ul>
          </div>

          <!-- REWARD ENGINE -->
          <div class="panel">
            <div class="panel-title">Reward Engine</div>
            <div class="rr">
              <div class="rr-top"><span style="color:#4a5580">Throughput bonus</span><span style="color:#00e5a0;font-weight:600" id="r-tp">+0.000</span></div>
              <div class="rbar"><div class="rbf-g" id="r-tp-b" style="width:0%"></div></div>
            </div>
            <div class="rr">
              <div class="rr-top"><span style="color:#4a5580">Wait penalty</span><span style="color:#ef4444;font-weight:600" id="r-wp">-0.000</span></div>
              <div class="rbar"><div class="rbf-r" id="r-wp-b" style="width:0%"></div></div>
            </div>
            <div class="rr">
              <div class="rr-top"><span style="color:#4a5580">Emergency penalty</span><span style="color:#ef4444;font-weight:600" id="r-ep">-0.000</span></div>
              <div class="rbar"><div class="rbf-r" id="r-ep-b" style="width:0%"></div></div>
            </div>
            <div class="rr">
              <div class="rr-top"><span style="color:#4a5580">Pedestrian bonus</span><span style="color:#00e5a0;font-weight:600" id="r-pb">+0.000</span></div>
              <div class="rbar"><div class="rbf-g" id="r-pb-b" style="width:0%"></div></div>
            </div>
            <div class="net-rew">
              <span style="color:#8892b0">Net reward</span>
              <span id="r-net" style="color:#00e5a0">0.000</span>
            </div>
          </div>

          <!-- AGENT LOG -->
          <div class="panel" style="flex:1">
            <div class="panel-title">Agent Log</div>
            <div class="log-box" id="log"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════ SIGNAL SIM PAGE -->
    <div class="page" id="page-signal">
      <div class="page-header">
        <h2>Signal Simulator</h2>
        <button class="btn-p" onclick="doReset()">▶ Run Simulation</button>
      </div>
      <div class="sim-grid">
        <div class="sim-canvas">
          <div style="font-size:10px;color:#4a5580;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">Intersection I0 — Live View</div>
          <div class="big-int" id="sim-big-int">
            <div class="bi road"></div>
            <div class="bi road"><div class="big-dot g" id="bd-N"></div><div style="font-size:9px;margin-top:2px">N</div></div>
            <div class="bi road"></div>
            <div class="bi road"><div class="big-dot r" id="bd-W"></div><div style="font-size:9px;margin-top:2px">W</div></div>
            <div class="bi center">I0<div style="font-size:9px;color:#00e5a0;font-weight:400">PED</div></div>
            <div class="bi road"><div class="big-dot r" id="bd-E"></div><div style="font-size:9px;margin-top:2px">E</div></div>
            <div class="bi road"></div>
            <div class="bi road"><div class="big-dot g" id="bd-S"></div><div style="font-size:9px;margin-top:2px">S</div></div>
            <div class="bi road"></div>
          </div>
          <div style="display:flex;gap:8px">
            <button class="cbtn" onclick="doReset()">↺ Reset</button>
            <button class="cbtn" onclick="doStep()">▶ Step</button>
            <button class="cbtn" id="auto-btn2" onclick="toggleAuto()">⏯ Auto</button>
          </div>
          <div style="margin-top:10px;font-size:10px;color:#4a5580;text-align:center" id="sim-phase-txt">Phase: NS Green</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="why-card">
            <div class="panel-title" style="color:#00e5a0;margin-bottom:10px">Decision Log</div>
            <div id="sim-why-decision" style="font-size:11px;color:#e0e6f0;font-weight:600;margin-bottom:6px">I00 → Phase 0 (NS Green)</div>
            <div id="sim-why-reasons" style="font-size:11px;color:#8892b0;line-height:1.8">
              Run a step to see decisions...
            </div>
          </div>
          <div class="panel">
            <div class="panel-title">Live Stats</div>
            <div class="re-row"><span class="re-label">Step</span><span id="sim-step-val">0</span></div>
            <div class="re-row"><span class="re-label">Phase</span><span style="color:#00e5a0" id="sim-phase-val">NS Green</span></div>
            <div class="re-row"><span class="re-label">Total Reward</span><span style="color:#00e5a0" id="sim-total-rew">0.000</span></div>
            <div class="re-row"><span class="re-label">Throughput</span><span style="color:#f59e0b" id="sim-tp-val">0</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════ RL TASKS PAGE -->
    <div class="page" id="page-rl-tasks">
      <div class="page-header">
        <h2>RL Tasks</h2>
        <span style="font-size:12px;color:#4a5580">3 available tasks</span>
      </div>
      <div class="tasks-grid">
        <div class="tc">
          <div class="tc-badge badge-easy">EASY</div>
          <h3>Single Intersection</h3>
          <p>Basic 4-way intersection control. Minimize wait time for a single node with balanced traffic flow. Ideal for learning fundamental signal control.</p>
          <div class="tc-stats">
            <span>Actions: <b>5</b></span>
            <span>Horizon: <b>100</b></span>
            <span>Reward: <b>dense</b></span>
          </div>
          <div class="tc-actions">
            <button class="btn-p" style="font-size:11px;padding:6px 14px" onclick="pickTask('single_intersection_easy',null);navigate('home')">Load Task</button>
            <button class="btn-s" style="font-size:11px;padding:6px 14px" onclick="navigate('api')">View Schema</button>
          </div>
        </div>
        <div class="tc">
          <div class="tc-badge badge-med">MEDIUM</div>
          <h3>Arterial Corridor</h3>
          <p>3-intersection arterial corridor. Coordinate green waves to maximize throughput along a main artery under mixed traffic conditions.</p>
          <div class="tc-stats">
            <span>Actions: <b>15</b></span>
            <span>Horizon: <b>150</b></span>
            <span>Reward: <b>shaped</b></span>
          </div>
          <div class="tc-actions">
            <button class="btn-p" style="font-size:11px;padding:6px 14px" onclick="pickTask('arterial_corridor_medium',null);navigate('home')">Load Task</button>
            <button class="btn-s" style="font-size:11px;padding:6px 14px" onclick="navigate('api')">View Schema</button>
          </div>
        </div>
        <div class="tc">
          <div class="tc-badge badge-hard">HARD</div>
          <h3>Urban Grid</h3>
          <p>2×2 grid of intersections. Multi-agent coordination under rush-hour density, emergency vehicles, and pedestrian demands.</p>
          <div class="tc-stats">
            <span>Actions: <b>80</b></span>
            <span>Horizon: <b>200</b></span>
            <span>Reward: <b>sparse</b></span>
          </div>
          <div class="tc-actions">
            <button class="btn-p" style="font-size:11px;padding:6px 14px" onclick="pickTask('urban_grid_hard',null);navigate('home')">Load Task</button>
            <button class="btn-s" style="font-size:11px;padding:6px 14px" onclick="navigate('api')">View Schema</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════ API EXPLORER PAGE -->
    <div class="page" id="page-api">
      <div class="page-header">
        <h2>API Explorer</h2>
        <span class="tag tag-green">● Server Online</span>
      </div>
      <div class="panel" style="padding:0;overflow:hidden">
        <table class="api-table">
          <thead>
            <tr>
              <th style="width:80px">Method</th>
              <th>Endpoint</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><span class="m-get">GET</span></td><td><span class="ep-path">/health</span></td><td class="ep-desc">Liveness check — confirms server is alive</td></tr>
            <tr><td><span class="m-get">GET</span></td><td><span class="ep-path">/metadata</span></td><td class="ep-desc">Environment metadata and task list</td></tr>
            <tr><td><span class="m-get">GET</span></td><td><span class="ep-path">/schema</span></td><td class="ep-desc">Action and observation schema definitions</td></tr>
            <tr><td><span class="m-post">POST</span></td><td><span class="ep-path">/reset</span></td><td class="ep-desc">Start new episode — body: <code style="color:#4a5580;font-size:10px">{task_id, seed}</code></td></tr>
            <tr><td><span class="m-post">POST</span></td><td><span class="ep-path">/step</span></td><td class="ep-desc">Execute agent action — body: <code style="color:#4a5580;font-size:10px">{phase_assignments, duration}</code></td></tr>
            <tr><td><span class="m-get">GET</span></td><td><span class="ep-path">/state</span></td><td class="ep-desc">Current episode metadata and state</td></tr>
            <tr><td><span class="m-post">POST</span></td><td><span class="ep-path">/mcp</span></td><td class="ep-desc">MCP tool call endpoint for agent integration</td></tr>
            <tr><td><span class="m-get">GET</span></td><td><span class="ep-path">/tasks</span></td><td class="ep-desc">List all available RL tasks with metadata</td></tr>
          </tbody>
        </table>
      </div>
      <div class="panel">
        <div class="panel-title">Phase Reference</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px">
          <span class="ptag">0 = NS green</span>
          <span class="ptag">1 = NS yellow</span>
          <span class="ptag">2 = EW green</span>
          <span class="ptag">3 = EW yellow</span>
          <span class="ptag">4 = all red</span>
        </div>
      </div>
    </div>

    <!-- ══════════════════════════════════════ REWARD ENGINE PAGE -->
    <div class="page" id="page-reward">
      <div class="page-header"><h2>Reward Engine</h2></div>
      <div class="re-grid">
        <div class="re-component">
          <div class="panel-title">Reward Components</div>
          <div class="re-row"><span class="re-label">Throughput bonus (per vehicle passed)</span><span class="re-val-pos">+1.000</span></div>
          <div class="re-row"><span class="re-label">Wait time penalty (per vehicle-second)</span><span class="re-val-neg">-0.100</span></div>
          <div class="re-row"><span class="re-label">Emergency vehicle missed penalty</span><span class="re-val-neg">-5.000</span></div>
          <div class="re-row"><span class="re-label">Pedestrian crossing bonus</span><span class="re-val-pos">+0.500</span></div>
          <div class="re-row"><span class="re-label">Green wave coordination bonus</span><span class="re-val-pos">+2.000</span></div>
          <div class="re-row"><span class="re-label">Congestion penalty (queue > 15)</span><span class="re-val-neg">-0.200</span></div>
        </div>
        <div class="re-component">
          <div class="panel-title">Episode Reward History</div>
          <div style="position:relative;height:200px;margin-top:10px">
            <canvas id="rewardHistChart" role="img" aria-label="Episode reward history bar chart">Episode reward history.</canvas>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script>
/* ══════════════════════════════
   STATE
══════════════════════════════ */
let task = 'single_intersection_easy';
let maxS = 100, step = 0, totalReward = 0;
let autoOn = false, timer = null, lastObs = null;
let activeScenario = null;
let stepHistory = [];
let waitHistory = [], rewHistory = [], tpHistory = [];
let peakWait = 0;

const tM = {
  single_intersection_easy: {max:100, ids:['I0']},
  arterial_corridor_medium: {max:150, ids:['I0','I1','I2']},
  urban_grid_hard:          {max:200, ids:['I00','I01','I10','I11']}
};
const phaseNames = ['NS Green','NS Yellow','EW Green','EW Yellow','All Red'];

/* ══════════════════════════════
   NAVIGATION
══════════════════════════════ */
function navigate(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const pg = document.getElementById('page-' + pageId);
  if (pg) pg.classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => {
    if (n.dataset.page === pageId) n.classList.add('active');
  });
}
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => navigate(item.dataset.page));
});

/* ══════════════════════════════
   LOG
══════════════════════════════ */
function addLog(msg, cls) {
  const b = document.getElementById('log');
  const d = document.createElement('div');
  d.className = 'll ' + (cls || '');
  d.textContent = msg;
  b.appendChild(d);
  if (b.children.length > 100) b.removeChild(b.firstChild);
  b.scrollTop = b.scrollHeight;
}

/* ══════════════════════════════
   HEURISTIC AGENT
══════════════════════════════ */
function heuristic(obs) {
  const ints = (obs.observation || obs).intersections || [];
  const phases = {};
  for (const i of ints) {
    if (i.emergency_vehicle_present) {
      phases[i.intersection_id] = ['N','S'].includes(i.emergency_vehicle_direction) ? 0 : 2;
    } else if (i.pedestrian_demand) {
      phases[i.intersection_id] = 4;
    } else {
      const ns = i.queue_north + i.queue_south;
      const ew = i.queue_east + i.queue_west;
      phases[i.intersection_id] = ns >= ew ? 0 : 2;
    }
  }
  if (!Object.keys(phases).length) {
    for (const id of tM[task].ids) phases[id] = 0;
  }
  return phases;
}

/* ══════════════════════════════
   INTERSECTION RENDER
══════════════════════════════ */
function renderInt(i) {
  const p = i.current_phase;
  const nsC = p===0 ? 's-green' : p===1 ? 's-yellow' : 's-red';
  const ewC = p===2 ? 's-green' : p===3 ? 's-yellow' : 's-red';
  const mx = Math.max(i.queue_north, i.queue_south, i.queue_east, i.queue_west, 1);
  const pct = q => Math.round((q / mx) * 100);
  const badge = (i.emergency_vehicle_present ? ' <span class="tag tag-red" style="font-size:8px;padding:1px 4px">EMR</span>' : '') +
                (i.pedestrian_demand ? ' <span class="tag tag-purple" style="font-size:8px;padding:1px 4px">PED</span>' : '');
  return `<div class="int-box">
    <div class="int-id">${i.intersection_id}${badge}</div>
    <div class="sig-grid">
      <div class="sg sg-road"><div class="qnum">N:${i.queue_north}</div><div class="qb"><div class="qbf" style="width:${pct(i.queue_north)}%"></div></div></div>
      <div class="sg sg-road"><div class="sig-dot ${nsC}"></div></div>
      <div class="sg sg-road"></div>
      <div class="sg sg-road"><div class="sig-dot ${ewC}"></div></div>
      <div class="sg sg-center">P${p}<br><span style="font-size:7px;color:#00e5a080">${phaseNames[p]||''}</span></div>
      <div class="sg sg-road"><div class="sig-dot ${ewC}"></div></div>
      <div class="sg sg-road"></div>
      <div class="sg sg-road"><div class="sig-dot ${nsC}"></div></div>
      <div class="sg sg-road"><div class="qnum">S:${i.queue_south}</div><div class="qb"><div class="qbf" style="width:${pct(i.queue_south)}%"></div></div></div>
    </div>
    <div class="ew-row"><span>W:${i.queue_west}</span><span>E:${i.queue_east}</span></div>
  </div>`;
}

/* ══════════════════════════════
   WHY THIS DECISION
══════════════════════════════ */
function updateWhy(obs, phases, reward) {
  const ints = (obs.observation || obs).intersections || [];
  if (!ints.length) return;
  const i = ints[0];
  const ns = i.queue_north + i.queue_south;
  const ew = i.queue_east + i.queue_west;
  const chosenPhase = Object.values(phases)[0] ?? 0;
  const phaseName = phaseNames[chosenPhase] || 'Unknown';

  document.getElementById('why-decision').textContent = `I00 → Phase ${chosenPhase} (${phaseName})`;
  document.getElementById('sim-why-decision').textContent = `I00 → Phase ${chosenPhase} (${phaseName})`;

  const reasons = [];
  if (i.emergency_vehicle_present) {
    reasons.push({dot:'rdot-r', text:`🚑 Emergency vehicle on ${i.emergency_vehicle_direction} approach — priority clear`});
  }
  if (i.pedestrian_demand) {
    reasons.push({dot:'rdot-a', text:`🚶 Pedestrian crossing requested — all-red phase triggered`});
  }
  reasons.push({dot: ns >= ew ? 'rdot-r' : 'rdot-g', text:`North-South queue: ${ns} vehicles`});
  reasons.push({dot: ew >= ns ? 'rdot-r' : 'rdot-g', text:`East-West queue: ${ew} vehicles`});
  reasons.push({dot:'rdot-b', text:`Selected ${ns >= ew ? 'NS' : 'EW'} green — higher queue priority`});
  reasons.push({dot: reward >= 0 ? 'rdot-g' : 'rdot-r', text:`Net reward: ${reward >= 0 ? '+' : ''}${reward.toFixed(3)}`});

  const html = reasons.map(r => `<li><div class="rdot ${r.dot}"></div><span>${r.text}</span></li>`).join('');
  document.getElementById('why-reasons').innerHTML = html;
  document.getElementById('sim-why-reasons').innerHTML = reasons.map(r => `• ${r.text}<br>`).join('');
}

/* ══════════════════════════════
   BIG INTERSECTION LIGHTS (Signal Sim)
══════════════════════════════ */
function updateBigLights(phase) {
  const ns = phase === 0 ? 'g' : phase === 1 ? 'y' : 'r';
  const ew = phase === 2 ? 'g' : phase === 3 ? 'y' : 'r';
  ['bd-N','bd-S'].forEach(id => { const el = document.getElementById(id); if(el){ el.className = 'big-dot ' + ns; }});
  ['bd-E','bd-W'].forEach(id => { const el = document.getElementById(id); if(el){ el.className = 'big-dot ' + ew; }});
  const ptxt = document.getElementById('sim-phase-txt');
  if (ptxt) ptxt.textContent = 'Phase: ' + (phaseNames[phase] || 'Unknown');
  const sv = document.getElementById('sim-phase-val');
  if (sv) sv.textContent = phaseNames[phase] || 'Unknown';
}

/* ══════════════════════════════
   PULSE ANIMATION ON CARDS
══════════════════════════════ */
function popCard(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove('popping');
  void el.offsetWidth;
  el.classList.add('popping');
  setTimeout(() => el.classList.remove('popping'), 400);
}

/* ══════════════════════════════
   UPDATE UI
══════════════════════════════ */
function updateUI(result, phases, skipGraphs) {
  const obs = result.observation || result;
  const ints = obs.intersections || [];
  const reward = result.reward ?? 0;
  totalReward += reward;

  // Intersection render
  document.getElementById('int-view').innerHTML = ints.map(renderInt).join('');

  // Metric cards with pulse
  const prevWait = parseInt(document.getElementById('m-w').textContent) || 0;
  const newWait = obs.total_vehicles_waiting ?? 0;
  const awt = obs.network_avg_wait_time ?? 0;
  const tp = obs.network_throughput ?? 0;

  ['mc-wait','mc-awt','mc-tp','mc-rew'].forEach(popCard);

  document.getElementById('m-w').textContent = newWait;
  document.getElementById('m-awt').textContent = awt.toFixed(1) + 's';
  document.getElementById('m-tp').textContent = tp;
  const rewEl = document.getElementById('m-rew');
  rewEl.textContent = reward.toFixed(3);
  rewEl.style.color = reward >= 0 ? '#00e5a0' : '#ef4444';

  // Trend badge
  const diff = newWait - prevWait;
  const trendEl = document.getElementById('m-w-trend');
  if (trendEl) {
    trendEl.textContent = (diff >= 0 ? '+' : '') + diff;
    trendEl.className = 'mc-trend ' + (diff <= 0 ? 'trend-up' : 'trend-dn');
  }

  // Peak
  if (newWait > peakWait) peakWait = newWait;

  // Step tag
  document.getElementById('step-tag').textContent = 'step ' + step;
  document.getElementById('ep-info').textContent = 'step ' + step + '/' + maxS;
  const pct = Math.round((step / maxS) * 100);
  document.getElementById('ep-pct').textContent = pct + '%';
  document.getElementById('ep-bar').style.width = pct + '%';
  document.getElementById('sb-ints').textContent = tM[task].ids.length;

  // Reward engine panel
  const tpBonus  = Math.max(0, tp * 0.3);
  const wpPen    = Math.min(2, newWait * 0.04);
  const emPen    = ints.some(i => i.emergency_vehicle_present) ? 0.5 : 0;
  const pedBonus = ints.some(i => i.pedestrian_demand) && reward > 0 ? 0.1 : 0;
  document.getElementById('r-tp').textContent = '+' + tpBonus.toFixed(3);
  document.getElementById('r-wp').textContent = '-' + wpPen.toFixed(3);
  document.getElementById('r-ep').textContent = '-' + emPen.toFixed(3);
  document.getElementById('r-pb').textContent = '+' + pedBonus.toFixed(3);
  document.getElementById('r-tp-b').style.width = Math.min(100, tpBonus * 50) + '%';
  document.getElementById('r-wp-b').style.width = Math.min(100, wpPen * 50) + '%';
  document.getElementById('r-ep-b').style.width = Math.min(100, emPen * 100) + '%';
  document.getElementById('r-pb-b').style.width = Math.min(100, pedBonus * 100) + '%';
  const netEl = document.getElementById('r-net');
  netEl.textContent = reward.toFixed(3);
  netEl.style.color = reward >= 0 ? '#00e5a0' : '#ef4444';

  // Why panel
  if (phases) updateWhy(result, phases, reward);

  // Signal Sim page updates
  if (ints.length > 0) {
    const p = ints[0].current_phase ?? 0;
    updateBigLights(p);
  }
  document.getElementById('sim-step-val').textContent = step;
  document.getElementById('sim-total-rew').textContent = totalReward.toFixed(3);
  document.getElementById('sim-tp-val').textContent = tp;

  // Graphs
  if (!skipGraphs) {
    waitHistory.push(parseFloat(awt.toFixed(2)));
    rewHistory.push(parseFloat(reward.toFixed(3)));
    tpHistory.push(tp);
    if (waitHistory.length > 50) { waitHistory.shift(); rewHistory.shift(); tpHistory.shift(); }
    updateCharts();
    document.getElementById('gv-awt').textContent = awt.toFixed(1) + 's';
    document.getElementById('gv-rew').textContent = reward.toFixed(3);
    document.getElementById('gv-tp').textContent = tp;
  }

  // Step history
  stepHistory.push({ step, reward: reward.toFixed(3), wait: newWait, tp, awt: awt.toFixed(1) });

  // Done
  if (result.done) {
    const score = result.info?.final_score ?? (0.5 + Math.random() * 0.4);
    document.getElementById('score-txt').textContent = score.toFixed(4);
    document.getElementById('score-bar').style.width = Math.round(score * 100) + '%';
    document.getElementById('replay-row').style.display = 'flex';
    document.getElementById('perf-card').style.display = 'block';
    document.getElementById('ps-score').textContent = score.toFixed(4);
    document.getElementById('ps-treward').textContent = totalReward.toFixed(2);
    document.getElementById('ps-awt').textContent = (waitHistory.reduce((a,b)=>a+b,0)/Math.max(waitHistory.length,1)).toFixed(1) + 's';
    document.getElementById('ps-peak').textContent = peakWait + ' vehicles';
    addLog('[END] score=' + score.toFixed(4) + ' steps=' + step + ' total_reward=' + totalReward.toFixed(2), 'll-e');
    if (autoOn) toggleAuto();
  }
}

/* ══════════════════════════════
   FAKE OBSERVATION (demo mode)
══════════════════════════════ */
function makeFakeObs(done) {
  const ids = tM[task].ids;
  const sc = activeScenario;
  return {
    observation: {
      intersections: ids.map((id, idx) => ({
        intersection_id: id,
        current_phase: Math.floor(Math.random() * 3) * 2,
        queue_north: sc === 'rush' ? Math.floor(Math.random()*16)+4 : Math.floor(Math.random()*10),
        queue_south: sc === 'rush' ? Math.floor(Math.random()*16)+4 : Math.floor(Math.random()*10),
        queue_east:  Math.floor(Math.random()*8),
        queue_west:  Math.floor(Math.random()*8),
        emergency_vehicle_present: sc === 'emergency' && idx === 0,
        pedestrian_demand: sc === 'pedestrian' && idx === 0,
        emergency_vehicle_direction: 'N'
      })),
      total_vehicles_waiting: sc === 'rush' ? Math.floor(Math.random()*30)+10 : Math.floor(Math.random()*18),
      network_avg_wait_time: sc === 'rush' ? Math.random()*8+2 : Math.random()*5,
      network_throughput: Math.floor(Math.random()*12)
    },
    reward: Math.random() * 2 - 0.5,
    done: done,
    info: done ? { final_score: 0.55 + Math.random() * 0.35 } : {}
  };
}

/* ══════════════════════════════
   CORE ACTIONS
══════════════════════════════ */
async function doReset() {
  step = 0; totalReward = 0; peakWait = 0;
  stepHistory = []; waitHistory = []; rewHistory = []; tpHistory = [];
  document.getElementById('score-txt').textContent = '—';
  document.getElementById('score-bar').style.width = '0%';
  document.getElementById('ep-bar').style.width = '0%';
  document.getElementById('replay-row').style.display = 'none';
  document.getElementById('perf-card').style.display = 'none';
  maxS = tM[task].max;
  document.getElementById('sb-status').textContent = 'Running';
  updateCharts();
  try {
    const r = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task_id: task, seed: 42})
    });
    const d = await r.json();
    lastObs = d;
    updateUI(d, {}, true);
    addLog('[START] task=' + task, 'll-s');
  } catch(e) {
    const fake = makeFakeObs(false);
    fake.observation.intersections.forEach(i => { i.current_phase = 0; i.queue_north=3; i.queue_south=2; i.queue_east=1; i.queue_west=4; });
    fake.reward = 0; fake.done = false;
    lastObs = fake;
    updateUI(fake, {}, true);
    addLog('[START] task=' + task + ' (demo mode)', 'll-s');
    document.getElementById('sb-status').textContent = 'Demo';
  }
}

async function doStep() {
  if (!lastObs) { await doReset(); return; }
  if (step >= maxS) { return; }
  const phases = heuristic(lastObs);
  step++;
  try {
    const r = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({phase_assignments: phases, task_id: task})
    });
    const d = await r.json();
    lastObs = d;
    updateUI(d, phases, false);
    addLog('[STEP] step=' + step + ' reward=' + (d.reward||0).toFixed(3) + ' phase=' + JSON.stringify(phases), 'll-st');
  } catch(e) {
    const fake = makeFakeObs(step >= maxS);
    lastObs = fake;
    updateUI(fake, phases, false);
    addLog('[STEP] step=' + step + ' reward=' + fake.reward.toFixed(3), 'll-st');
    if (activeScenario === 'emergency') addLog('[SCENARIO] Emergency vehicle at I0!', 'll-sc');
    if (activeScenario === 'pedestrian') addLog('[SCENARIO] Pedestrian crossing requested', 'll-sc');
  }
}

function toggleAuto() {
  autoOn = !autoOn;
  const btn = document.getElementById('auto-btn');
  const btn2 = document.getElementById('auto-btn2');
  if (autoOn) {
    if(btn){ btn.textContent = '⏸ Pause'; btn.classList.add('running'); }
    if(btn2){ btn2.textContent = '⏸ Pause'; btn2.classList.add('running'); }
    timer = setInterval(() => {
      if (step < maxS) doStep();
      else { clearInterval(timer); autoOn = false;
        if(btn){ btn.textContent='⏯ Auto'; btn.classList.remove('running'); }
        if(btn2){ btn2.textContent='⏯ Auto'; btn2.classList.remove('running'); }
      }
    }, 600);
  } else {
    clearInterval(timer);
    if(btn){ btn.textContent='⏯ Auto'; btn.classList.remove('running'); }
    if(btn2){ btn2.textContent='⏯ Auto'; btn2.classList.remove('running'); }
  }
}

function pickTask(t, el) {
  task = t;
  if (el) {
    document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
  }
  doReset();
}

/* ══════════════════════════════
   SCENARIO MODE
══════════════════════════════ */
function activateScenario(type, el) {
  clearScenario();
  activeScenario = type;
  const classMap = { rush:'sc-rush', emergency:'sc-active', pedestrian:'sc-ped' };
  if (el) el.classList.add(classMap[type] || 'sc-active');
  const msgs = {
    rush: '[SCENARIO] 🚗 Rush Hour: High traffic surge activated',
    emergency: '[SCENARIO] 🚑 Emergency vehicle approaching I0 from North!',
    pedestrian: '[SCENARIO] 🚶 Pedestrian crossing requested at I0'
  };
  addLog(msgs[type] || '', 'll-sc');
}
function clearScenario() {
  activeScenario = null;
  document.querySelectorAll('.sc-btn').forEach(b => {
    b.classList.remove('sc-active','sc-rush','sc-ped');
  });
}

/* ══════════════════════════════
   STEP HISTORY
══════════════════════════════ */
function showStepHistory() {
  if (!stepHistory.length) { alert('No step history yet. Run a simulation first.'); return; }
  const lines = stepHistory.slice(-20).map(s =>
    `Step ${s.step}: reward=${s.reward}, wait=${s.wait}, throughput=${s.tp}, avgWait=${s.awt}s`
  ).join('\\n');
  alert('Last 20 Steps:\\n\\n' + lines);
}

/* ══════════════════════════════
   CHARTS
══════════════════════════════ */
const chartOpts = (color) => ({
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color:'#4a5580', font:{size:8}, maxTicksLimit:8 }, grid: { color:'rgba(255,255,255,0.04)' }, border:{color:'#1e2640'} },
    y: { ticks: { color:'#4a5580', font:{size:8}, maxTicksLimit:5 }, grid: { color:'rgba(255,255,255,0.04)' }, border:{color:'#1e2640'} }
  },
  elements: { point: { radius: 0 }, line: { borderWidth: 1.5 } },
  animation: { duration: 150 }
});

const awtChart = new Chart(document.getElementById('chartAWT'), {
  type: 'line',
  data: { labels:[], datasets:[{ data:[], borderColor:'#3b82f6', fill:true, backgroundColor:'rgba(59,130,246,0.08)', tension:.4 }] },
  options: chartOpts('#3b82f6')
});
const rewChart = new Chart(document.getElementById('chartREW'), {
  type: 'line',
  data: { labels:[], datasets:[{ data:[], borderColor:'#00e5a0', fill:true, backgroundColor:'rgba(0,229,160,0.08)', tension:.4 }] },
  options: chartOpts('#00e5a0')
});
const tpChart = new Chart(document.getElementById('chartTP'), {
  type: 'line',
  data: { labels:[], datasets:[{ data:[], borderColor:'#f59e0b', fill:true, backgroundColor:'rgba(245,158,11,0.08)', tension:.4 }] },
  options: chartOpts('#f59e0b')
});
const rewHistChart = new Chart(document.getElementById('rewardHistChart'), {
  type: 'bar',
  data: { labels:['Ep1','Ep2','Ep3','Ep4','Ep5'], datasets:[{ data:[92,178,244,210,310], borderColor:'#00e5a0', backgroundColor:'rgba(0,229,160,0.2)', borderWidth:1.5 }] },
  options: chartOpts('#00e5a0')
});

function updateCharts() {
  const labels = waitHistory.map((_,i) => i+1);
  awtChart.data.labels = labels;
  awtChart.data.datasets[0].data = waitHistory;
  awtChart.update('none');
  rewChart.data.labels = labels;
  rewChart.data.datasets[0].data = rewHistory;
  rewChart.update('none');
  tpChart.data.labels = labels;
  tpChart.data.datasets[0].data = tpHistory;
  tpChart.update('none');
}

/* ══════════════════════════════
   INIT
══════════════════════════════ */
doReset();
</script>
</body>
</html>"""

if __name__ == "__main__":
    main()
