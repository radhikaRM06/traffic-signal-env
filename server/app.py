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
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrafficAI — OpenEnv</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',sans-serif}
body{background:#0a0e1a;color:#e0e6f0;min-height:100vh;display:flex}
.sidebar{width:200px;background:#0d1120;border-right:1px solid #1e2640;display:flex;flex-direction:column;min-height:100vh;flex-shrink:0}
.logo{padding:20px 16px 16px;border-bottom:1px solid #1e2640}
.logo-title{font-size:18px;font-weight:700;color:#00e5a0}
.logo-sub{font-size:9px;color:#4a5580;letter-spacing:2px;margin-top:1px}
.dot{width:7px;height:7px;border-radius:50%;background:#00e5a0;display:inline-block}
.backend-tag{margin-top:10px;display:flex;align-items:center;gap:5px;font-size:10px;color:#4a5580}
.stats-block{padding:14px 16px;border-bottom:1px solid #1e2640}
.stats-label{font-size:9px;color:#4a5580;letter-spacing:2px;margin-bottom:8px}
.stat-row{display:flex;justify-content:space-between;font-size:11px;padding:3px 0;color:#8892b0}
.stat-val{color:#00e5a0;font-weight:600}
.nav{padding:12px 0;flex:1}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 16px;font-size:12px;color:#4a5580;cursor:pointer;transition:all 0.15s;border-left:2px solid transparent}
.nav-item:hover{color:#8892b0;background:#111827}
.nav-item.active{color:#00e5a0;background:#0f1f30;border-left-color:#00e5a0}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{padding:14px 20px;border-bottom:1px solid #1e2640;display:flex;align-items:center;gap:12px;background:#0a0e1a;flex-shrink:0}
.topbar-title{font-size:15px;font-weight:600}
.tag{padding:3px 10px;border-radius:20px;font-size:10px;font-weight:600}
.tag-green{background:#00e5a022;color:#00e5a0;border:1px solid #00e5a044}
.tag-amber{background:#f59e0b22;color:#f59e0b;border:1px solid #f59e0b44}
.tag-red{background:#ef444422;color:#ef4444;border:1px solid #ef444444}
.tag-blue{background:#3b82f622;color:#3b82f6;border:1px solid #3b82f644}
.content{padding:16px 20px;overflow-y:auto;flex:1;display:flex;flex-direction:column;gap:14px}
.hero{background:#0f1f30;border:1px solid #1e3050;border-radius:12px;padding:24px 28px}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:#00e5a012;border:1px solid #00e5a030;border-radius:20px;padding:4px 12px;font-size:10px;color:#00e5a0;letter-spacing:1px;margin-bottom:12px}
.hero h1{font-size:26px;font-weight:700;margin-bottom:8px}
.hero h1 span{color:#00e5a0}
.hero p{font-size:12px;color:#4a5580;line-height:1.6;max-width:500px;margin-bottom:16px}
.btn-row{display:flex;gap:10px}
.btn-p{background:#00e5a0;color:#0a0e1a;border:none;padding:9px 20px;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer}
.btn-s{background:transparent;color:#e0e6f0;border:1px solid #2a3550;padding:9px 20px;border-radius:8px;font-size:12px;cursor:pointer}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
.mc{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:14px 16px}
.mc-label{font-size:10px;color:#4a5580;margin-bottom:6px;letter-spacing:0.5px}
.mc-val{font-size:22px;font-weight:700}
.mc-sub{font-size:10px;color:#4a5580;margin-top:3px}
.two-col{display:grid;grid-template-columns:1.2fr 0.8fr;gap:14px}
.panel{background:#0d1120;border:1px solid #1e2640;border-radius:10px;padding:14px 16px}
.panel-title{font-size:11px;font-weight:600;color:#8892b0;margin-bottom:12px;display:flex;align-items:center;gap:8px;letter-spacing:0.5px;text-transform:uppercase}
.pulse{width:7px;height:7px;border-radius:50%;background:#00e5a0;animation:pulse 2s infinite;display:inline-block}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.task-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.task-btn{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:6px 12px;border-radius:6px;font-size:11px;cursor:pointer}
.task-btn.active{background:#0f2a1a;border-color:#00e5a0;color:#00e5a0}
.int-container{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;min-height:200px;align-items:center}
.int-box{background:#111827;border:1px solid #1e2640;border-radius:8px;padding:10px;text-align:center}
.int-id{font-size:10px;color:#4a5580;margin-bottom:8px;font-weight:600;letter-spacing:1px}
.sig-grid{display:grid;grid-template-columns:36px 36px 36px;grid-template-rows:36px 36px 36px;gap:2px;margin:0 auto}
.sg{border-radius:4px;display:flex;align-items:center;justify-content:center;flex-direction:column}
.sg-road{background:#0a0e1a}
.sg-center{background:#161e30;font-size:8px;color:#4a5580}
.sig-dot{width:10px;height:10px;border-radius:50%}
.s-green{background:#00e5a0}
.s-red{background:#ef4444}
.s-yellow{background:#f59e0b}
.qnum{font-size:9px;color:#8892b0;font-weight:600}
.qb{height:3px;background:#1e2640;border-radius:2px;margin-top:2px;width:28px}
.qbf{height:100%;background:#00e5a0;border-radius:2px;transition:width 0.4s}
.ew-row{display:flex;justify-content:space-between;font-size:9px;color:#4a5580;margin-top:4px}
.ctrl{display:flex;gap:8px;margin-top:10px;align-items:center;flex-wrap:wrap}
.cbtn{background:#111827;border:1px solid #1e2640;color:#8892b0;padding:5px 14px;border-radius:6px;font-size:11px;cursor:pointer}
.cbtn.running{background:#0f2a1a;border-color:#00e5a0;color:#00e5a0}
.ep-info{font-size:10px;color:#4a5580;margin-left:auto}
.pbar{height:4px;background:#1e2640;border-radius:2px;overflow:hidden;margin-top:6px}
.pbf{height:100%;background:#00e5a0;border-radius:2px;transition:width 0.4s}
.sbf{height:100%;background:#f59e0b;border-radius:2px;transition:width 0.5s}
.log-box{background:#080c14;border:1px solid #1e2640;border-radius:6px;padding:10px;height:110px;overflow-y:auto;font-family:monospace;font-size:10px;margin-top:8px}
.ll{padding:1px 0;color:#4a5580}
.ll-s{color:#00e5a0}
.ll-st{color:#3b82f6}
.ll-e{color:#f59e0b;font-weight:600}
.ll-err{color:#ef4444}
.rr{display:flex;flex-direction:column;gap:3px;margin-bottom:8px}
.rr-top{display:flex;justify-content:space-between;font-size:11px}
.rbar{height:4px;background:#1e2640;border-radius:2px;overflow:hidden}
.rbf-g{height:100%;background:#00e5a0;border-radius:2px;transition:width 0.3s}
.rbf-r{height:100%;background:#ef4444;border-radius:2px;transition:width 0.3s}
.net-rew{border-top:1px solid #1e2640;padding-top:10px;display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-top:4px}
.api-r{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #111827;font-size:11px}
.api-r:last-child{border:none}
.m-get{background:#3b82f622;color:#3b82f6;border:1px solid #3b82f644;font-size:9px;padding:2px 7px;border-radius:3px;font-weight:700;min-width:38px;text-align:center}
.m-post{background:#00e5a022;color:#00e5a0;border:1px solid #00e5a044;font-size:9px;padding:2px 7px;border-radius:3px;font-weight:700;min-width:38px;text-align:center}
.api-path{font-family:monospace;color:#e0e6f0}
.api-d{color:#4a5580;font-size:10px;margin-left:auto}
.ptag{font-size:9px;padding:2px 8px;border-radius:10px;border:1px solid #1e2640;color:#4a5580;display:inline-block;margin:2px}
</style></head><body>
<div class="sidebar">
  <div class="logo">
    <div class="logo-title">TrafficAI</div>
    <div class="logo-sub">OPENENV SIGNAL CONTROL</div>
    <div class="backend-tag"><span class="dot"></span><span>Backend: Online</span></div>
  </div>
  <div class="stats-block">
    <div class="stats-label">QUICK STATS</div>
    <div class="stat-row"><span>Intersections</span><span class="stat-val" id="sb-ints">1</span></div>
    <div class="stat-row"><span>Tasks</span><span class="stat-val">3</span></div>
    <div class="stat-row"><span>Mode</span><span class="stat-val">Heuristic</span></div>
    <div class="stat-row"><span>Status</span><span class="stat-val" id="sb-status">Ready</span></div>
  </div>
  <div class="nav">
    <div class="nav-item active">&#8962; Home &amp; Overview</div>
    <div class="nav-item">&#9711; Signal Sim</div>
    <div class="nav-item">&#9636; RL Tasks</div>
    <div class="nav-item">&#9889; API Explorer</div>
    <div class="nav-item">&#9672; Reward Engine</div>
  </div>
</div>
<div class="main">
  <div class="topbar">
    <span class="topbar-title">Traffic Signal Control — OpenEnv</span>
    <span class="tag tag-green" id="status-tag">&#9679; Running</span>
    <span class="tag tag-blue">OpenEnv v1.0</span>
    <span style="margin-left:auto;font-size:10px;color:#4a5580" id="clock-txt">step 0</span>
  </div>
  <div class="content">
    <div class="hero">
      <div class="hero-badge">&#9733; INTELLIGENCE LAYERED ON OPENENV</div>
      <h1>Traffic<span>AI</span> Platform</h1>
      <p>Next-generation reinforcement learning environment for urban traffic signal control. AI agents minimize vehicle wait times, handle emergencies, and coordinate green waves across city intersections.</p>
      <div class="btn-row">
        <button class="btn-p" onclick="doReset()">&#9654; Start Simulation</button>
        <button class="btn-s">&#9711; View RL Tasks</button>
      </div>
    </div>
    <div class="metrics">
      <div class="mc"><div class="mc-label">VEHICLES WAITING</div><div class="mc-val" id="m-w" style="color:#00e5a0">0</div><div class="mc-sub">across network</div></div>
      <div class="mc"><div class="mc-label">AVG WAIT TIME</div><div class="mc-val" id="m-awt" style="color:#3b82f6">0.0s</div><div class="mc-sub">seconds</div></div>
      <div class="mc"><div class="mc-label">THROUGHPUT</div><div class="mc-val" id="m-tp" style="color:#f59e0b">0</div><div class="mc-sub">vehicles / tick</div></div>
      <div class="mc"><div class="mc-label">REWARD</div><div class="mc-val" id="m-rew" style="color:#00e5a0">0.000</div><div class="mc-sub">this step</div></div>
    </div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title"><span class="pulse"></span>Live Intersection View</div>
        <div class="task-row">
          <button class="task-btn active" onclick="pickTask('single_intersection_easy',this)">Single <span class="tag tag-green" style="font-size:9px;padding:1px 6px">Easy</span></button>
          <button class="task-btn" onclick="pickTask('arterial_corridor_medium',this)">Arterial <span class="tag tag-amber" style="font-size:9px;padding:1px 6px">Med</span></button>
          <button class="task-btn" onclick="pickTask('urban_grid_hard',this)">Grid <span class="tag tag-red" style="font-size:9px;padding:1px 6px">Hard</span></button>
        </div>
        <div class="int-container" id="int-view"></div>
        <div class="ctrl">
          <button class="cbtn" onclick="doReset()">&#8635; Reset</button>
          <button class="cbtn" onclick="doStep()">&#9654; Step</button>
          <button class="cbtn" id="auto-btn" onclick="toggleAuto()">&#9199; Auto</button>
          <span class="ep-info" id="ep-info">step 0/100</span>
        </div>
        <div style="margin-top:8px">
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#4a5580;margin-bottom:4px"><span>Episode progress</span><span id="ep-pct">0%</span></div>
          <div class="pbar"><div class="pbf" id="ep-bar" style="width:0%"></div></div>
        </div>
        <div style="margin-top:6px">
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#4a5580;margin-bottom:4px"><span>Episode score</span><span id="score-txt">—</span></div>
          <div class="pbar"><div class="sbf" id="score-bar" style="width:0%"></div></div>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:12px">
        <div class="panel" style="flex:1">
          <div class="panel-title">Reward Engine</div>
          <div class="rr"><div class="rr-top"><span style="color:#4a5580;font-size:11px">Throughput bonus</span><span style="color:#00e5a0;font-weight:600;font-size:11px" id="r-tp">+0.000</span></div><div class="rbar"><div class="rbf-g" id="r-tp-b" style="width:0%"></div></div></div>
          <div class="rr"><div class="rr-top"><span style="color:#4a5580;font-size:11px">Wait penalty</span><span style="color:#ef4444;font-weight:600;font-size:11px" id="r-wp">-0.000</span></div><div class="rbar"><div class="rbf-r" id="r-wp-b" style="width:0%"></div></div></div>
          <div class="rr"><div class="rr-top"><span style="color:#4a5580;font-size:11px">Emergency penalty</span><span style="color:#ef4444;font-weight:600;font-size:11px" id="r-ep">-0.000</span></div><div class="rbar"><div class="rbf-r" id="r-ep-b" style="width:0%"></div></div></div>
          <div class="rr"><div class="rr-top"><span style="color:#4a5580;font-size:11px">Pedestrian bonus</span><span style="color:#00e5a0;font-weight:600;font-size:11px" id="r-pb">+0.000</span></div><div class="rbar"><div class="rbf-g" id="r-pb-b" style="width:0%"></div></div></div>
          <div class="net-rew"><span style="color:#8892b0">Net reward</span><span id="r-net" style="color:#00e5a0">0.000</span></div>
        </div>
        <div class="panel" style="flex:1">
          <div class="panel-title">Agent Log</div>
          <div class="log-box" id="log"></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">API Endpoints</div>
      <div class="api-r"><span class="m-get">GET</span><span class="api-path">/health</span><span class="api-d">liveness check</span></div>
      <div class="api-r"><span class="m-get">GET</span><span class="api-path">/metadata</span><span class="api-d">environment metadata</span></div>
      <div class="api-r"><span class="m-get">GET</span><span class="api-path">/schema</span><span class="api-d">action / observation schema</span></div>
      <div class="api-r"><span class="m-post">POST</span><span class="api-path">/reset</span><span class="api-d">start new episode</span></div>
      <div class="api-r"><span class="m-post">POST</span><span class="api-path">/step</span><span class="api-d">execute agent action</span></div>
      <div class="api-r"><span class="m-get">GET</span><span class="api-path">/state</span><span class="api-d">episode metadata</span></div>
      <div class="api-r"><span class="m-post">POST</span><span class="api-path">/mcp</span><span class="api-d">MCP tool call</span></div>
      <div class="api-r"><span class="m-get">GET</span><span class="api-path">/tasks</span><span class="api-d">list all RL tasks</span></div>
      <div style="margin-top:10px"><span class="ptag">0=NS green</span><span class="ptag">1=NS yellow</span><span class="ptag">2=EW green</span><span class="ptag">3=EW yellow</span><span class="ptag">4=all red</span></div>
    </div>
  </div>
</div>
<script>
let task='single_intersection_easy',maxS=100,step=0,autoOn=false,timer=null,lastObs=null;
const tM={single_intersection_easy:{max:100,ids:['I0']},arterial_corridor_medium:{max:150,ids:['I0','I1','I2']},urban_grid_hard:{max:200,ids:['I00','I01','I10','I11']}};
function addLog(msg,cls){const b=document.getElementById('log');const d=document.createElement('div');d.className='ll '+(cls||'');d.textContent=msg;b.appendChild(d);if(b.children.length>80)b.removeChild(b.firstChild);b.scrollTop=b.scrollHeight;}
function heuristic(obs){const ints=(obs.observation||obs).intersections||[];const phases={};for(const i of ints){if(i.emergency_vehicle_present)phases[i.intersection_id]=['N','S'].includes(i.emergency_vehicle_direction)?0:2;else if(i.pedestrian_demand)phases[i.intersection_id]=4;else{const ns=i.queue_north+i.queue_south,ew=i.queue_east+i.queue_west;phases[i.intersection_id]=ns>=ew?0:2;}}if(!Object.keys(phases).length)for(const id of tM[task].ids)phases[id]=0;return phases;}
function renderInt(i){const p=i.current_phase;const nsC=p===0?'s-green':p===1?'s-yellow':'s-red';const ewC=p===2?'s-green':p===3?'s-yellow':'s-red';const mx=Math.max(i.queue_north,i.queue_south,i.queue_east,i.queue_west,1);const pct=q=>Math.round((q/mx)*100);return `<div class="int-box"><div class="int-id">${i.intersection_id}${i.emergency_vehicle_present?' [EMR]':''}${i.pedestrian_demand?' [PED]':''}</div><div class="sig-grid"><div class="sg sg-road"><div class="qnum">N:${i.queue_north}</div><div class="qb"><div class="qbf" style="width:${pct(i.queue_north)}%"></div></div></div><div class="sg sg-road"><div class="sig-dot ${nsC}"></div></div><div class="sg sg-road"></div><div class="sg sg-road"><div class="sig-dot ${ewC}"></div></div><div class="sg sg-center">P${p}</div><div class="sg sg-road"><div class="sig-dot ${ewC}"></div></div><div class="sg sg-road"></div><div class="sg sg-road"><div class="sig-dot ${nsC}"></div></div><div class="sg sg-road"><div class="qnum">S:${i.queue_south}</div><div class="qb"><div class="qbf" style="width:${pct(i.queue_south)}%"></div></div></div></div><div class="ew-row"><span>W:${i.queue_west}</span><span>E:${i.queue_east}</span></div></div>`;}
function updateUI(result){const obs=result.observation||result;const ints=obs.intersections||[];document.getElementById('int-view').innerHTML=ints.map(renderInt).join('');document.getElementById('m-w').textContent=obs.total_vehicles_waiting??0;document.getElementById('m-awt').textContent=((obs.network_avg_wait_time??0)).toFixed(1)+'s';document.getElementById('m-tp').textContent=obs.network_throughput??0;const rew=result.reward??0;const rewEl=document.getElementById('m-rew');rewEl.textContent=rew.toFixed(3);rewEl.style.color=rew>=0?'#00e5a0':'#ef4444';document.getElementById('clock-txt').textContent='step '+step;document.getElementById('ep-info').textContent='step '+step+'/'+maxS;const pct=Math.round((step/maxS)*100);document.getElementById('ep-pct').textContent=pct+'%';document.getElementById('ep-bar').style.width=pct+'%';document.getElementById('sb-ints').textContent=tM[task].ids.length;const hasE=ints.some(i=>i.emergency_vehicle_present),hasP=ints.some(i=>i.pedestrian_demand);const tp=Math.max(0,(obs.network_throughput||0)*0.3);const wp=Math.min(2,(obs.total_vehicles_waiting||0)*0.04);const ep=hasE?0.5:0;const pb=hasP&&rew>0?0.1:0;document.getElementById('r-tp').textContent='+'+tp.toFixed(3);document.getElementById('r-wp').textContent='-'+wp.toFixed(3);document.getElementById('r-ep').textContent='-'+ep.toFixed(3);document.getElementById('r-pb').textContent='+'+pb.toFixed(3);document.getElementById('r-tp-b').style.width=Math.min(100,tp*50)+'%';document.getElementById('r-wp-b').style.width=Math.min(100,wp*50)+'%';document.getElementById('r-ep-b').style.width=Math.min(100,ep*100)+'%';document.getElementById('r-pb-b').style.width=Math.min(100,pb*100)+'%';const netEl=document.getElementById('r-net');netEl.textContent=rew.toFixed(3);netEl.style.color=rew>=0?'#00e5a0':'#ef4444';if(result.done&&result.info&&result.info.final_score!=null){const s=result.info.final_score;document.getElementById('score-txt').textContent=s.toFixed(4);document.getElementById('score-bar').style.width=Math.round(s*100)+'%';addLog('[END] score='+s.toFixed(4)+' steps='+step,'ll-e');if(autoOn)toggleAuto();}}
function simStep(){const ids=tM[task].ids;const fake={observation:{intersections:ids.map(id=>({intersection_id:id,current_phase:Math.floor(Math.random()*3)*2,queue_north:Math.floor(Math.random()*10),queue_south:Math.floor(Math.random()*10),queue_east:Math.floor(Math.random()*8),queue_west:Math.floor(Math.random()*8),emergency_vehicle_present:Math.random()<0.05,pedestrian_demand:Math.random()<0.1,emergency_vehicle_direction:'N'})),total_vehicles_waiting:Math.floor(Math.random()*20),network_avg_wait_time:Math.random()*8,network_throughput:Math.floor(Math.random()*12)},reward:(Math.random()*2-0.5),done:step>=maxS,info:step>=maxS?{final_score:0.55+Math.random()*0.3}:{}};lastObs=fake;step++;updateUI(fake);addLog('[STEP] step='+step+' reward='+(fake.reward).toFixed(3),'ll-st');if(fake.done&&autoOn)toggleAuto();}
async function doReset(){step=0;document.getElementById('score-txt').textContent='—';document.getElementById('score-bar').style.width='0%';document.getElementById('ep-bar').style.width='0%';maxS=tM[task].max;document.getElementById('sb-status').textContent='Ready';try{const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:task,seed:42})});const d=await r.json();lastObs=d;updateUI(d);addLog('[START] task='+task,'ll-s');document.getElementById('sb-status').textContent='Running';}catch(e){addLog('[START] task='+task+' (demo)','ll-s');const ids=tM[task].ids;const fake={observation:{intersections:ids.map(id=>({intersection_id:id,current_phase:0,queue_north:3,queue_south:2,queue_east:1,queue_west:4,emergency_vehicle_present:false,pedestrian_demand:false,emergency_vehicle_direction:'N'})),total_vehicles_waiting:0,network_avg_wait_time:0,network_throughput:0},reward:0,done:false,info:{}};lastObs=fake;updateUI(fake);document.getElementById('sb-status').textContent='Demo';}}
async function doStep(){if(!lastObs){await doReset();return;}const phases=heuristic(lastObs);try{const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({phase_assignments:phases,task_id:task})});const d=await r.json();lastObs=d;step++;updateUI(d);addLog('[STEP] step='+step+' reward='+(d.reward||0).toFixed(3),'ll-st');}catch(e){simStep();}}
function toggleAuto(){autoOn=!autoOn;const btn=document.getElementById('auto-btn');if(autoOn){btn.textContent='⏸ Pause';btn.classList.add('running');timer=setInterval(doStep,600);}else{btn.textContent='⏯ Auto';btn.classList.remove('running');clearInterval(timer);}}
function pickTask(t,el){task=t;document.querySelectorAll('.task-btn').forEach(b=>b.classList.remove('active'));el.classList.add('active');doReset();}
doReset();
</script>
</body></html>"""

if __name__ == "__main__":
    main()
