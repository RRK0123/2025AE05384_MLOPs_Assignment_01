"""Inline dashboard assets stay covered by the bundle's Python source checksums."""

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart Disease API · Monitoring</title>
<style>
:root{font-family:system-ui,sans-serif;color:#203139;background:#f3f6f6;font-size:15px}
*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:38px 26px}
header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:28px}
h1{font-size:30px;letter-spacing:-1px;margin:8px 0}h2{font-size:17px;margin:0 0 18px}
p{line-height:1.6}.eyebrow{font-size:12px;letter-spacing:2px;color:#42656b;font-weight:700}
.muted,small{color:#566b73}.badge{padding:9px 15px;border-radius:30px;background:#d9efea;color:#176550}
.bad{background:#fce3dd;color:#9d3926}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.card,.panel{background:white;border:1px solid #dae4e5;border-radius:12px;padding:22px}
.label{font-size:13px;color:#566b73}.value{font-size:32px;font-weight:650;margin:10px 0}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;margin:18px 0}
svg{width:100%;height:140px;display:block}.chart-label{display:flex;justify-content:space-between;font-size:12px;color:#566b73}
.row{display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid #eef2f2}
code{font-size:12px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #eef2f2}th{color:#566b73;font-weight:500}
.table-wrap{overflow-x:auto}footer{font-size:12px;color:#566b73;margin-top:20px;line-height:1.7}
a{color:#146b72}button{background:#fff;border:1px solid #b8cacc;border-radius:6px;padding:8px 12px;cursor:pointer}
#error{background:#fce3dd;padding:12px;border-radius:6px;margin-bottom:16px}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}header{align-items:flex-start}main{padding:24px 16px}}
</style></head><body><main>
<header><div><div class="eyebrow">HEART DISEASE / SERVING</div>
<h1>API monitoring</h1><div class="muted">Live service activity · refreshes every 3 seconds</div></div>
<span id="state" class="badge" role="status">Connecting…</span></header>
<div id="error" role="alert" hidden></div>
<section class="cards" aria-label="Service totals">
<div class="card"><div class="label">Completed requests</div><div class="value" id="requests">—</div><small>Since this process started</small></div>
<div class="card"><div class="label">Mean response time</div><div class="value" id="latency">—</div><small>All counted HTTP requests</small></div>
<div class="card"><div class="label">Server errors · 5xx</div><div class="value" id="errors">—</div><small id="rejected">— client errors · 4xx</small></div>
<div class="card"><div class="label">Predicted records</div><div class="value" id="records">—</div><small id="batches">— successful batches</small></div>
</section>
<section class="grid"><div class="panel"><h2>Recent request traffic</h2>
<div class="muted" id="traffic-summary">Waiting for metrics</div>
<svg id="traffic" viewBox="0 0 600 140" role="img" aria-label="Requests in five second intervals"><title>Request traffic</title></svg>
<div class="chart-label"><span>Up to 5 minutes ago</span><span>Now · requests / 5 seconds</span></div>
</div><div class="panel"><h2>Model and process</h2>
<div class="row"><span>Model run</span><code id="model">—</code></div>
<div class="row"><span>Uptime</span><span id="uptime">—</span></div>
<div class="row"><span>Requests in progress</span><span id="active">—</span></div>
<div class="row"><span>Mean inference / batch</span><span id="inference">—</span></div>
<div class="row"><span>Process ID</span><span id="pid">—</span></div>
</div></section>
<section class="grid"><div class="panel"><h2>Requests by endpoint and status</h2>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>Endpoint</th><th>Status</th><th>Count</th></tr></thead><tbody id="routes"></tbody></table></div>
</div><div class="panel"><h2>Aggregate prediction classes</h2>
<div class="row"><span>Class 0 · absence</span><strong id="class0">—</strong></div>
<div class="row"><span>Class 1 · presence</span><strong id="class1">—</strong></div>
<div class="row"><span>Failed inference batches</span><strong id="failed">—</strong></div>
<p class="muted">Counts describe API outputs. Model accuracy and data drift require additional data and are not measured here.</p>
</div></section>
<footer><div><button id="pause" type="button">Pause refresh</button> &nbsp; <a href="metrics">JSON metrics</a> · <a href="docs">API documentation</a> · <span id="updated">Not yet updated</span></div>
<p>Single process · counters reset on restart · no patient records stored in metrics.
Health probes, dashboard visits and metrics reads are excluded from request totals.
Response time ends when response headers are available; inference time covers model scoring and output validation.
Logs are written to server stdout. The latest traffic bucket is partial.</p></footer>
</main><script>
const el=id=>document.getElementById(id);
const put=(id,value)=>{el(id).textContent=value;};
const number=value=>Number(value).toLocaleString();
const ms=value=>(value*1000).toFixed(1)+' ms';
let paused=false;
el('pause').addEventListener('click',()=>{paused=!paused;put('pause',paused?'Resume refresh':'Pause refresh');
 if(paused){put('state','Refresh paused');el('state').className='badge bad';}else{refresh();}});
function draw(buckets){
 const svg=el('traffic');svg.replaceChildren();
 const ns='http://www.w3.org/2000/svg';
 const maximum=Math.max(1,...buckets.map(b=>b.requests));
 const offset=60-buckets.length;
 for(let i=0;i<60;i++){
  const value=i<offset?0:buckets[i-offset].requests;
  const rect=document.createElementNS(ns,'rect');
  rect.setAttribute('x',i*10+1);rect.setAttribute('y',135-value/maximum*120);
  rect.setAttribute('width','8');rect.setAttribute('height',Math.max(1,value/maximum*120));
  rect.setAttribute('fill',value?'#238187':'#e1ebec');
  const title=document.createElementNS(ns,'title');title.textContent=value+' requests';rect.append(title);svg.append(rect);
 }
}
async function refresh(){
 if(paused)return;
 try{
  const response=await fetch('metrics',{cache:'no-store',signal:AbortSignal.timeout(5000)});
  if(!response.ok)throw new Error('HTTP '+response.status);
  const m=await response.json(),r=m.requests,p=m.predictions;
  el('error').hidden=true;put('state',m.model.ready?'Model ready':'Model unavailable');
  el('state').className=m.model.ready?'badge':'badge bad';
  put('requests',number(r.total));put('latency',ms(r.duration_seconds_mean));put('errors',number(r.server_errors));
  put('rejected',number(r.client_errors)+' client errors · 4xx');put('records',number(p.records));
  put('batches',number(p.successful_batches)+' successful batches');put('model',m.model.run_id||'No model');
  put('uptime',Math.floor(m.uptime_seconds/60)+'m '+Math.floor(m.uptime_seconds%60)+'s');
  put('active',r.in_flight);put('inference',ms(p.inference_seconds_mean));put('pid',m.process_id);
  put('class0',number(p.by_class['0']));put('class1',number(p.by_class['1']));put('failed',number(p.failed_batches));
  const rows=el('routes');rows.replaceChildren();
  for(const item of r.by_route){const row=document.createElement('tr');
   for(const value of [item.method,item.route,item.status,number(item.count)]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}rows.append(row);}
  if(!r.by_route.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=4;cell.textContent='No application requests yet';row.append(cell);rows.append(row);}
  const b=m.recent.buckets;draw(b);
  put('traffic-summary',number(b.reduce((s,v)=>s+v.requests,0))+' requests in the displayed window · peak '+Math.max(0,...b.map(v=>v.requests))+' / 5s');
  put('updated','Updated '+new Date(m.observed_at_utc).toLocaleTimeString());
 }catch(error){el('error').hidden=false;put('error','Metrics unavailable. Last values may be stale. '+error.message);
  put('state','Disconnected');el('state').className='badge bad';}
}
async function poll(){await refresh();setTimeout(poll,3000);}poll();
</script></body></html>"""
