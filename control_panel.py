#!/usr/bin/env python3
"""One-page web control panel for the SO-101 -> YAM teleop rig + camera dashboard.

Auto-discovers leaders (by controller serial), YAM CAN channels, and RealSense
cameras, then gives you buttons:
  * Connect      - start the camera dashboard (Rerun) + refresh hardware status
  * Start Teleop - launch teleop for each configured pair (auto-resolves ports)
  * Stop Teleop / Stop All

The live camera streams are embedded from the Rerun dashboard.

Run:
    cd ~/Mission/i2rt
    .venv/bin/python control_panel.py
Then open  http://localhost:8080

Future stage (placeholder): train ACT policy -> deploy -> visualize.
"""
import glob
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RECORD_BASE = "http://localhost:8090/record"


def rec_get():
    try:
        with urllib.request.urlopen(RECORD_BASE + "/status", timeout=1) as r:
            return json.load(r)
    except Exception:
        return None


def rec_post(action):
    try:
        req = urllib.request.Request(f"{RECORD_BASE}/{action}", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.load(r).get("msg", "ok")
    except Exception as e:
        return f"recorder unreachable ({e}) — is the camera dashboard running?"

REPO = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(REPO, ".venv", "bin", "python")
PANEL_PORT = 8080
RERUN_WEB = 9090
RERUN_GRPC = 9876
RERUN_URL = f"http://localhost:{RERUN_WEB}/?url=rerun+http://127.0.0.1:{RERUN_GRPC}/proxy"

# Which leader controller drives which YAM (CAN channel).
PAIRINGS = [
    {"name": "Arm A", "leader_serial": "5B14115162", "channel": "can1"},
    {"name": "Arm B", "leader_serial": "5AE6080681", "channel": "can0"},
]


# ------------------------------------------------------------------ discovery
def leader_ports():
    """{controller_serial: /dev/ttyACMx} for every connected leader."""
    byid = "/dev/serial/by-id"
    out = {}
    if os.path.isdir(byid):
        for name in os.listdir(byid):
            m = re.search(r"Serial_([A-Za-z0-9]+)", name)
            if m:
                out[m.group(1)] = os.path.realpath(os.path.join(byid, name))
    return out


def cameras():
    try:
        import pyrealsense2 as rs
        return [
            (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
            for d in rs.context().query_devices()
        ]
    except Exception:
        return []


def can_up(ch):
    try:
        out = subprocess.run(["ip", "-br", "link", "show", ch],
                             capture_output=True, text=True, timeout=3).stdout
        return "UP" in out
    except Exception:
        return False


def procs_matching(substr):
    """(pid, cmdline) for python processes whose cmdline contains substr."""
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        try:
            if not open(f"{p}/comm").read().strip().startswith("python"):
                continue
            cl = open(f"{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            if substr in cl:
                out.append((int(os.path.basename(p)), cl))
        except Exception:
            pass
    return out


def teleop_running_channels():
    chans = set()
    for _, cl in procs_matching("so101_teleop"):
        m = re.search(r"channel (can\d)", cl)
        if m:
            chans.add(m.group(1))
    return chans


# ------------------------------------------------------------------ actions
def start_cameras():
    # Always restart so newly-plugged / re-enumerated cameras get picked up
    # (camera_dashboard discovers cameras once at startup).
    _kill("camera_dashboard.py")
    time.sleep(1)
    subprocess.Popen(
        [PY, os.path.join(REPO, "camera_dashboard.py"),
         "--width", "424", "--height", "240", "--fps", "15"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    n = len(cameras())
    return f"restarting cameras ({n} detected) — stream reloads in ~10s"


def start_teleop():
    ports = leader_ports()
    running = teleop_running_channels()
    msgs = []
    for pair in PAIRINGS:
        ch, serial, name = pair["channel"], pair["leader_serial"], pair["name"]
        if ch in running:
            msgs.append(f"{name}: already running on {ch}")
            continue
        if serial not in ports:
            msgs.append(f"{name}: leader {serial} NOT connected — skipped")
            continue
        if not can_up(ch):
            msgs.append(f"{name}: {ch} is down — skipped")
            continue
        subprocess.Popen(
            [PY, os.path.join(REPO, "so101_teleop.py"),
             "--port", ports[serial], "--channel", ch, "--seconds", "0", "--hz", "60"],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        msgs.append(f"{name}: started {ports[serial]} -> {ch}")
    return " | ".join(msgs)


def _kill(substr):
    killed = 0
    for pid, _ in procs_matching(substr):
        try:
            os.kill(pid, signal.SIGINT)
            killed += 1
        except Exception:
            pass
    time.sleep(2)
    for pid, _ in procs_matching(substr):
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return killed


def stop_teleop():
    return f"stopped {_kill('so101_teleop')} teleop process(es)"


def stop_all():
    t = _kill("so101_teleop")
    c = _kill("camera_dashboard.py")
    return f"stopped {t} teleop + {c} camera process(es)"


# ------------------------------------------------------------------ status
def status():
    ports = leader_ports()
    running = teleop_running_channels()
    cams = cameras()
    pairs = []
    for pair in PAIRINGS:
        serial = pair["leader_serial"]
        pairs.append({
            "name": pair["name"],
            "leader_serial": serial,
            "leader_connected": serial in ports,
            "leader_port": ports.get(serial, "-"),
            "channel": pair["channel"],
            "can_up": can_up(pair["channel"]),
            "teleop_running": pair["channel"] in running,
        })
    return {
        "pairs": pairs,
        "cameras": [{"name": n, "serial": s} for n, s in cams],
        "cameras_streaming": bool(procs_matching("camera_dashboard.py")),
        "rerun_url": RERUN_URL,
        "record": rec_get(),
    }


# ------------------------------------------------------------------ web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Teleop Control</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:14px 20px;background:#171a21;border-bottom:1px solid #2a2f3a;display:flex;
   align-items:center;gap:16px;flex-wrap:wrap}
 h1{font-size:18px;margin:0}
 button{font-size:14px;padding:9px 16px;border-radius:8px;border:1px solid #2a2f3a;
   background:#222835;color:#e6e6e6;cursor:pointer}
 button:hover{background:#2c3444}
 .go{background:#1f6f43;border-color:#2c8f59}.go:hover{background:#268a53}
 .stop{background:#7a2230;border-color:#a13141}.stop:hover{background:#93293a}
 input{font-size:14px;padding:8px 10px;border-radius:8px;border:1px solid #2a2f3a;background:#0f1115;color:#e6e6e6;width:150px}
 #msg{margin-left:auto;font-size:13px;color:#9fb0c3;max-width:40ch}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .card{background:#171a21;border:1px solid #2a2f3a;border-radius:12px;padding:14px 16px}
 .status{min-width:320px}
 table{border-collapse:collapse;width:100%}
 td,th{padding:6px 10px;text-align:left;font-size:13px;border-bottom:1px solid #222835}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
 .ok{background:#37c871}.bad{background:#e0555f}.idle{background:#c9a227}
 .streams{flex:1;min-width:520px}
 iframe{width:100%;height:70vh;border:1px solid #2a2f3a;border-radius:10px;background:#000}
 a.link{color:#6db3ff}
 .sub{font-size:12px;color:#7d8aa0}
</style></head><body>
<header>
 <h1>🦾 Teleop Control</h1>
 <button class="go" onclick="connectCams()">Connect (cameras)</button>
 <button class="go" onclick="startTeleop()">Start Teleop</button>
 <button class="stop" onclick="act('teleop/stop')">Stop Teleop</button>
 <button class="stop" onclick="act('stop_all')">Stop All</button>
 <span id="msg"></span>
</header>
<div class="wrap">
 <div class="card status">
   <div style="font-weight:600;margin-bottom:8px">Hardware</div>
   <table id="pairs"></table>
   <div style="font-weight:600;margin:14px 0 6px">Cameras</div>
   <table id="cams"></table>
   <div style="font-weight:600;margin:16px 0 6px">Dataset</div>
   <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
     <input id="dsname" placeholder="e.g. test1" onkeydown="if(event.key==='Enter')setDataset()">
     <button class="go" onclick="setDataset()">Create / Select</button>
   </div>
   <div id="dslist" class="sub" style="margin-top:6px">—</div>
   <div style="font-weight:600;margin:16px 0 6px">Episode recording</div>
   <div id="recstatus" class="sub">—</div>
   <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
     <button class="go" onclick="act('record/start')">● Start Episode</button>
     <button onclick="act('record/stop')">Stop</button>
     <button class="go" onclick="act('record/save')">Save</button>
     <button class="stop" onclick="act('record/discard')">Discard</button>
   </div>
   <div class="sub" style="margin-top:12px">Stage 2 (soon): train ACT · deploy · visualize</div>
 </div>
 <div class="card streams">
   <div style="font-weight:600;margin-bottom:8px">Camera streams
     &nbsp;<a class="link" id="openrr" href="#" target="_blank">open in new tab ↗</a></div>
   <iframe id="rr" src="about:blank"></iframe>
 </div>
</div>
<script>
function dot(state){return '<span class="dot '+state+'"></span>'}
async function refresh(){
  const s = await (await fetch('/api/status')).json();
  document.getElementById('openrr').href = s.rerun_url;
  const rr = document.getElementById('rr');
  if(s.cameras_streaming && rr.src.indexOf('rerun')<0) rr.src = s.rerun_url;
  if(!s.cameras_streaming) rr.src = 'about:blank';
  let ph = '<tr><th>Pair</th><th>Leader</th><th>YAM</th><th>Teleop</th></tr>';
  for(const p of s.pairs){
    ph += '<tr><td>'+p.name+'</td>'
      + '<td>'+dot(p.leader_connected?'ok':'bad')+(p.leader_connected?p.leader_port.split('/').pop():'—')+'</td>'
      + '<td>'+dot(p.can_up?'ok':'bad')+p.channel+'</td>'
      + '<td>'+dot(p.teleop_running?'ok':'idle')+(p.teleop_running?'live':'off')+'</td></tr>';
  }
  document.getElementById('pairs').innerHTML = ph;
  let ch = '<tr><th>Camera</th><th>Serial</th></tr>';
  for(const c of s.cameras) ch += '<tr><td>'+dot('ok')+c.name.replace('Intel RealSense ','')+'</td><td>'+c.serial+'</td></tr>';
  if(s.cameras.length===0) ch += '<tr><td colspan=2>'+dot('bad')+'none detected</td></tr>';
  document.getElementById('cams').innerHTML = ch;
  const rec = s.record;
  let rs;
  if(!rec){
    rs = dot('bad')+'recorder offline (start cameras)';
    document.getElementById('dslist').textContent = 'start cameras to manage datasets';
  } else {
    const cts = Object.entries(rec.counts).map(([k,v])=>k+'='+v).join(' ') || '—';
    rs = dot(rec.recording?'ok':'idle') + (rec.recording?'● REC · ':'idle · ')
       + rec.n_frames + ' frames [' + cts + '] · this dataset has ' + rec.episodes_on_disk + ' saved';
    const dl = (rec.datasets||[]).map(d=>
      '<a href="#" class="link" onclick="useDataset(' + "'" + d.name + "'" + ');return false">'+d.name+'</a> ('+d.episodes+')'
    ).join(' · ') || 'none yet';
    document.getElementById('dslist').innerHTML = 'active: <b>'+rec.dataset+'</b> &nbsp;|&nbsp; resume: '+dl;
    document.getElementById('dsname').placeholder = rec.dataset;
  }
  document.getElementById('recstatus').innerHTML = rs;
}
function setDataset(){
  const n = document.getElementById('dsname').value.trim();
  if(n){ act('record/dataset?name='+encodeURIComponent(n)); document.getElementById('dsname').value=''; }
}
function useDataset(n){ act('record/dataset?name='+encodeURIComponent(n)); }
async function act(path){
  document.getElementById('msg').textContent = '...';
  const r = await (await fetch('/api/'+path,{method:'POST'})).json();
  document.getElementById('msg').textContent = r.msg;
  setTimeout(refresh, 600);
}
function startTeleop(){
  if(confirm('This MOVES the arms. Is the workspace clear and hand near e-stop?')) act('teleop/start');
}
async function connectCams(){
  const rr = document.getElementById('rr');
  rr.src = 'about:blank';                 // drop the old (dead) stream
  await act('connect');                   // restart camera_dashboard
  setTimeout(async () => {                // reconnect once it's back up
    const s = await (await fetch('/api/status')).json();
    rr.src = s.rerun_url;
  }, 11000);
}
refresh(); setInterval(refresh, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json(status())
        else:
            self.send_error(404)

    def do_POST(self):
        # record calls forward straight through to the capture service (keep query string)
        if self.path.startswith("/api/record/"):
            self._json({"msg": rec_post(self.path[len("/api/record/"):])})
            return
        routes = {
            "/api/connect": lambda: start_cameras(),
            "/api/teleop/start": lambda: start_teleop(),
            "/api/teleop/stop": lambda: stop_teleop(),
            "/api/stop_all": lambda: stop_all(),
        }
        fn = routes.get(self.path)
        if not fn:
            self.send_error(404)
            return
        try:
            msg = fn()
        except Exception as e:
            msg = f"error: {e}"
        self._json({"msg": msg})


def main():
    print(f"Control panel: http://localhost:{PANEL_PORT}")
    ThreadingHTTPServer(("0.0.0.0", PANEL_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
