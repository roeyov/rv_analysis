#!/usr/bin/env python3
import argparse
import json
import http.server
import os
import re
import socketserver
from pathlib import Path
from typing import Dict, List, Tuple

INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase Residuals Browser</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{
    --bg:#0b0f14; --panel:#111827; --muted:#94a3b8; --fg:#e5e7eb; --accent:#60a5fa; --border:#1f2937;
  }
  *{box-sizing:border-box}
  html,body{height:100%; margin:0; background:var(--bg); color:var(--fg); font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Arial,sans-serif}
  .app{display:grid; grid-template-columns: 280px 1fr; min-height:100vh}
  .side{background:var(--panel); border-right:1px solid var(--border); padding:16px; display:flex; flex-direction:column; gap:12px}
  .title{font-size:18px; font-weight:700; margin:0}
  .muted{color:var(--muted); font-size:12px}
  .select{width:100%; padding:8px 10px; border-radius:10px; border:1px solid var(--border); background:#0f172a; color:var(--fg)}
  .btn{padding:6px 10px; border:1px solid var(--border); border-radius:8px; background:#0f172a; color:var(--fg); font-size:12px; cursor:pointer}
  .btn:hover{border-color:var(--accent); color:var(--accent)}
  .row{display:flex; gap:8px; align-items:center}
  .main{display:flex; flex-direction:column; padding:12px; gap:12px}
  .toolbar{display:flex; align-items:center; gap:10px; border:1px solid var(--border); background:#0f172a; border-radius:10px; padding:10px}
  .spacer{flex:1}
  .pill{font-size:12px; padding:4px 8px; border-radius:999px; border:1px solid var(--border); background:#0b1220}
  iframe{width:100%; height:78vh; border:0; background:#0b1220; border-radius:10px; border:1px solid var(--border)}
  a.link{color:var(--accent); text-decoration:none}
  a.link:hover{text-decoration:underline}
  kbd{background:#0f172a;border:1px solid var(--border);border-bottom-width:2px;border-radius:6px;padding:2px 6px;font-size:12px}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <h1 class="title">Phase Residuals</h1>
    <div class="muted">Use <kbd>↑</kbd>/<kbd>↓</kbd> to change star, <kbd>←</kbd>/<kbd>→</kbd> to change solution.</div>

    <label class="muted">Star</label>
    <select id="starSelect" class="select"></select>

    <div class="row">
      <button id="prevStar" class="btn">Prev star ↑</button>
      <button id="nextStar" class="btn">Next star ↓</button>
    </div>

    <label class="muted">Solution (P, e)</label>
    <select id="solSelect" class="select"></select>

    <div class="row">
      <button id="prevSol" class="btn">Prev ←</button>
      <button id="nextSol" class="btn">Next →</button>
    </div>

    <div class="muted">Open current in a new tab:</div>
    <a id="openRaw" class="btn" href="#" target="_blank" rel="noreferrer">Open HTML ↗</a>
  </aside>

  <main class="main">
    <div class="toolbar">
      <div id="infoStar" class="pill">—</div>
      <div id="infoSol" class="pill">—</div>
      <div class="spacer"></div>
      <div class="muted"><strong>Total stars:</strong> <span id="countStars">0</span> • <strong>Solutions:</strong> <span id="countSols">0</span></div>
    </div>

    <iframe id="viewer" title="Phase residuals"></iframe>
  </main>
</div>

<script>
  const DATA = __DATA_JSON__;   // { stars: [{name, solutions:[{p,e,href,label}]}], base: "." }

  // UI elements
  const starSelect = document.getElementById('starSelect');
  const solSelect  = document.getElementById('solSelect');
  const prevStar   = document.getElementById('prevStar');
  const nextStar   = document.getElementById('nextStar');
  const prevSol    = document.getElementById('prevSol');
  const nextSol    = document.getElementById('nextSol');
  const openRaw    = document.getElementById('openRaw');
  const infoStar   = document.getElementById('infoStar');
  const infoSol    = document.getElementById('infoSol');
  const countStars = document.getElementById('countStars');
  const countSols  = document.getElementById('countSols');
  const viewer     = document.getElementById('viewer');

  let state = {
    stars: DATA.stars,
    starIdx: 0,
    solIdx: 0
  };

  // restore last selection
  const saved = JSON.parse(localStorage.getItem('phaseBrowserState') || 'null');
  if (saved){
    state.starIdx = Math.min(Math.max(saved.starIdx, 0), Math.max(DATA.stars.length-1, 0));
    state.solIdx  = Math.max(0, saved.solIdx|0);
  }

  function clampSol(){
    const sols = state.stars[state.starIdx]?.solutions || [];
    if (!sols.length){ state.solIdx = 0; return; }
    if (state.solIdx < 0) state.solIdx = sols.length - 1;
    if (state.solIdx >= sols.length) state.solIdx = 0;
  }

  function saveState(){
    localStorage.setItem('phaseBrowserState', JSON.stringify({starIdx: state.starIdx, solIdx: state.solIdx}));
  }

  function populateStars(){
    starSelect.innerHTML = '';
    state.stars.forEach((s, i)=>{
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = s.name;
      starSelect.appendChild(opt);
    });
    starSelect.value = String(state.starIdx);
    countStars.textContent = state.stars.length;
  }

  function populateSolutions(){
    solSelect.innerHTML = '';
    const sols = state.stars[state.starIdx]?.solutions || [];
    sols.forEach((sol, i)=>{
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = sol.label;
      solSelect.appendChild(opt);
    });
    clampSol();
    solSelect.value = String(state.solIdx);
    countSols.textContent = sols.length;
  }

  function updateView(){
    const star = state.stars[state.starIdx];
    const sols = star?.solutions || [];
    clampSol();
    const sol = sols[state.solIdx];

    infoStar.textContent = star ? `★ ${star.name}` : '—';
    infoSol.textContent  = sol ? `P=${sol.p}, e=${sol.e} (${state.solIdx+1}/${sols.length})` : '—';

    if (sol && sol.href){
      const url = DATA.base + '/' + sol.href;
      viewer.src = url;
      openRaw.href = url;
    } else {
      viewer.src = 'about:blank';
      openRaw.removeAttribute('href');
    }
  }

  function selectStar(idx){
    const n = state.stars.length;
    if (!n) return;
    state.starIdx = (idx + n) % n;
    state.solIdx = 0; // reset to first solution for new star
    populateStars();
    populateSolutions();
    updateView();
    saveState();
  }

  function selectSol(idx){
    const sols = state.stars[state.starIdx]?.solutions || [];
    if (!sols.length) return;
    state.solIdx = (idx + sols.length) % sols.length;
    populateSolutions();
    updateView();
    saveState();
  }

  // Initialize
  populateStars();
  populateSolutions();
  updateView();

  // Events
  starSelect.onchange = ()=> selectStar(parseInt(starSelect.value, 10) || 0);
  solSelect.onchange  = ()=> selectSol(parseInt(solSelect.value, 10) || 0);

  prevStar.onclick = ()=> selectStar(state.starIdx - 1);
  nextStar.onclick = ()=> selectStar(state.starIdx + 1);
  prevSol.onclick  = ()=> selectSol(state.solIdx - 1);
  nextSol.onclick  = ()=> selectSol(state.solIdx + 1);

  // Keyboard navigation
  window.addEventListener('keydown', (e)=>{
    if (e.key === 'ArrowUp')    { e.preventDefault(); selectStar(state.starIdx - 1); }
    if (e.key === 'ArrowDown')  { e.preventDefault(); selectStar(state.starIdx + 1); }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); selectSol(state.solIdx - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); selectSol(state.solIdx + 1); }
  });
</script>
</body>
</html>
"""

FNAME_RE = re.compile(
    r'^(?P<star>.+?)_sid-(?P<sid>\d+)_phase_residuals\.html$'
)


def scan_input(root: Path) -> Dict:
    """
    Return JSON-friendly structure:
    {
      "base": ".",
      "stars": [
        {"name": "StarA", "solutions": [{"sid":0,"href":"StarA/StarA_sid-0_phase_residuals.html","label":"sid=1"}, ...]},
        ...
      ]
    }
    """
    stars: Dict[str, List[Tuple[int, Path]]] = {}

    # Expect subdirs per star
    if not root.is_dir():
        return {"base": ".", "stars": []}

    for star_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        star_name = star_dir.name
        triples: List[Tuple[int, Path]] = []
        for html_file in star_dir.glob("*.html"):
            m = FNAME_RE.match(html_file.name)
            if not m:
                continue
            if m.group("star") != star_name:
                # Only accept files that match the enclosing dir's name
                continue
            try:
                sid = int(m.group("sid"))
            except ValueError:
                continue
            triples.append((sid, html_file))
        if triples:
            # sort by P then e
            triples.sort(key=lambda t: t[0])
            stars[star_name] = triples

    star_entries = []
    for name in sorted(stars.keys()):
        sols = []
        for sid, path in stars[name]:
            # href should be relative to the index location (we will write index.html at root)
            href = f"{name}/{path.name}"
            # pretty formatting (don’t change the file’s precision, only the label)
            label = f"sid={sid:.6g}"
            sols.append({"sid": f"{sid}", "href": href, "label": label})
        star_entries.append({"name": name, "solutions": sols})

    return {"base": ".", "stars": star_entries}

def write_index(root: Path, data: Dict) -> Path:
    index_path = root / "index.html"
    html = INDEX_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    index_path.write_text(html, encoding="utf-8")
    return index_path

def run_server(root_dir: Path, port: int):
    os.chdir(root_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing {root_dir} at http://localhost:{port}/index.html")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")

def main():
    ap = argparse.ArgumentParser(description="Browse *_phase_residuals.html files by star and (P,e).")
    ap.add_argument("input_dir", help="Path to input dir containing per-star subdirectories")
    ap.add_argument("--serve", action="store_true", help="Serve via a simple HTTP server")
    ap.add_argument("--port", type=int, default=8000, help="Port for --serve (default 8000)")
    args = ap.parse_args()

    root = Path(args.input_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    data = scan_input(root)
    if not data["stars"]:
        print("No matching *_phase_residuals.html files found.")
    index_path = write_index(root, data)
    print("Wrote:", index_path)

    if args.serve:
        run_server(root, args.port)
    else:
        print(f"\nOpen in your browser:\n  {index_path}\n")
        print("Tip: to avoid file:// iframe restrictions, run:")
        print(f"  python3 -m http.server -d {root} 8000")
        print("  -> then open:  http://localhost:8000/index.html")

if __name__ == "__main__":
    main()
