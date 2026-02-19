
#!/usr/bin/env python3
import argparse
import json
import http.server
import os
import socketserver
from pathlib import Path

INDEX_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Star Visuals</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{
    --bg:#0b0f14; --panel:#111827; --muted:#94a3b8; --fg:#e5e7eb; --accent:#60a5fa; --border:#1f2937;
    --paneA:50%; --paneB:50%;  --workspaceH: 66vh;
  }
  *{box-sizing:border-box}
  html,body{height:100%; margin:0; background:var(--bg); color:var(--fg); font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Arial,sans-serif}
  .app{display:grid; grid-template-columns: 280px 1fr; min-height:100vh; overflow:visible}
  /* Sidebar */
  .side{
      background:var(--panel);
      border-right:1px solid var(--border);
      display:flex;
      flex-direction:column;
      gap:12px;
      padding:16px;
      height:100vh;
      overflow:hidden;
  }
  .title{font-size:16px; font-weight:600; letter-spacing:.3px; margin:0 0 4px}
  .search{width:100%; padding:10px 12px; border-radius:10px; border:1px solid var(--border); background:#0f172a; color:var(--fg)}
  .list{
      flex:1;
      overflow-y:auto;
      border:1px solid var(--border);
      border-radius:10px;
      background:#0f172a;
      min-height:0;
  }
  .item{padding:10px 12px; cursor:pointer; border-bottom:1px solid #0b1220}
  .item:last-child{border-bottom:0}
  .item:hover{background:#0b1220}
  .item.active{background:#122033; color:#dbeafe}

  /* Main */
  .main{display:flex; flex-direction:column; height:auto; overflow:visible}
  .toolbar{display:flex; align-items:center; gap:8px; padding:10px 14px; border-bottom:1px solid var(--border); background:var(--panel)}
  .btn{padding:6px 10px; border:1px solid var(--border); border-radius:8px; background:#0f172a; color:var(--fg); font-size:12px; text-decoration:none; cursor:pointer}
  .btn:hover{border-color:var(--accent); color:var(--accent)}
  .spacer{flex:1}
  .paths{font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:45vw}
    
  /* Workspace (top visuals) */
  .workspace{position:relative; flex:1 0 auto; display:grid; height: var(--workspaceH); overflow:hidden}
  .workspace.stacked{grid-template-rows: var(--paneA) 6px var(--paneB); grid-template-columns: 100%}
  .workspace.side{grid-template-columns: var(--paneA) 6px var(--paneB); grid-template-rows: 100%}

  iframe{width:100%; height:100%; border:0; background:#0b1220; border-radius:8px; border:1px solid var(--border)}
  .splitter{background:#0b1220; border:1px solid var(--border); border-radius:6px; cursor:ns-resize}
  .workspace.side .splitter{cursor:ew-resize}
  .frames{display:contents}

  /* Results section */
  .results{ flex: 1 1 auto; display:flex; flex-direction:column; min-height:0; }
  .results-toolbar{display:flex; align-items:center; gap:10px; padding:10px 14px; border-bottom:1px solid var(--border)}
  .col-toggles{display:flex; gap:8px; flex-wrap:wrap}
  .chk{display:flex; align-items:center; gap:6px; background:#0f172a; border:1px solid var(--border); border-radius:8px; padding:4px 8px; font-size:12px}
  .res-body{ flex: 1 1 auto; display:flex; flex-direction:column; min-height:0; }
  .table-wrap{overflow:auto; border:1px solid var(--border); border-radius:10px; background:#0f172a; width:100%; margin-bottom:10px;}
  table{width:100%; border-collapse:collapse; font-size:12px}
  th, td{padding:8px 10px; border-bottom:1px solid #0b1220; white-space:nowrap}
  th{position:sticky; top:0; background:#0d1526; z-index:1}
  tr:hover td{background:#0b1220}
  td.sel{background:#13233b}
  .actions{display:flex; align-items:center; gap:10px}
  .two-up{ flex: 1 1 auto; display:grid; grid-template-columns: 7fr 3fr; gap:10px; min-height:0; }
  .stack{display:grid; grid-template-rows: 1fr 1fr; gap:10px; height:100%}
  /* Panels fill their grid cells; internal scrolling for long content */
  .panel{ display:flex; flex-direction:column; min-height:0; }
  .panel-header{padding:8px 10px; border-bottom:1px solid #0b1220; font-size:12px; color:#000000} /* choose color you prefer */
  .panel-body{ flex: 1 1 auto; min-height:180vh; overflow:visible; }

  /* Optional: control the split between time vs phase iframes */
  .stack{
    display: grid;
    grid-template-rows: 1fr 1fr;  /* equal split; e.g., 3fr 2fr for 60/40 */
    gap: 10px;
    height: 100%;
  }
  pre{margin:0; padding:12px; white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; color:#e5e7eb}
  /* Responsive tweak: on narrow screens, stack the two panels */
  @media (max-width: 1100px){
    .two-up{ grid-template-columns: 1fr; }
  }
  .warn{color:#fbbf24}
  
  /* Columns popover */
.col-control{ position:relative; }
.popover{
  position:absolute; right:0; top: calc(100% + 8px);
  width: 420px; max-width: 90vw;
  background:#0f172a; border:1px solid var(--border); border-radius:10px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
  opacity:0; pointer-events:none; transform: translateY(-6px);
  transition: opacity .12s ease, transform .12s ease;
  z-index: 50;
}
.popover.open{ opacity:1; pointer-events:auto; transform: translateY(0); }
.pop-head{ display:flex; align-items:center; gap:8px; padding:10px; border-bottom:1px solid #0b1220; }
.pop-head .search{ flex:1; }
.pop-actions{ display:flex; gap:8px; }
.pop-body{ padding:8px; max-height: 50vh; overflow:auto; display:flex; flex-wrap:wrap; gap:6px; }
.pop-body .chk{ margin:0; }
/* Periodogram pair */
/* Periodogram pair */
.periodos{
  display:grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap:10px;
  margin:10px 0;
}
.periodos .panel-body{
  min-height: 0;                  /* override your global 180vh */
  height: clamp(220px, 70vh, 420px);
  overflow: hidden;               /* or auto if you want inner scroll */
}

/* If you *never* want stacking on small screens, remove this media rule.
   If you DO want stacking on narrow screens, keep it: */
@media (max-width:1100px){
  .periodos{ grid-template-columns: 1fr; }
  .periodos .panel-body{ height: clamp(200px, 60vh, 360px); }
}
.table-filter {
  display:flex;
  align-items:center;
  gap:0.25rem;
  margin-left:0.5rem;
}

.table-filter select {
  max-width:180px;
}

.table-filter .search {
  max-width:180px;
}


</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <h1 class="title">Star Visuals</h1>
    <input id="search" class="search" placeholder="Search stars..." />
    <div id="list" class="list"></div>
  </aside>

  <main class="main">
    <div class="toolbar">
      <button class="btn" id="modeStack">Stacked</button>
      <button class="btn" id="modeSide">Side by side</button>
      <div class="spacer"></div>
      <a class="btn" id="openOverlay" target="_blank" rel="noreferrer">Open overlay ↗</a>
      <a class="btn" id="openCombined" target="_blank" rel="noreferrer">Open RVs ↗</a>
      <span class="paths" id="pathInfo"></span>
    </div>

    <!-- Top visuals -->
    <section id="workspace" class="workspace stacked">
      <div class="frames" id="frameA">
        <iframe id="overlayFrame" title="All Spectra Min/Max Overlay" allow="fullscreen"></iframe>
      </div>
      <div class="splitter" id="splitter"></div>
      <div class="frames" id="frameB">
        <iframe id="combinedFrame" title="RV vs MJD Combined" allow="fullscreen"></iframe>
      </div>
    </section>
    <div class="periodos">
      <div class="panel">
        <div class="panel-header">PDC Periodogram</div>
        <div class="panel-body">
          <iframe id="pdcFrame" title="PDC periodogram"></iframe>
        </div>
      </div>
    
      <div class="panel">
        <div class="panel-header">Lomb–Scargle Periodogram</div>
        <div class="panel-body">
          <iframe id="lsFrame" title="LS periodogram"></iframe>
        </div>
      </div>
    
      <div class="panel">
        <div class="panel-header">PDC WA Periodogram</div>
        <div class="panel-body">
          <iframe id="pdcWAFrame" title="PDC WA periodogram"></iframe>
        </div>
      </div>
    
      <div class="panel">
        <div class="panel-header">Lomb–Scargle WA Periodogram</div>
        <div class="panel-body">
          <iframe id="lsWAFrame" title="LS WA periodogram"></iframe>
        </div>
      </div>
    </div>


    <!-- Results section -->
    <section class="results">
      <div class="results-toolbar">
        <div class="actions">
          <button class="btn" id="showSolution">Show solution</button>
          <span id="csvStatus" class="paths"></span>
        </div>
        <div class="spacer"></div>
        <div class="col-control">
          <button class="btn" id="colsBtn" aria-haspopup="true" aria-expanded="false">Columns ▾</button>
          <div class="popover" id="colsPopover" role="dialog" aria-label="Choose table columns">
            <div class="pop-head">
              <input id="colsSearch" class="search" placeholder="Filter columns..." />
              <div class="pop-actions">
                <a href="#" class="btn" id="colsSelectAll">Select all</a>
                <a href="#" class="btn" id="colsReset">Reset</a>
              </div>
            </div>
            <div class="pop-body" id="colsBody"><!-- checkboxes go here --></div>
          </div>
        </div>
      </div>
      <div class="table-filter">
        <select id="filterCol"></select>
        <input id="filterValue" class="search" placeholder="Filter values…" />
        <button class="btn" id="clearFilter" title="Clear filter">✕</button>
      </div>
      <div class="res-body">
      

        <div class="table-wrap">
          <table id="csvTable">
            <thead></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="two-up">
          <div class="panel">
            <div class="panel-header" id="leftHeader">Time & Phase residuals</div>
            <div class="panel-body">
              <div class="stack">
                <iframe id="timeFrame" title="Time residuals"></iframe>
                <iframe id="phaseFrame" title="Phase residuals"></iframe>
              </div>
            </div>
          </div>
          <div class="panel">
            <div class="panel-header" id="rightHeader">Report.txt</div>
            <div class="panel-body">
              <pre id="reportView">(Select a row and click “Show solution”)</pre>
            </div>
          </div>
        </div>
      </div>
    </section>

  </main>
</div>

<script>
  const STARS = __STARS_JSON__;
  const BASE  = "__BASE_URL__";
  const RESBASE = "__RESULTS_BASE__";

  const searchEl = document.getElementById('search');
  const listEl = document.getElementById('list');
  const overlayFrame = document.getElementById('overlayFrame');
  const combinedFrame = document.getElementById('combinedFrame');
  const pathInfo = document.getElementById('pathInfo');
  const openOverlay = document.getElementById('openOverlay');
  const openCombined = document.getElementById('openCombined');
  const workspace = document.getElementById('workspace');
  const splitter = document.getElementById('splitter');
  const modeStack = document.getElementById('modeStack');
  const modeSide  = document.getElementById('modeSide');

  // Results panel
  const csvTable = document.getElementById('csvTable');
  const thead = csvTable.querySelector('thead');
  const tbody = csvTable.querySelector('tbody');
  const colToggles = document.getElementById('colToggles');
  const showBtn = document.getElementById('showSolution');
  const csvStatus = document.getElementById('csvStatus');
  const timeFrame = document.getElementById('timeFrame');
  const phaseFrame = document.getElementById('phaseFrame');
  const reportView = document.getElementById('reportView');
  const leftHeader = document.getElementById('leftHeader');
  const rightHeader = document.getElementById('rightHeader');
  const colsBtn = document.getElementById('colsBtn');
  const colsPopover = document.getElementById('colsPopover');
  const colsBody = document.getElementById('colsBody');
  const colsSearch = document.getElementById('colsSearch');
  const colsSelectAll = document.getElementById('colsSelectAll');
  const colsReset = document.getElementById('colsReset');
  const filterCol = document.getElementById('filterCol');
  const filterValue = document.getElementById('filterValue');
  const clearFilter = document.getElementById('clearFilter');

  const pdcFrame = document.getElementById('pdcFrame');
  const lsFrame  = document.getElementById('lsFrame');
  const pdcWAFrame = document.getElementById('pdcWAFrame');
  const lsWAFrame  = document.getElementById('lsWAFrame');


let state = {
  stars: STARS,
  filtered: STARS,
  active: localStorage.getItem('activeStar') || (STARS[0] ? STARS[0].star : null),
  layout: localStorage.getItem('layout') || 'stacked',
  paneA: parseFloat(localStorage.getItem('paneA')) || 50,
  paneB: parseFloat(localStorage.getItem('paneB')) || 50,
  // CSV data per star
  tableCols: [],
  tableRows: [],
  viewRows: [],        // <- visible (filtered/sorted) rows
  selectedRowIndex: null,
  periodCol: null,
  eccCol: null,
  // filtering / sorting
  filterCol: null,
  filterValue: '',
  sortCol: null,
  sortDir: 'asc'
};

  function parseNumber(value) {
    if (value == null) return NaN;
    const s = String(value).replace(',', '.'); // handle decimal comma
    const m = s.match(/[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/);
    return m ? parseFloat(m[0]) : NaN;
  }

  function setPanes(aPct){
    aPct = Math.max(10, Math.min(90, aPct));
    state.paneA = aPct; state.paneB = 100 - aPct;
    document.documentElement.style.setProperty('--paneA', state.paneA + '%');
    document.documentElement.style.setProperty('--paneB', state.paneB + '%');
    localStorage.setItem('paneA', state.paneA);
    localStorage.setItem('paneB', state.paneB);
  }

    async function firstOkUrl(urls) {
      for (const url of urls) {
        try {
          const r = await fetch(url, { method: 'HEAD', cache: 'no-store' });
          if (r.ok) return url;
        } catch (_) {}
      }
      return null;
    }
    
    function setIframeOrBlank(frame, url) {
      frame.src = url || 'about:blank';
    }
    
    function candidatePeriodogramUrls(star, kind) {
      // kind: "ls", "pdc", "ls_wa", "pdc_wa"
      const base = `${RESBASE}/${encodeURIComponent(star)}/periodogram/`;
      const s = encodeURIComponent(star);
    
      // generate common case variants
      const LS  = ["LS","ls","Ls","lS"];
      const PDC = ["PDC","pdc","Pdc","pDc","pdc_opt"];
      const WA  = ["WA","wa","Wa","wA"];
    
      const urls = [];
    
      if (kind === "ls") {
        for (const ls of LS) {
          urls.push(`${base}${s}_${ls}_periodogram.html`);
        }
      } else if (kind === "pdc") {
        for (const pdc of PDC) {
          urls.push(`${base}${s}_${pdc}_periodogram.html`);
        }
      } else if (kind === "ls_wa") {
        for (const ls of LS) for (const wa of WA) {
          urls.push(`${base}${s}_${ls}_${wa}_periodogram.html`);
          urls.push(`${base}${s}_${ls}-${wa}_periodogram.html`); // sometimes generators use -
          urls.push(`${base}${s}_${ls}${wa}_periodogram.html`);  // sometimes no separator
          urls.push(`${base}${s}_${ls}_${wa}_Periodogram.html`); // occasional capital P
        }
      } else if (kind === "pdc_wa") {
        for (const pdc of PDC) for (const wa of WA) {
          urls.push(`${base}${s}_${pdc}_${wa}_periodogram.html`);
          urls.push(`${base}${s}_${pdc}-${wa}_periodogram.html`);
          urls.push(`${base}${s}_${pdc}${wa}_periodogram.html`);
          urls.push(`${base}${s}_${pdc}_${wa}_Periodogram.html`);
        }
      }
      return urls;
    }
    
    function candidateCsvUrls(star) {
      const base = `${RESBASE}/${encodeURIComponent(star)}/`;
      // common variants
      return [
        `${base}lmfit_summary.csv`,
        `${base}lmfit_Summary.csv`,
        `${base}LMFIT_summary.csv`,
        `${base}LMFIT_SUMMARY.csv`,
      ];
    }
    
      function renderList(){
        listEl.innerHTML = '';
        if (!state.filtered.length){
          listEl.innerHTML = '<div class="item"><span class="warn">No matches</span></div>';
          return;
        }
        state.filtered.forEach(s=>{
          const el = document.createElement('div');
          el.className = 'item' + (s.star===state.active?' active':'');
          el.textContent = s.star;
          el.onclick = ()=> selectStar(s.star);
          listEl.appendChild(el);
        });
      }

async function selectStar(name){
  state.active = name;
  localStorage.setItem('activeStar', name);
  const s = state.stars.find(x=>x.star===name);
  if (!s){ return; }

  const overlay = BASE + '/' + s.overlay;
  const combined = BASE + '/' + s.combined;
  overlayFrame.src = overlay;
  combinedFrame.src = combined;
  openOverlay.href = overlay;
  openCombined.href = combined;
  pathInfo.textContent = s.overlay + '  |  ' + s.combined;

  renderList();

  // ---- Periodograms: try multiple filename case variants ----
  const pdcUrl   = await firstOkUrl(candidatePeriodogramUrls(name, "pdc"));
  const lsUrl    = await firstOkUrl(candidatePeriodogramUrls(name, "ls"));
  const pdcWAUrl = await firstOkUrl(candidatePeriodogramUrls(name, "pdc_wa"));
  const lsWAUrl  = await firstOkUrl(candidatePeriodogramUrls(name, "ls_wa"));

  setIframeOrBlank(pdcFrame, pdcUrl);
  setIframeOrBlank(lsFrame, lsUrl);
  setIframeOrBlank(pdcWAFrame, pdcWAUrl);
  setIframeOrBlank(lsWAFrame, lsWAUrl);

  // ---- Load CSV for results (case-insensitive candidates) ----
  await loadCsvForStar(name);
}


  function setLayout(mode){
    state.layout = mode;
    localStorage.setItem('layout', mode);
    workspace.classList.toggle('stacked', mode==='stacked');
    workspace.classList.toggle('side', mode==='side');
    splitter.style.cursor = mode==='side' ? 'ew-resize' : 'ns-resize';
  }

  function init(){
    setLayout(state.layout);
    setPanes(state.paneA);
    renderList();
    if (state.active){ selectStar(state.active); }

    searchEl.addEventListener('input', ()=>{
      const q = searchEl.value.toLowerCase().trim();
      state.filtered = state.stars.filter(s=> s.star.toLowerCase().includes(q));
      renderList();
    });

    modeStack.onclick = ()=> setLayout('stacked');
    modeSide.onclick  = ()=> setLayout('side');

    // Splitter drag
    let dragging=false, startPos=0, startA=state.paneA;
    const onDown = (e)=>{
      dragging=true;
      startPos = (state.layout==='side') ? e.clientX : e.clientY;
      startA = state.paneA;
      e.preventDefault();
    };
    const onMove = (e)=>{
      if(!dragging) return;
      const delta = (state.layout==='side') ? e.clientX - startPos : e.clientY - startPos;
      const size = (state.layout==='side') ? workspace.clientWidth : workspace.clientHeight;
      const deltaPct = (delta / size) * 100;
      setPanes(startA + deltaPct);
    };
    const onUp = ()=> dragging=false;
    splitter.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    // Keyboard: splitter
    window.addEventListener('keydown', (e)=>{
      if (e.key==='ArrowUp' || e.key==='ArrowLeft'){ setPanes(state.paneA - 5); }
      if (e.key==='ArrowDown' || e.key==='ArrowRight'){ setPanes(state.paneA + 5); }
    });
    // Columns popover open/close
    const closeCols = ()=>{
      colsPopover.classList.remove('open');
      colsBtn.setAttribute('aria-expanded','false');
    };
    const openCols = ()=>{
      colsPopover.classList.add('open');
      colsBtn.setAttribute('aria-expanded','true');
      colsSearch.focus();
    };
    colsBtn.addEventListener('click', (e)=>{
      e.preventDefault();
      colsPopover.classList.contains('open') ? closeCols() : openCols();
    });
    // click outside to close
    document.addEventListener('click', (e)=>{
      if (!colsPopover.contains(e.target) && e.target !== colsBtn) closeCols();
    });
    // Escape key to close
    window.addEventListener('keydown', (e)=>{
      if (e.key === 'Escape') closeCols();
    });

    // Show solution
    showBtn.addEventListener('click', onShowSolution);
  }
  init();

  // ===== Results CSV handling =====

async function loadCsvForStar(star){
  const url = await firstOkUrl(candidateCsvUrls(star));
  if (!url) {
    state.tableCols = [];
    state.tableRows = [];
    state.viewRows = [];
    thead.innerHTML = "";
    tbody.innerHTML = "";
    csvStatus.textContent = "No CSV found for this star.";
    timeFrame.src = "about:blank";
    phaseFrame.src = "about:blank";
    reportView.textContent = "(No CSV found)";
    return;
  }

  csvStatus.textContent = "Loading CSV...";

  fetch(url)
    .then(r => {
      if(!r.ok) throw new Error("CSV not found");
      return r.text();
    })
    .then(text => {
      const {columns, rows} = parseCSV(text);
      state.tableCols = columns;
      state.tableRows = rows;
      state.viewRows = rows.slice();
      state.selectedRowIndex = null;
      state.filterCol = null;
      state.filterValue = '';
      state.sortCol = null;
      state.sortDir = 'asc';

      state.periodCol = detectColumn(
        columns,
        ['Period_value','period','best_period','P_days','P'],
        'period'
      );
      state.eccCol = detectColumn(
        columns,
        ['Eccentricity_value','eccentricity_value','eccentricity','ecc'],
        'eccentricity'
      );
      state.solidCol = detectColumn(columns, ['solution_id'], 'solution');

      buildColumnToggles(columns);
      buildFilterControls(columns);
      updateView();

      csvStatus.textContent = `${rows.length} rows • ${columns.length} cols`;
      leftHeader.textContent = "Time & Phase residuals";
      rightHeader.textContent = "Report.txt";
      reportView.textContent = "(Select a row and click “Show solution”)";
      timeFrame.src = "about:blank";
      phaseFrame.src = "about:blank";

      const bestCol = detectIsBestColumn(columns);
      if (bestCol) {
        const idx = state.viewRows.findIndex(r => isTruthy(r[bestCol]));
        if (idx !== -1) {
          selectTableRow(idx);
          onShowSolution();
          csvStatus.textContent += ` • auto-selected: row ${idx+1} (${bestCol}=true)`;
        }
      }
    })
    .catch(err => {
      state.tableCols = [];
      state.tableRows = [];
      state.viewRows = [];
      thead.innerHTML = "";
      tbody.innerHTML = "";
      csvStatus.textContent = "No CSV found for this star.";
      timeFrame.src = "about:blank";
      phaseFrame.src = "about:blank";
      reportView.textContent = "(No CSV found)";
      console.error(err);
    });
}



  // Simple CSV parser (handles quotes). Assumes comma delimiter.
  function parseCSV(text){
    const rows = [];
    let row = [], val = '', inQuotes = false;
    for (let i=0; i<text.length; i++){
      const c = text[i], n = text[i+1];
      if (inQuotes){
        if (c === '"' && n === '"'){ val += '"'; i++; }
        else if (c === '"'){ inQuotes = false; }
        else { val += c; }
      } else {
        if (c === '"'){ inQuotes = true; }
        else if (c === ','){ row.push(val); val=''; }
        else if (c === '\n' || c === '\r'){
          if (val!=='' || row.length>0){ row.push(val); rows.push(row); row=[]; val=''; }
          // skip \r\n second char
          if (c === '\r' && n === '\n'){ i++; }
        } else { val += c; }
      }
    }
    if (val!=='' || row.length>0){ row.push(val); rows.push(row); }
    if (rows.length === 0) return {columns: [], rows: []};
    const columns = rows[0].map(h=>h.trim());
    const data = rows.slice(1).map(r=>{
      const o = {};
      for (let i=0;i<columns.length;i++){ o[columns[i]] = (r[i] ?? '').trim(); }
      return o;
    });
    return {columns, rows: data};
  }

  // Replace the old detectColumn(...) with this:
function detectColumn(columns, preferredExact, fallbackWord=null) {
  const lc = columns.map(c => c.toLowerCase());

  // 1) exact (case-insensitive) match among preferred names
  for (const key of preferredExact) {
    const idx = lc.indexOf(key.toLowerCase());
    if (idx !== -1) return columns[idx];
  }

  // 2) optional safe fallback: match a WHOLE WORD (split on non-alnum)
  if (fallbackWord) {
    const fw = fallbackWord.toLowerCase();
    for (let i = 0; i < lc.length; i++) {
      const parts = lc[i].split(/[^a-z0-9]+/);
      if (parts.includes(fw)) return columns[i];
    }
  }
  return null;
}

function isTruthy(v) {
  if (v == null) return false;
  const s = String(v).trim().toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'y';
}

// Reuse your detectColumn style for the "best" flag
function detectIsBestColumn(columns) {
  return detectColumn(
    columns,
    ['is_best','best','isBest','best_flag','is_best_candidate','isBestCandidate'],
    'best'
  );
}

// Single place to select a table row (adds highlight + sets state)
function selectTableRow(idx) {
  state.selectedRowIndex = idx;
  // update highlight
  Array.from(tbody.children).forEach((rowEl, rIdx) => {
    Array.from(rowEl.children).forEach(td => {
      if (rIdx === idx) td.classList.add('sel');
      else td.classList.remove('sel');
    });
  });
}

  // initial visible columns (tweak as you wish)
  function initialVisibleSet(cols){
    const preferred = [
      'candidate_method','Period_value','Eccentricity_value','bin_flag','is_best',
      'mass_flag_peri', 'bin_flag_Ftest', 'bic', 'chisqr', 'prob_bic', 'LS_iter_fap','prob_bicc', 'prob_ev_bic'
    ];
    const set = new Set();
    const lower = cols.map(c=>c.toLowerCase());
    for (const p of preferred){
      const idx = lower.indexOf(p.toLowerCase());
      if (idx !== -1) set.add(cols[idx]);
    }
    // ensure period/ecc present if detected
    if (state.periodCol) set.add(state.periodCol);
    if (state.eccCol) set.add(state.eccCol);
    if (state.solidCol) set.add(state.solidCol);
    // if empty, fall back to first ~10
    if (set.size === 0){
      for (let i=0;i<Math.min(10, cols.length); i++) set.add(cols[i]);
    }
    return set;
  }

  let currentVisible = new Set();
let lastColumnsForPopover = [];
function updateView(){
  if (!state.tableRows || !state.tableCols) return;

  // start from all rows
  let rows = state.tableRows.slice();

  // filtering
  if (state.filterCol && state.filterValue){
    const fv = String(state.filterValue).toLowerCase();
    rows = rows.filter(r => String(r[state.filterCol] ?? '').toLowerCase().includes(fv));
  }

  // sorting
  if (state.sortCol){
    const col = state.sortCol;
    const dir = state.sortDir === 'desc' ? -1 : 1;
    rows.sort((a,b)=>{
      const na = parseNumber(a[col]);
      const nb = parseNumber(b[col]);
      const aNum = !isNaN(na), bNum = !isNaN(nb);

      if (aNum && bNum){
        if (na < nb) return -1*dir;
        if (na > nb) return  1*dir;
        return 0;
      }
      // fallback: string compare
      const sa = String(a[col] ?? '');
      const sb = String(b[col] ?? '');
      if (sa < sb) return -1*dir;
      if (sa > sb) return  1*dir;
      return 0;
    });
  }

  state.viewRows = rows;
  // re-draw table with current visible columns
  renderTable(state.tableCols, state.viewRows);
}
function buildFilterControls(columns){
  if (!filterCol || !filterValue || !clearFilter) return;

  // populate column dropdown
  filterCol.innerHTML = '';
  const optAll = document.createElement('option');
  optAll.value = '';
  optAll.textContent = 'All columns';
  filterCol.appendChild(optAll);

  columns.forEach(c=>{
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    filterCol.appendChild(opt);
  });

  // reset state
  state.filterCol = null;
  state.filterValue = '';
  filterCol.value = '';
  filterValue.value = '';

  filterCol.onchange = ()=>{
    state.filterCol = filterCol.value || null;
    state.selectedRowIndex = null;
    updateView();
  };

  filterValue.oninput = ()=>{
    state.filterValue = filterValue.value;
    state.selectedRowIndex = null;
    updateView();
  };

  clearFilter.onclick = (e)=>{
    e.preventDefault();
    state.filterCol = null;
    state.filterValue = '';
    filterCol.value = '';
    filterValue.value = '';
    state.selectedRowIndex = null;
    updateView();
  };
}

function buildColumnToggles(columns){
  lastColumnsForPopover = columns.slice();
  // initialize the visible set (once or after CSV load)
  currentVisible = initialVisibleSet(columns);
  renderTable(columns, state.viewRows.length ? state.viewRows : state.tableRows);

  // render the checkbox list into the popover body
  renderColsPopover(columns);

  // actions
  colsSelectAll.onclick = (e)=>{
    e.preventDefault();
    currentVisible = new Set(columns);
    renderColsPopover(columns);
    renderTable(columns, state.viewRows.length ? state.viewRows : state.tableRows);
  };
  colsReset.onclick = (e)=>{
    e.preventDefault();
    currentVisible = initialVisibleSet(columns);
    renderColsPopover(columns);
    renderTable(columns, state.viewRows.length ? state.viewRows : state.tableRows);
  };

  // live filter
  colsSearch.oninput = ()=>{
    renderColsPopover(columns, colsSearch.value.trim().toLowerCase());
  };
}

function renderColsPopover(columns, filter=""){
  colsBody.innerHTML = '';
  const filtered = filter
    ? columns.filter(c => c.toLowerCase().includes(filter))
    : columns;

  if (filtered.length === 0){
    const p = document.createElement('p');
    p.className = 'paths';
    p.textContent = 'No columns match your filter.';
    colsBody.appendChild(p);
    return;
  }

  filtered.forEach(c=>{
    const wrap = document.createElement('label');
    wrap.className = 'chk';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = currentVisible.has(c);
    cb.onchange = ()=>{
      if (cb.checked) currentVisible.add(c); else currentVisible.delete(c);
      renderTable(columns, state.tableRows);
    };
    const span = document.createElement('span');
    span.textContent = c;
    wrap.appendChild(cb);
    wrap.appendChild(span);
    colsBody.appendChild(wrap);
  });
}

function renderTable(columns, rows){
  const visCols = columns.filter(c=> currentVisible.has(c));

  // header with sort indicators
  const headerHtml = visCols.map(c=>{
    let sortMark = '';
    if (state.sortCol === c){
      sortMark = state.sortDir === 'asc' ? ' ▲' : ' ▼';
    }
    return `<th>${escapeHtml(c)}${sortMark}</th>`;
  }).join('');
  thead.innerHTML = '<tr>' + headerHtml + '</tr>';

  // body
  tbody.innerHTML = '';
  rows.forEach((r, idx)=>{
    const tr = document.createElement('tr');
    tr.onclick = ()=>{
      selectTableRow(idx);
    };
    visCols.forEach(c=>{
      const td = document.createElement('td');
      td.textContent = r[c] ?? '';
      if (idx === state.selectedRowIndex) td.classList.add('sel');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // header click = sort by that column
  Array.from(thead.querySelectorAll('th')).forEach((th, i)=>{
    const colName = visCols[i];
    th.style.cursor = 'pointer';
    th.onclick = ()=>{
      if (state.sortCol === colName){
        state.sortDir = (state.sortDir === 'asc') ? 'desc' : 'asc';
      } else {
        state.sortCol = colName;
        state.sortDir = 'asc';
      }
      state.selectedRowIndex = null;
      updateView();
    };
  });
}



  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

    function onShowSolution(){
      if (state.selectedRowIndex == null){
        alert("Select a row in the table first.");
        return;
      }
      const star = state.active;
      const sourceRows = (state.viewRows && state.viewRows.length) ? state.viewRows : state.tableRows;
      const row = sourceRows[state.selectedRowIndex];
      if (!row){ return; }
    
      const pCol = state.periodCol, eCol = state.eccCol, sidCol = state.solidCol;
      if (!pCol || !eCol || !sidCol){
        alert("Could not detect period/ecc/solution_id columns in CSV.");
        return;
      }
      const P   = parseNumber(row[pCol]);
      const ecc = parseNumber(row[eCol]);
      const sid = parseNumber(row[sidCol]);
    
      if (!isFinite(sid)){
        alert("Selected row has invalid numeric solution_id.");
        return;
      }
    
      const SIDf = sid.toFixed(0);
    
      const base = `${RESBASE}/${encodeURIComponent(star)}/lmfit_solutions/${encodeURIComponent(star)}_sid-${SIDf}`;
      const timeUrl  = `${base}_time_residuals.html`;
      const phaseUrl = `${base}_phase_residuals.html`;
      const txtUrl   = `${base}_report.txt`;
    
      timeFrame.src = timeUrl;
      phaseFrame.src = phaseUrl;
      leftHeader.textContent = `Time & Phase residuals — sid-${SIDf}`;
    
      fetch(txtUrl)
        .then(r=> r.ok ? r.text() : "(report not found)")
        .then(t=>{
          rightHeader.textContent = `Report.txt — sid-${SIDf}`;
          reportView.textContent = t || "(empty report)";
        })
        .catch(_=>{
          rightHeader.textContent = `Report.txt — sid-${SIDf}`;
          reportView.textContent = "(report not found)";
        });
    }


</script>
</body>
</html>
"""

def find_stars(visuals_dir: Path):
    stars = []
    for star_dir in sorted(p for p in visuals_dir.iterdir() if p.is_dir()):
        star = star_dir.name
        overlay = star_dir / f"{star}_all_spectra_minmax_overlay.html"
        combined = star_dir / f"{star}_rv_vs_mjd_combined.html"
        if overlay.exists() and combined.exists():
            stars.append({
                "star": star,
                "overlay": overlay.relative_to(visuals_dir).as_posix(),
                "combined": combined.relative_to(visuals_dir).as_posix()
            })
    return stars

def write_index_html(visuals_dir: Path, results_relbase: str, stars):
    index_path = visuals_dir / "index.html"
    html = INDEX_TEMPLATE.replace("__STARS_JSON__", json.dumps(stars))
    html = html.replace("__BASE_URL__", ".")
    html = html.replace("__RESULTS_BASE__", results_relbase)
    index_path.write_text(html, encoding="utf-8")
    return index_path

def run_server(root_dir: Path, port: int):
    os.chdir(root_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing {root_dir} at http://localhost:{port}/visuals/index.html")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")

def main():
    ap = argparse.ArgumentParser(description="Build an improved GUI to browse star visuals + results")
    ap.add_argument("parent_dir", help="Path to parent directory that contains 'visuals/'")
    ap.add_argument("--results_dir", required=True, help="Path to parent directory that contains per-star results")
    ap.add_argument("--serve", action="store_true", help="Serve via a simple HTTP server")
    ap.add_argument("--port", type=int, default=8000, help="Port for --serve (default 8000)")
    args = ap.parse_args()

    parent = Path(args.parent_dir).expanduser().resolve()
    visuals = parent / "visuals"
    if not visuals.is_dir():
        raise SystemExit(f"Not found: {visuals}")

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Not found: {results_dir}")

    # Compute a path from /visuals/index.html to results_dir for browser fetches
    # i.e., relative path from visuals_dir to results_dir
    results_relbase = os.path.relpath(results_dir, start=visuals).replace(os.sep, "/")

    stars = find_stars(visuals)
    if not stars:
        print("No stars with both HTML files found under:", visuals)
    index_path = write_index_html(visuals, results_relbase, stars)
    print("Wrote:", index_path)
    print(f"Results base (relative): {results_relbase}")

    # You must serve a directory that contains BOTH the 'visuals' dir and the 'results_dir'
    # E.g., serve their common ancestor. Here we serve the parent of 'visuals' by default.
    if args.serve:
        # If results_dir is outside 'parent', advise user
        common = os.path.commonpath([parent, results_dir])
        if common != str(parent):
            print("\n[Note] You're serving:", parent)
            print("       Your results_dir is outside this tree.")
            print("       Either re-run with a common ancestor via: python -m http.server -d <COMMON_ROOT>")
        run_server(parent, args.port)
    else:
        print(f"\nOpen this file in your browser:\n  {index_path}\n")
        print("Tip: to avoid file:// iframe restrictions and ensure results are reachable, run a server at a common root, e.g.:")
        common_root = os.path.commonpath([str(parent), str(results_dir)])
        print(f"  python3 -m http.server -d {common_root} 8000")
        print("  -> then open:  http://localhost:8000/visuals/index.html")

if __name__ == "__main__":
    main()
