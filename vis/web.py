"""Generate a self-contained interactive HTML dashboard for ablation results.

Usage (standalone):
    python -m vis.web --work_dir /path/to/WORK_DIR -o dashboard.html

Or via main.py:
    python -m vis.main --work_dir /path/to/WORK_DIR --web
"""

import json
import math
import os

import numpy as np

from .config import MODEL_INFO, OVERALL_BENCHMARKS, BASE_MODEL, MODEL_LABELS
from .data_loader import ResultLoader


# ── Data export helpers ──────────────────────────────────────────────────

def _sanitize(obj):
    """Convert numpy / NaN values to JSON-safe Python types."""
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    return obj


def _df_to_dict(df):
    """DataFrame → {row_label: {col_label: value}} dict."""
    return {
        str(r): {str(c): df.loc[r, c] for c in df.columns}
        for r in df.index
    }


def export_data(loader):
    """Export all chart data as a JSON-serializable dict."""
    data = {}

    # Model registry
    data['models'] = [
        {'key': k, 'label': v[0], 'group': v[1], 'color': v[2],
         'isBase': k == BASE_MODEL}
        for k, v in MODEL_INFO.items()
    ]
    data['baseLabel'] = MODEL_LABELS[BASE_MODEL]
    data['benchmarks'] = list(OVERALL_BENCHMARKS)

    # Overall matrix (models × 7 benchmarks)
    data['overall'] = _df_to_dict(loader.load_overall_matrix())

    # VideoMME duration
    data['vmmeDuration'] = _df_to_dict(loader.load_videomme_duration())

    # MVBench sub-tasks
    data['mvbench'] = _df_to_dict(loader.load_mvbench_tasks())

    # VideoMME task types
    data['vmmeTasktype'] = _df_to_dict(loader.load_videomme_tasktype())

    # Video-Holmes question types
    data['videoholmes'] = _df_to_dict(loader.load_videoholmes_types())

    # PerceptionTest dims → {split: {model: {cat: val}}}
    dims = loader.load_perception_dims()
    data['perception'] = {s: _df_to_dict(df) for s, df in dims.items()}

    # Charades metrics
    data['charades'] = _df_to_dict(loader.load_charades_metrics())

    return _sanitize(data)


# ── Dashboard generation ─────────────────────────────────────────────────

def generate_dashboard(loader, output_path):
    """Generate a self-contained interactive HTML dashboard."""
    data = export_data(loader)
    data_json = json.dumps(data, ensure_ascii=False)
    html = _TEMPLATE.replace('/**__DATA__**/', f'const DATA = {data_json};')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  Dashboard saved: {output_path}')


def serve(loader, port=8890):
    """Serve the dashboard on localhost:port, regenerating HTML on each browser request."""
    import http.server
    import socketserver

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            data = export_data(loader)
            data_json = json.dumps(data, ensure_ascii=False)
            html = _TEMPLATE.replace('/**__DATA__**/', f'const DATA = {data_json};')
            content = html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt, *args):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('', port), _Handler) as httpd:
        import socket
        hostname = socket.gethostname()
        print(f'  Listening on 0.0.0.0:{port}')
        print(f'  Remote:  http://{hostname}:{port}/')
        print(f'  Local:   http://localhost:{port}/')
        print('  (or SSH tunnel: ssh -L {p}:localhost:{p} <server>)'.format(p=port))
        print('  Press Ctrl+C to stop.')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\n  Server stopped.')


# ── CLI entry ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate interactive dashboard')
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('-o', '--output', default='vis/output/dashboard.html')
    parser.add_argument('--serve', action='store_true', help='Start HTTP server instead of writing file')
    parser.add_argument('--port', type=int, default=8890)
    args = parser.parse_args()

    loader = ResultLoader(args.work_dir)
    if args.serve:
        serve(loader, port=args.port)
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        generate_dashboard(loader, args.output)


# ── HTML template ────────────────────────────────────────────────────────

_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ablation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
/* ── Reset & base ──────────────────────────────────────────────── */
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f0f2f5;color:#1a1a2e;display:flex;height:100vh;overflow:hidden}

/* ── Sidebar ───────────────────────────────────────────────────── */
.sidebar{width:240px;min-width:240px;background:#fff;border-right:1px solid #e0e0e0;
         display:flex;flex-direction:column;overflow:hidden}
.sidebar-header{padding:16px 16px 12px;border-bottom:1px solid #eee;flex-shrink:0}
.sidebar-header h1{font-size:15px;font-weight:700;color:#1a1a2e}
.sidebar-header p{font-size:11px;color:#888;margin-top:2px}
.model-panel{flex:1;overflow-y:auto;padding:8px 12px 16px}
.group-header{display:flex;align-items:center;justify-content:space-between;
              margin:12px 0 6px;padding:0 2px}
.group-header:first-child{margin-top:4px}
.group-title{font-size:12px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.5px}
.group-btns{display:flex;gap:4px}
.group-btns button{font-size:10px;padding:1px 6px;border:1px solid #ddd;
                   background:#fafafa;border-radius:3px;cursor:pointer;color:#666}
.group-btns button:hover{background:#e8e8e8}
.model-item{display:flex;align-items:center;padding:3px 4px;border-radius:4px;
            cursor:pointer;margin:1px 0;transition:background .15s}
.model-item:hover{background:#f5f5f5}
.model-item input[type="checkbox"]{margin-right:7px;accent-color:var(--mc)}
.model-item .dot{width:10px;height:10px;border-radius:50%;margin-right:6px;flex-shrink:0}
.model-item label{font-size:12px;cursor:pointer;user-select:none;white-space:nowrap;
                  overflow:hidden;text-overflow:ellipsis}
.model-item.is-base label{font-weight:700}
.sidebar-divider{height:1px;background:#e0e0e0;margin:0 12px}
.bench-item{display:flex;align-items:center;padding:3px 4px;border-radius:4px;
            cursor:pointer;margin:1px 0}
.bench-item:hover{background:#f5f5f5}
.bench-item input{margin-right:7px}
.bench-item label{font-size:12px;cursor:pointer;user-select:none}

/* ── Main ──────────────────────────────────────────────────────── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.tab-bar{display:flex;gap:0;background:#fff;border-bottom:1px solid #e0e0e0;
         padding:0 20px;flex-shrink:0}
.tab-btn{padding:12px 20px;font-size:13px;font-weight:600;color:#888;cursor:pointer;
         border:none;background:none;border-bottom:2px solid transparent;transition:all .2s}
.tab-btn:hover{color:#333}
.tab-btn.active{color:#1a73e8;border-bottom-color:#1a73e8}
.content{flex:1;overflow-y:auto;padding:20px}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* ── Cards & charts ────────────────────────────────────────────── */
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08);
      padding:20px;margin-bottom:20px}
.card-title{font-size:14px;font-weight:700;margin-bottom:14px;color:#1a1a2e}
.chart-box{width:100%;min-height:400px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:1200px){.chart-row{grid-template-columns:1fr}}

/* ── Score table ───────────────────────────────────────────────── */
.score-table{width:100%;border-collapse:collapse;font-size:12px}
.score-table th{position:sticky;top:0;background:#fafafa;padding:8px 10px;
                text-align:center;font-weight:600;border-bottom:2px solid #e0e0e0;
                cursor:pointer;user-select:none;white-space:nowrap}
.score-table th:hover{background:#f0f0f0}
.score-table th:first-child{text-align:left;cursor:default}
.score-table th.sort-asc::after{content:" ▲";font-size:9px;color:#1a73e8}
.score-table th.sort-desc::after{content:" ▼";font-size:9px;color:#1a73e8}
.score-table td{padding:6px 10px;text-align:center;border-bottom:1px solid #f0f0f0;
                font-variant-numeric:tabular-nums}
.score-table td:first-child{text-align:left;font-weight:600;white-space:nowrap}
.score-table td.cell{border-radius:3px}
.score-table tr.base-row{background:#f8f9ff}
.score-table td.avg-col{font-weight:700;border-left:2px solid #e0e0e0}

/* ── Sub-tabs for breakdown ────────────────────────────────────── */
.sub-tabs{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.sub-tab{padding:6px 14px;font-size:12px;border:1px solid #ddd;border-radius:16px;
         cursor:pointer;background:#fafafa;color:#555;transition:all .2s}
.sub-tab:hover{border-color:#aaa}
.sub-tab.active{background:#1a73e8;color:#fff;border-color:#1a73e8}
.sub-panel{display:none}
.sub-panel.active{display:block}

/* ── Misc ──────────────────────────────────────────────────────── */
.empty-msg{text-align:center;color:#999;padding:60px 20px;font-size:14px}
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
    <h1>Model Selection</h1>
    <p>Toggle models to compare</p>
  </div>
  <div class="model-panel" id="modelPanel"></div>
  <div class="sidebar-divider"></div>
  <div class="sidebar-header" style="border-bottom:none;padding:12px 16px 4px">
    <h1>Benchmarks</h1>
  </div>
  <div class="model-panel" id="benchPanel"
       style="flex:0 0 auto;max-height:220px;overflow-y:auto"></div>
</div>

<!-- Main -->
<div class="main">
  <div class="tab-bar" id="tabBar">
    <button class="tab-btn active" data-tab="overview">Overview</button>
    <button class="tab-btn" data-tab="radar">Radar</button>
    <button class="tab-btn" data-tab="delta">Delta</button>
    <button class="tab-btn" data-tab="breakdown">Breakdown</button>
  </div>
  <div class="content" id="content">
    <!-- Overview -->
    <div class="tab-panel active" id="tab-overview">
      <div class="card">
        <div class="card-title">Score Heatmap (models × benchmarks)</div>
        <div id="scoreTableWrap" style="overflow-x:auto"></div>
      </div>
      <div class="card">
        <div class="card-title">Average Score Ranking</div>
        <div class="chart-box" id="chartRanking"></div>
      </div>
    </div>

    <!-- Radar -->
    <div class="tab-panel" id="tab-radar">
      <div class="chart-row">
        <div class="card">
          <div class="card-title">AoT Ablation Radar (adaptive scale)</div>
          <div class="chart-box" id="chartRadarAot" style="min-height:500px"></div>
        </div>
        <div class="card">
          <div class="card-title">TG Ablation Radar (adaptive scale)</div>
          <div class="chart-box" id="chartRadarTg" style="min-height:500px"></div>
        </div>
      </div>
    </div>

    <!-- Delta -->
    <div class="tab-panel" id="tab-delta">
      <div class="card">
        <div class="card-title">Per-Benchmark Delta vs Base</div>
        <div class="chart-box" id="chartDelta" style="min-height:500px"></div>
      </div>
    </div>

    <!-- Breakdown -->
    <div class="tab-panel" id="tab-breakdown">
      <div class="sub-tabs" id="breakdownTabs">
        <button class="sub-tab active" data-sub="mvbench">MVBench</button>
        <button class="sub-tab" data-sub="vmme">VideoMME</button>
        <button class="sub-tab" data-sub="videoholmes">Video-Holmes</button>
        <button class="sub-tab" data-sub="perception">PerceptionTest</button>
        <button class="sub-tab" data-sub="charades">Charades</button>
      </div>
      <div class="sub-panel active" id="sub-mvbench">
        <div class="card">
          <div class="card-title">MVBench Sub-task Breakdown</div>
          <div class="chart-box" id="chartMvbench" style="min-height:500px"></div>
        </div>
      </div>
      <div class="sub-panel" id="sub-vmme">
        <div class="chart-row">
          <div class="card">
            <div class="card-title">VideoMME Duration</div>
            <div class="chart-box" id="chartVmmeDur"></div>
          </div>
          <div class="card">
            <div class="card-title">VideoMME Task Type</div>
            <div class="chart-box" id="chartVmmeTask"></div>
          </div>
        </div>
      </div>
      <div class="sub-panel" id="sub-videoholmes">
        <div class="card">
          <div class="card-title">Video-Holmes Question Type Breakdown</div>
          <div class="chart-box" id="chartHolmes" style="min-height:450px"></div>
        </div>
      </div>
      <div class="sub-panel" id="sub-perception">
        <div class="card">
          <div class="card-title">PerceptionTest Dimension Breakdown</div>
          <div class="chart-box" id="chartPerception" style="min-height:500px"></div>
        </div>
      </div>
      <div class="sub-panel" id="sub-charades">
        <div class="card">
          <div class="card-title">CharadesTimeLens Metrics</div>
          <div class="chart-box" id="chartCharades" style="min-height:450px"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
/**__DATA__**/

// ══════════════════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════════════════
const selected = new Set(DATA.models.map(m => m.label));
const selectedBench = new Set(DATA.benchmarks);
let activeTab = 'overview';
let activeSub = 'mvbench';
const charts = {};  // id → ECharts instance

// ══════════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════════
function getSelectedLabels() {
  return DATA.models.filter(m => selected.has(m.label)).map(m => m.label);
}
function getSelectedBenchmarks() {
  return DATA.benchmarks.filter(b => selectedBench.has(b));
}
function modelByLabel(label) {
  return DATA.models.find(m => m.label === label);
}
function vals(obj) { return obj ? Object.values(obj) : []; }
function keys(obj) { return obj ? Object.keys(obj) : []; }
function validNum(v) { return v !== null && v !== undefined && !isNaN(v); }

// Color interpolation for heatmap cells
function heatColor(val, lo, hi) {
  if (!validNum(val)) return 'transparent';
  const t = hi > lo ? (val - lo) / (hi - lo) : 0.5;
  // green(good) → yellow → red(bad) reversed: low=red, high=green
  const r = t < 0.5 ? 255 : Math.round(255 * (1 - t) * 2);
  const g = t > 0.5 ? 200 : Math.round(200 * t * 2);
  const b = 60;
  return `rgba(${r},${g},${b},0.18)`;
}

function ensureChart(id) {
  const dom = document.getElementById(id);
  if (!dom) return null;
  if (charts[id]) {
    charts[id].resize();
    return charts[id];
  }
  const c = echarts.init(dom);
  charts[id] = c;
  return c;
}

function disposeChart(id) {
  if (charts[id]) { charts[id].dispose(); delete charts[id]; }
}

// ══════════════════════════════════════════════════════════════════════════
// Sidebar: model panel
// ══════════════════════════════════════════════════════════════════════════
function buildModelPanel() {
  const panel = document.getElementById('modelPanel');
  const groups = {};
  DATA.models.forEach(m => {
    (groups[m.group] = groups[m.group] || []).push(m);
  });
  const groupNames = {'base': 'Base', 'aot': 'AoT Ablation', 'tg': 'TG Ablation'};
  let html = '';
  for (const [g, label] of Object.entries(groupNames)) {
    const models = groups[g] || [];
    if (!models.length) continue;
    html += `<div class="group-header">
      <span class="group-title">${label}</span>
      <span class="group-btns">
        <button onclick="toggleGroup('${g}',true)">All</button>
        <button onclick="toggleGroup('${g}',false)">None</button>
      </span>
    </div>`;
    models.forEach(m => {
      const cls = m.isBase ? 'model-item is-base' : 'model-item';
      html += `<div class="${cls}" style="--mc:${m.color}">
        <input type="checkbox" id="cb-${m.label}" checked
               onchange="onModelToggle('${m.label}', this.checked)">
        <span class="dot" style="background:${m.color}"></span>
        <label for="cb-${m.label}" title="${m.key}">${m.label}</label>
      </div>`;
    });
  }
  panel.innerHTML = html;
}

window.toggleGroup = function(group, on) {
  DATA.models.filter(m => m.group === group).forEach(m => {
    const cb = document.getElementById('cb-' + m.label);
    if (cb) { cb.checked = on; }
    if (on) selected.add(m.label); else selected.delete(m.label);
  });
  updateCharts();
};

window.onModelToggle = function(label, checked) {
  if (checked) selected.add(label); else selected.delete(label);
  updateCharts();
};

// ══════════════════════════════════════════════════════════════════════════
// Sidebar: benchmark panel
// ══════════════════════════════════════════════════════════════════════════
function buildBenchmarkPanel() {
  const panel = document.getElementById('benchPanel');
  let html = '<div class="group-header">' +
    '<span class="group-btns">' +
    '<button onclick="toggleAllBench(true)">All</button>' +
    '<button onclick="toggleAllBench(false)">None</button>' +
    '</span></div>';
  DATA.benchmarks.forEach(b => {
    html += `<div class="bench-item">` +
      `<input type="checkbox" id="bcb-${b}" checked ` +
      `onchange="onBenchToggle('${b}',this.checked)">` +
      `<label for="bcb-${b}">${b}</label></div>`;
  });
  panel.innerHTML = html;
}

window.toggleAllBench = function(on) {
  DATA.benchmarks.forEach(b => {
    const cb = document.getElementById('bcb-' + b);
    if (cb) cb.checked = on;
    if (on) selectedBench.add(b); else selectedBench.delete(b);
  });
  updateCharts();
};

window.onBenchToggle = function(b, checked) {
  if (checked) selectedBench.add(b); else selectedBench.delete(b);
  updateCharts();
};

// ══════════════════════════════════════════════════════════════════════════
// Tabs
// ══════════════════════════════════════════════════════════════════════════
function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.getElementById('tab-' + tab).classList.add('active');
      activeTab = tab;
      updateCharts();
    });
  });
  document.querySelectorAll('.sub-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sub-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.sub-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('sub-' + btn.dataset.sub).classList.add('active');
      activeSub = btn.dataset.sub;
      updateCharts();
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════
// Update orchestrator
// ══════════════════════════════════════════════════════════════════════════
function updateCharts() {
  switch (activeTab) {
    case 'overview': renderScoreTable(); renderRanking(); break;
    case 'radar': renderAotRadar(); renderTgRadar(); break;
    case 'delta': renderDeltaBar(); break;
    case 'breakdown':
      switch (activeSub) {
        case 'mvbench': renderGroupedBar('chartMvbench', DATA.mvbench, 'MVBench Sub-tasks'); break;
        case 'vmme': renderGroupedBar('chartVmmeDur', DATA.vmmeDuration, 'Duration');
                     renderGroupedBar('chartVmmeTask', DATA.vmmeTasktype, 'Task Type'); break;
        case 'videoholmes': renderGroupedBar('chartHolmes', DATA.videoholmes, 'Question Type'); break;
        case 'perception': renderPerception(); break;
        case 'charades': renderGroupedBar('chartCharades', DATA.charades, 'Metrics'); break;
      }
      break;
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Chart: Score Table (Overview)
// ══════════════════════════════════════════════════════════════════════════
let tableSortCol = 'avg';
let tableSortAsc = false;

function renderScoreTable() {
  const wrap = document.getElementById('scoreTableWrap');
  const labels = getSelectedLabels();
  const benchmarks = getSelectedBenchmarks();
  if (!labels.length) { wrap.innerHTML = '<p class="empty-msg">Select at least one model</p>'; return; }

  // Compute data rows
  const rows = labels.map(label => {
    const d = DATA.overall[label] || {};
    const scores = benchmarks.map(b => d[b] ?? null);
    const valid = scores.filter(validNum);
    const avg = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
    return { label, scores, avg };
  });

  // Sort
  if (tableSortCol === 'avg') {
    rows.sort((a, b) => {
      const va = a.avg ?? -Infinity, vb = b.avg ?? -Infinity;
      return tableSortAsc ? va - vb : vb - va;
    });
  } else if (tableSortCol === 'model') {
    rows.sort((a, b) => tableSortAsc ? a.label.localeCompare(b.label) : b.label.localeCompare(a.label));
  } else {
    const ci = benchmarks.indexOf(tableSortCol);
    rows.sort((a, b) => {
      const va = a.scores[ci] ?? -Infinity, vb = b.scores[ci] ?? -Infinity;
      return tableSortAsc ? va - vb : vb - va;
    });
  }

  // Column ranges for heatmap
  const colRanges = benchmarks.map((_, ci) => {
    const vs = rows.map(r => r.scores[ci]).filter(validNum);
    return vs.length ? [Math.min(...vs), Math.max(...vs)] : [0, 100];
  });
  const avgVs = rows.map(r => r.avg).filter(validNum);
  const avgRange = avgVs.length ? [Math.min(...avgVs), Math.max(...avgVs)] : [0, 100];

  // Build HTML
  const sortCls = (col) => {
    if (tableSortCol !== col) return '';
    return tableSortAsc ? 'sort-asc' : 'sort-desc';
  };
  let html = '<table class="score-table"><thead><tr>';
  html += `<th class="${sortCls('model')}" onclick="sortTable('model')">Model</th>`;
  benchmarks.forEach(b => {
    html += `<th class="${sortCls(b)}" onclick="sortTable('${b}')">${b}</th>`;
  });
  html += `<th class="${sortCls('avg')}" onclick="sortTable('avg')">Avg</th>`;
  html += '</tr></thead><tbody>';
  rows.forEach(r => {
    const m = modelByLabel(r.label);
    const isBase = m && m.isBase;
    html += `<tr class="${isBase ? 'base-row' : ''}">`;
    const c = m ? m.color : '#888';
    html += `<td><span class="dot" style="display:inline-block;width:8px;height:8px;` +
      `border-radius:50%;background:${c};margin-right:5px;vertical-align:middle">` +
      `</span>${r.label}</td>`;
    r.scores.forEach((s, ci) => {
      const bg = heatColor(s, colRanges[ci][0], colRanges[ci][1]);
      html += `<td class="cell" style="background:${bg}">${validNum(s) ? s.toFixed(1) : '—'}</td>`;
    });
    const avgBg = heatColor(r.avg, avgRange[0], avgRange[1]);
    html += `<td class="avg-col cell" style="background:${avgBg}">${validNum(r.avg) ? r.avg.toFixed(1) : '—'}</td>`;
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

window.sortTable = function(col) {
  if (tableSortCol === col) tableSortAsc = !tableSortAsc;
  else { tableSortCol = col; tableSortAsc = false; }
  renderScoreTable();
};

// ══════════════════════════════════════════════════════════════════════════
// Chart: Ranking Bar (Overview)
// ══════════════════════════════════════════════════════════════════════════
function renderRanking() {
  const chart = ensureChart('chartRanking');
  if (!chart) return;
  const labels = getSelectedLabels();
  const benchmarks = getSelectedBenchmarks();

  // Compute averages
  const items = labels.map(label => {
    const d = DATA.overall[label] || {};
    const vs = benchmarks.map(b => d[b]).filter(validNum);
    return { label, avg: vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null,
             scores: benchmarks.map(b => d[b]) };
  }).filter(i => validNum(i.avg)).sort((a, b) => a.avg - b.avg);

  if (!items.length) { chart.clear(); return; }

  // Scatter series for individual benchmarks
  const scatterSeries = benchmarks.map((b, bi) => ({
    type: 'scatter',
    name: b,
    data: items.map((it, yi) => {
      const v = it.scores[bi];
      return validNum(v) ? [v, yi] : null;
    }).filter(Boolean),
    symbolSize: 8,
    itemStyle: { opacity: 0.7 },
    z: 10,
  }));

  chart.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: params => {
        let tip = `<b>${items[params[0].dataIndex]?.label || ''}</b><br>`;
        params.forEach(p => {
          if (p.seriesType === 'bar') tip += `Avg: <b>${p.value.toFixed(1)}</b><br>`;
          else tip += `${p.seriesName}: ${p.value[0].toFixed(1)}<br>`;
        });
        return tip;
      }
    },
    legend: { show: true, top: 0, type: 'scroll', textStyle: { fontSize: 10 } },
    grid: { left: 100, right: 40, top: 40, bottom: 20 },
    xAxis: { type: 'value', name: 'Score' },
    yAxis: {
      type: 'category',
      data: items.map(i => i.label),
      axisLabel: { fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        name: 'Average',
        data: items.map(i => i.avg),
        barWidth: '50%',
        itemStyle: {
          color: params => {
            const m = modelByLabel(items[params.dataIndex]?.label);
            return m ? m.color : '#888';
          }
        },
        label: { show: true, position: 'right', fontSize: 11,
                 formatter: p => p.value.toFixed(1) },
        z: 5,
      },
      ...scatterSeries,
    ],
  }, true);
}

// ══════════════════════════════════════════════════════════════════════════
// Chart: Radar
// ══════════════════════════════════════════════════════════════════════════
function renderRadar_(chartId, groupFilter) {
  const chart = ensureChart(chartId);
  if (!chart) return;
  const benchmarks = getSelectedBenchmarks();

  // Filter to relevant models (base + matching group)
  const relevant = DATA.models.filter(m =>
    (m.isBase || m.group === groupFilter) && selected.has(m.label)
  );
  if (relevant.length < 2 || !benchmarks.length) {
    chart.clear();
    chart.setOption({ title: { text: 'Select at least 2 models', left: 'center', top: 'center',
                               textStyle: { color: '#999', fontSize: 14 } } });
    return;
  }

  // Gather raw scores
  const rawScores = {};
  relevant.forEach(m => {
    const d = DATA.overall[m.label] || {};
    rawScores[m.label] = benchmarks.map(b => d[b] ?? null);
  });

  // Adaptive per-axis range: center on data; min span = 8pp; pad = max(3, spread*0.5)
  const ranges = benchmarks.map((b, i) => {
    const vs = relevant.map(m => rawScores[m.label][i]).filter(validNum);
    if (!vs.length) return [0, 100];
    const lo = Math.min(...vs), hi = Math.max(...vs);
    const pad = Math.max(3, (hi - lo) * 0.5);
    let mn = Math.max(0, lo - pad), mx = Math.min(100, hi + pad);
    if (mx - mn < 8) { const mid = (mn + mx) / 2; mn = Math.max(0, mid - 4); mx = Math.min(100, mid + 4); }
    return [mn, mx];
  });

  const indicator = benchmarks.map((b, i) => ({
    name: `${b}\n(${ranges[i][0].toFixed(1)}~${ranges[i][1].toFixed(1)})`,
    min: ranges[i][0], max: ranges[i][1],
  }));

  const series = relevant.map(m => ({
    value: rawScores[m.label].map((v, i) => validNum(v) ? v : ranges[i][0]),
    name: m.label,
    lineStyle: m.isBase ? { type: 'dashed', width: 3 } : { width: 2 },
    areaStyle: { opacity: 0.05 },
    itemStyle: { color: m.color },
    symbol: 'circle',
    symbolSize: 4,
  }));

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: params => {
        if (!params.value) return '';
        let tip = `<b>${params.name}</b><br>`;
        benchmarks.forEach((b, i) => {
          const raw = rawScores[params.name]?.[i];
          tip += `${b}: ${validNum(raw) ? raw.toFixed(1) : '—'}<br>`;
        });
        return tip;
      }
    },
    legend: { data: relevant.map(m => m.label), top: 0, type: 'scroll',
              textStyle: { fontSize: 10 } },
    radar: { indicator, radius: '65%', center: ['50%', '55%'],
             axisName: { fontSize: 10, color: '#555' },
             splitArea: { areaStyle: { color: ['#fff', '#f9f9f9'] } } },
    series: [{ type: 'radar', data: series }],
  }, true);
}

function renderAotRadar() { renderRadar_('chartRadarAot', 'aot'); }
function renderTgRadar() { renderRadar_('chartRadarTg', 'tg'); }

// ══════════════════════════════════════════════════════════════════════════
// Chart: Delta Bar
// ══════════════════════════════════════════════════════════════════════════
function renderDeltaBar() {
  const chart = ensureChart('chartDelta');
  if (!chart) return;
  const benchmarks = getSelectedBenchmarks();
  const baseData = DATA.overall[DATA.baseLabel] || {};

  // Only non-base selected models
  const models = DATA.models.filter(m => !m.isBase && selected.has(m.label));
  if (!models.length) {
    chart.clear();
    chart.setOption({ title: { text: 'Select ablation models to compare', left: 'center',
                               top: 'center', textStyle: { color: '#999', fontSize: 14 } } });
    return;
  }

  const series = models.map(m => ({
    type: 'bar',
    name: m.label,
    data: benchmarks.map(b => {
      const base = baseData[b];
      const cur = (DATA.overall[m.label] || {})[b];
      return (validNum(base) && validNum(cur)) ? Math.round((cur - base) * 100) / 100 : null;
    }),
    itemStyle: { color: m.color },
  }));

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: models.map(m => m.label), top: 0, type: 'scroll',
              textStyle: { fontSize: 10 } },
    grid: { left: 60, right: 20, top: 50, bottom: 80 },
    xAxis: {
      type: 'category', data: benchmarks,
      axisLabel: { rotate: 25, fontSize: 11 },
    },
    yAxis: { type: 'value', name: 'Delta vs Base', splitLine: { lineStyle: { type: 'dashed' } } },
    series,
  }, true);
}

// ══════════════════════════════════════════════════════════════════════════
// Chart: Generic Grouped Bar (for breakdowns)
// ══════════════════════════════════════════════════════════════════════════
function renderGroupedBar(chartId, dataDict, title) {
  const chart = ensureChart(chartId);
  if (!chart) return;
  const labels = getSelectedLabels().filter(l => dataDict[l]);
  if (!labels.length) {
    chart.clear();
    chart.setOption({ title: { text: 'No data for selected models', left: 'center',
                               top: 'center', textStyle: { color: '#999', fontSize: 14 } } });
    return;
  }

  // Get all categories (union of all keys)
  const catSet = new Set();
  labels.forEach(l => keys(dataDict[l]).forEach(k => catSet.add(k)));
  const categories = [...catSet].sort();

  if (!categories.length) { chart.clear(); return; }

  const series = labels.map(label => {
    const m = modelByLabel(label);
    return {
      type: 'bar',
      name: label,
      data: categories.map(c => dataDict[label]?.[c] ?? null),
      itemStyle: { color: m ? m.color : '#888' },
    };
  });

  const rotate = categories.length > 6 ? 45 : 0;
  const bottomPad = categories.length > 6 ? 120 : 60;

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
               confine: true },
    legend: { data: labels, top: 0, type: 'scroll', textStyle: { fontSize: 10 } },
    grid: { left: 60, right: 20, top: 50, bottom: bottomPad },
    xAxis: {
      type: 'category', data: categories,
      axisLabel: { rotate, fontSize: 10, interval: 0 },
    },
    yAxis: { type: 'value', name: 'Score (%)', splitLine: { lineStyle: { type: 'dashed' } } },
    series,
    dataZoom: categories.length > 15 ? [
      { type: 'slider', start: 0, end: 100, bottom: 10, height: 20 },
    ] : [],
  }, true);
}

// ══════════════════════════════════════════════════════════════════════════
// Chart: PerceptionTest (multi-split)
// ══════════════════════════════════════════════════════════════════════════
function renderPerception() {
  const chart = ensureChart('chartPerception');
  if (!chart) return;
  const splits = keys(DATA.perception);
  const labels = getSelectedLabels();
  if (!splits.length || !labels.length) { chart.clear(); return; }

  // Merge all splits into one grouped bar with split prefix
  const categories = [];
  const dataMap = {};
  splits.sort().forEach(split => {
    const splitData = DATA.perception[split];
    const cats = new Set();
    labels.forEach(l => { if (splitData[l]) keys(splitData[l]).forEach(c => cats.add(c)); });
    [...cats].sort().forEach(cat => {
      const fullCat = `${split}/${cat}`;
      categories.push(fullCat);
      labels.forEach(l => {
        dataMap[l] = dataMap[l] || [];
        dataMap[l].push(splitData[l]?.[cat] ?? null);
      });
    });
  });

  const series = labels.filter(l => dataMap[l]).map(label => {
    const m = modelByLabel(label);
    return {
      type: 'bar', name: label, data: dataMap[label],
      itemStyle: { color: m ? m.color : '#888' },
    };
  });

  chart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true },
    legend: { data: labels, top: 0, type: 'scroll', textStyle: { fontSize: 10 } },
    grid: { left: 60, right: 20, top: 50, bottom: 120 },
    xAxis: { type: 'category', data: categories,
             axisLabel: { rotate: 45, fontSize: 9, interval: 0 } },
    yAxis: { type: 'value', name: 'Accuracy (%)',
             splitLine: { lineStyle: { type: 'dashed' } } },
    series,
    dataZoom: categories.length > 12 ? [
      { type: 'slider', start: 0, end: 100, bottom: 10, height: 20 },
    ] : [],
  }, true);
}

// ══════════════════════════════════════════════════════════════════════════
// Resize handler
// ══════════════════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {
  Object.values(charts).forEach(c => c.resize());
});

// ══════════════════════════════════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════════════════════════════════
buildModelPanel();
setupTabs();
updateCharts();

</script>
</body>
</html>
'''


if __name__ == '__main__':
    main()
