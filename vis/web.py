"""Self-contained interactive HTML dashboard for evaluation results.

Generates a single HTML file with embedded ECharts that:
1. Score Table — heatmap-colored matrix with config badges & breakdown drill-down
2. Model Config — side-by-side comparison of model settings
3. Charts — radar / bar chart selector
"""

import html
import http.server
import json
import math
import os
import socketserver

import numpy as np

from .data_loader import ResultLoader


def _sanitize(obj):
    """Convert numpy/NaN types to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 2)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def export_data(loader: ResultLoader) -> dict:
    """Build the complete JSON data payload for the dashboard."""
    df = loader.load_all_scores()
    models = loader.models
    benchmarks = loader.benchmarks

    # Build score matrix
    scores = {}
    for m in models:
        row = {}
        for b in benchmarks:
            val = df.loc[m, b] if m in df.index and b in df.columns else float('nan')
            row[b] = _sanitize(val)
        scores[m] = row

    # Model configs
    configs = {}
    for m in models:
        cfg = loader.model_config(m)
        configs[m] = _sanitize(cfg) if cfg else None

    # Breakdowns
    breakdowns = {}
    for m in models:
        bd = {}
        for b in benchmarks:
            data = loader.load_breakdown(m, b)
            if data:
                bd[b] = _sanitize(data)
        if bd:
            breakdowns[m] = bd

    return _sanitize({
        'models': [{'key': m, 'color': loader.model_color(m)} for m in models],
        'benchmarks': benchmarks,
        'scores': scores,
        'configs': configs,
        'breakdowns': breakdowns,
    })


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background:#f5f6fa; color:#2c3e50; }

.header { background:#2c3e50; color:#fff; padding:12px 24px; display:flex; align-items:center; gap:16px; }
.header h1 { font-size:18px; font-weight:600; }
.tabs { display:flex; gap:4px; }
.tab { padding:6px 16px; border-radius:4px; cursor:pointer; font-size:13px; background:rgba(255,255,255,0.1); color:#ccc; transition:all 0.15s; }
.tab:hover { background:rgba(255,255,255,0.15); }
.tab.active { background:rgba(255,255,255,0.25); color:#fff; font-weight:600; }

.content { padding:16px 24px; }

/* ── Score Table tab ── */
.table-wrap { overflow-x:auto; max-height:calc(100vh - 120px); overflow-y:auto; }
table.score-table { border-collapse:collapse; font-size:12px; width:100%; }
table.score-table th, table.score-table td { padding:6px 10px; border:1px solid #ddd; text-align:center; white-space:nowrap; }
table.score-table th { background:#34495e; color:#fff; cursor:pointer; user-select:none; position:sticky; top:0; z-index:2; }
table.score-table th:hover { background:#4a6785; }
table.score-table td.model-name { text-align:left; font-weight:500; background:#fafafa; position:sticky; left:0; z-index:1; }
table.score-table tr:hover td { outline: 2px solid #3498db; }
.badge { display:inline-block; font-size:9px; padding:1px 5px; border-radius:3px; margin-left:4px; font-weight:600; vertical-align:middle; }
.badge-vllm { background:#3498db; color:#fff; }
.badge-adaptive { background:#e67e22; color:#fff; }
.badge-fps { background:#27ae60; color:#fff; }
.badge-nframe { background:#9b59b6; color:#fff; }
td.clickable { cursor:pointer; }
td.clickable:hover { filter:brightness(0.9); }

/* Breakdown row */
tr.breakdown-row td { padding:8px 12px; background:#f8f9fa; border-top:none; }
.breakdown-chart { width:100%; height:180px; }

/* ── Config tab ── */
.config-layout { display:flex; gap:16px; min-height:calc(100vh - 120px); }
.config-sidebar { width:240px; flex-shrink:0; background:#fff; border-radius:8px; padding:12px; overflow-y:auto; max-height:calc(100vh - 120px); box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.config-main { flex:1; background:#fff; border-radius:8px; padding:16px; overflow-x:auto; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.config-section { margin-bottom:12px; }
.config-section h4 { font-size:12px; color:#888; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; border-bottom:1px solid #eee; padding-bottom:4px; }
table.config-table { border-collapse:collapse; font-size:12px; width:100%; }
table.config-table th, table.config-table td { padding:5px 10px; border:1px solid #eee; text-align:left; white-space:nowrap; }
table.config-table th { background:#f0f3f5; font-weight:600; position:sticky; top:0; }
table.config-table td.key-col { font-weight:500; color:#555; background:#fafafa; width:180px; }
table.config-table td.diff { background:#fff3cd; }

.check-item { display:flex; align-items:center; gap:6px; padding:3px 0; font-size:12px; cursor:pointer; }
.check-item input { margin:0; }
.check-item .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }

/* ── Charts tab ── */
.chart-layout { display:flex; gap:16px; min-height:calc(100vh - 120px); }
.chart-sidebar { width:260px; flex-shrink:0; background:#fff; border-radius:8px; padding:12px; overflow-y:auto; max-height:calc(100vh - 120px); box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.chart-main { flex:1; display:flex; flex-direction:column; gap:16px; }
.chart-box { background:#fff; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); flex:1; min-height:400px; }

.sidebar-section { margin-bottom:16px; }
.sidebar-section h3 { font-size:13px; font-weight:600; margin-bottom:8px; color:#555; display:flex; align-items:center; justify-content:space-between; }
.sidebar-section h3 .btn-group { display:flex; gap:4px; }
.sidebar-section h3 .btn-group button { font-size:10px; padding:2px 6px; border:1px solid #ccc; border-radius:3px; background:#f9f9f9; cursor:pointer; }
.sidebar-section h3 .btn-group button:hover { background:#eee; }
.check-list { max-height:300px; overflow-y:auto; }

.chart-type-bar { display:flex; gap:4px; margin-bottom:12px; }
.chart-type-btn { padding:4px 12px; border:1px solid #ccc; border-radius:4px; cursor:pointer; font-size:12px; background:#f9f9f9; }
.chart-type-btn.active { background:#3498db; color:#fff; border-color:#3498db; }
</style>
</head>
<body>

<div class="header">
  <h1>Eval Dashboard</h1>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('table')">Score Table</div>
    <div class="tab" onclick="switchTab('config')">Model Config</div>
    <div class="tab" onclick="switchTab('charts')">Charts</div>
  </div>
</div>

<div class="content">
  <!-- Tab 1: Score Table -->
  <div id="tab-table">
    <div class="table-wrap" id="score-table-wrap"></div>
  </div>

  <!-- Tab 2: Model Config -->
  <div id="tab-config" style="display:none">
    <div class="config-layout">
      <div class="config-sidebar">
        <div class="sidebar-section">
          <h3>Select Models</h3>
          <div class="check-list" id="config-model-checks"></div>
        </div>
      </div>
      <div class="config-main" id="config-table-wrap">
        <p style="color:#999;text-align:center;padding:40px">Select models to compare</p>
      </div>
    </div>
  </div>

  <!-- Tab 3: Charts -->
  <div id="tab-charts" style="display:none">
    <div class="chart-layout">
      <div class="chart-sidebar">
        <div class="sidebar-section">
          <h3>Models <span class="btn-group"><button onclick="toggleAll('model',true)">All</button><button onclick="toggleAll('model',false)">None</button></span></h3>
          <div class="check-list" id="model-checks"></div>
        </div>
        <div class="sidebar-section">
          <h3>Benchmarks <span class="btn-group"><button onclick="toggleAll('bench',true)">All</button><button onclick="toggleAll('bench',false)">None</button></span></h3>
          <div class="check-list" id="bench-checks"></div>
        </div>
      </div>
      <div class="chart-main">
        <div class="chart-type-bar">
          <div class="chart-type-btn active" onclick="switchChartType('radar')">Radar</div>
          <div class="chart-type-btn" onclick="switchChartType('bar')">Bar</div>
        </div>
        <div class="chart-box" id="chart-container"></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = /**__DATA__**/;

const models = DATA.models;
const benchmarks = DATA.benchmarks;
const scores = DATA.scores;
const configs = DATA.configs || {};
const breakdowns = DATA.breakdowns || {};

let activeTab = 'table';
let chartType = 'radar';
let selectedModels = new Set(models.map(m => m.key));
let selectedBench = new Set(benchmarks);
let selectedConfigModels = new Set(models.slice(0, Math.min(4, models.length)).map(m => m.key));
let chartInstance = null;
let sortCol = null;
let sortAsc = true;
let expandedCell = null;  // {model, bench} or null
let breakdownCharts = {};

// ── Tab switching ──
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  ['table','config','charts'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t === tab ? '' : 'none';
  });
  if (tab === 'charts') updateChart();
  if (tab === 'config') renderConfigTable();
}

// ═══════════════════════════════════════════════════════
//  TAB 1: Score Table
// ═══════════════════════════════════════════════════════

function configBadges(model) {
  const cfg = configs[model];
  if (!cfg) return '';
  let b = '';
  if (cfg.use_vllm) b += '<span class="badge badge-vllm">vLLM</span>';
  if (cfg.dataset_adaptive) b += '<span class="badge badge-adaptive">adaptive</span>';
  if (cfg.fps != null && !cfg.dataset_adaptive) b += `<span class="badge badge-fps">${cfg.fps}fps</span>`;
  if (cfg.dataset_fps != null && cfg.dataset_fps > 0) b += `<span class="badge badge-fps">ds:${cfg.dataset_fps}fps</span>`;
  if (cfg.nframe != null && cfg.nframe > 0 && !cfg.dataset_adaptive) b += `<span class="badge badge-nframe">${cfg.nframe}f</span>`;
  return b;
}

function renderTable() {
  let modelKeys = models.map(m => m.key);

  if (sortCol !== null && sortCol !== '__avg__') {
    modelKeys.sort((a, b) => {
      let va = scores[a]?.[sortCol], vb = scores[b]?.[sortCol];
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
      return sortAsc ? va - vb : vb - va;
    });
  } else if (sortCol === '__avg__') {
    modelKeys.sort((a, b) => {
      const avgA = calcAvg(a), avgB = calcAvg(b);
      return sortAsc ? avgA - avgB : avgB - avgA;
    });
  }

  const colMin = {}, colMax = {};
  benchmarks.forEach(b => {
    let vals = modelKeys.map(m => scores[m]?.[b]).filter(v => v != null);
    colMin[b] = vals.length ? Math.min(...vals) : 0;
    colMax[b] = vals.length ? Math.max(...vals) : 100;
  });

  let h = '<table class="score-table"><thead><tr><th>Model</th>';
  benchmarks.forEach(b => {
    const arrow = sortCol === b ? (sortAsc ? ' ▲' : ' ▼') : '';
    h += `<th onclick="sortTable('${esc(b)}')">${b}${arrow}</th>`;
  });
  h += '<th onclick="sortTable(\'__avg__\')">Avg' + (sortCol === '__avg__' ? (sortAsc ? ' ▲' : ' ▼') : '') + '</th>';
  h += '</tr></thead><tbody>';

  modelKeys.forEach(m => {
    h += '<tr>';
    h += `<td class="model-name">${m}${configBadges(m)}</td>`;
    let sum = 0, cnt = 0;
    benchmarks.forEach(b => {
      const v = scores[m]?.[b];
      if (v != null) {
        const range = colMax[b] - colMin[b];
        const ratio = range > 0 ? (v - colMin[b]) / range : 0.5;
        const bg = heatColor(ratio);
        const hasBd = breakdowns[m]?.[b] != null;
        const cls = hasBd ? 'clickable' : '';
        h += `<td class="${cls}" style="background:${bg}" ${hasBd ? `onclick="toggleBreakdown('${esc(m)}','${esc(b)}')"` : ''}>${v.toFixed(1)}</td>`;
        sum += v; cnt++;
      } else {
        h += '<td style="color:#ccc">\u2014</td>';
      }
    });
    const avg = cnt > 0 ? (sum / cnt).toFixed(1) : '\u2014';
    h += `<td style="font-weight:600">${avg}</td>`;
    h += '</tr>';

    // Breakdown row (hidden by default)
    if (expandedCell && expandedCell.model === m) {
      const bd = breakdowns[m]?.[expandedCell.bench];
      if (bd) {
        h += `<tr class="breakdown-row" id="bd-row-${esc(m)}">`;
        h += `<td colspan="${benchmarks.length + 2}">`;
        h += `<strong>${expandedCell.bench}</strong> breakdown for ${m}`;
        h += `<div class="breakdown-chart" id="bd-chart-${esc(m)}-${esc(expandedCell.bench)}"></div>`;
        h += '</td></tr>';
      }
    }
  });

  h += '</tbody></table>';
  document.getElementById('score-table-wrap').innerHTML = h;

  // Render breakdown chart if expanded
  if (expandedCell) {
    const bd = breakdowns[expandedCell.model]?.[expandedCell.bench];
    if (bd) renderBreakdownChart(expandedCell.model, expandedCell.bench, bd);
  }
}

function toggleBreakdown(model, bench) {
  if (expandedCell && expandedCell.model === model && expandedCell.bench === bench) {
    expandedCell = null;
  } else {
    expandedCell = {model, bench};
  }
  renderTable();
}

function renderBreakdownChart(model, bench, bd) {
  const id = `bd-chart-${esc(model)}-${esc(bench)}`;
  const el = document.getElementById(id);
  if (!el) return;

  const keys = Object.keys(bd).filter(k => typeof bd[k] === 'number').sort();
  const vals = keys.map(k => bd[k]);

  const chart = echarts.init(el);
  chart.setOption({
    grid: { left: 200, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', max: Math.max(100, ...vals) },
    yAxis: { type: 'category', data: keys, axisLabel: { fontSize: 10, width: 180, overflow: 'truncate' } },
    series: [{ type: 'bar', data: vals, itemStyle: { color: '#3498db' }, barWidth: 14, label: { show: true, position: 'right', fontSize: 10, formatter: p => p.value.toFixed(1) } }],
    tooltip: { trigger: 'axis' },
  });
}

function calcAvg(m) {
  let s = 0, c = 0;
  benchmarks.forEach(b => { const v = scores[m]?.[b]; if (v != null) { s += v; c++; } });
  return c > 0 ? s / c : -Infinity;
}

function heatColor(ratio) {
  const r = Math.round(255 * (1 - ratio) * 0.8 + 240 * 0.2);
  const g = Math.round(255 * ratio * 0.7 + 240 * 0.3);
  const b = Math.round(240 * 0.3);
  return `rgb(${r},${g},${b})`;
}

function sortTable(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = false; }
  renderTable();
}

function esc(s) { return s.replace(/'/g, "\\'").replace(/"/g, '&quot;'); }

// ═══════════════════════════════════════════════════════
//  TAB 2: Model Config
// ═══════════════════════════════════════════════════════

const CONFIG_GROUPS = [
  { label: 'Inference', keys: ['model_class', 'model_path', 'use_vllm', 'use_lmdeploy'] },
  { label: 'Sampling', keys: ['fps', 'nframe', 'dataset_adaptive', 'dataset_fps', 'dataset_nframe'] },
  { label: 'Resolution', keys: ['min_pixels', 'max_pixels', 'total_pixels', 'limit_mm_per_prompt'] },
  { label: 'Generation', keys: ['system_prompt', 'generation_kwargs'] },
];

function initConfigSidebar() {
  let h = '';
  models.forEach((m, i) => {
    const checked = selectedConfigModels.has(m.key) ? 'checked' : '';
    h += `<label class="check-item">
      <input type="checkbox" ${checked} onchange="toggleConfigModel('${esc(m.key)}')" data-key="${m.key}">
      <span class="dot" style="background:${m.color}"></span>
      <span>${m.key}</span>
    </label>`;
  });
  document.getElementById('config-model-checks').innerHTML = h;
}

function toggleConfigModel(key) {
  if (selectedConfigModels.has(key)) selectedConfigModels.delete(key);
  else selectedConfigModels.add(key);
  renderConfigTable();
}

function renderConfigTable() {
  const sel = models.filter(m => selectedConfigModels.has(m.key));
  if (!sel.length) {
    document.getElementById('config-table-wrap').innerHTML = '<p style="color:#999;text-align:center;padding:40px">Select models to compare</p>';
    return;
  }

  // Collect all config values
  const cfgMap = {};
  sel.forEach(m => { cfgMap[m.key] = configs[m.key] || {}; });

  let h = '';
  CONFIG_GROUPS.forEach(group => {
    h += `<div class="config-section"><h4>${group.label}</h4>`;
    h += '<table class="config-table"><thead><tr><th>Key</th>';
    sel.forEach(m => { h += `<th style="min-width:120px"><span class="dot" style="background:${m.color};display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px"></span>${m.key}</th>`; });
    h += '</tr></thead><tbody>';

    group.keys.forEach(k => {
      const vals = sel.map(m => formatVal(cfgMap[m.key]?.[k]));
      const allSame = vals.every(v => v === vals[0]);
      h += '<tr>';
      h += `<td class="key-col">${k}</td>`;
      vals.forEach(v => {
        const cls = (!allSame && sel.length > 1) ? 'diff' : '';
        h += `<td class="${cls}">${v}</td>`;
      });
      h += '</tr>';
    });

    h += '</tbody></table></div>';
  });

  document.getElementById('config-table-wrap').innerHTML = h;
}

function formatVal(v) {
  if (v === null || v === undefined) return '<span style="color:#ccc">null</span>';
  if (typeof v === 'boolean') return v ? '<span style="color:#27ae60">true</span>' : '<span style="color:#999">false</span>';
  if (typeof v === 'object') return '<span style="color:#888;font-size:10px">' + JSON.stringify(v).slice(0, 60) + '</span>';
  return String(v);
}

// ═══════════════════════════════════════════════════════
//  TAB 3: Charts
// ═══════════════════════════════════════════════════════

function initSidebar() {
  let mhtml = '';
  models.forEach(m => {
    mhtml += `<label class="check-item">
      <input type="checkbox" checked onchange="toggleModel('${esc(m.key)}')" data-group="model" data-key="${m.key}">
      <span class="dot" style="background:${m.color}"></span>
      <span>${m.key}</span>
    </label>`;
  });
  document.getElementById('model-checks').innerHTML = mhtml;

  let bhtml = '';
  benchmarks.forEach(b => {
    bhtml += `<label class="check-item">
      <input type="checkbox" checked onchange="toggleBench('${esc(b)}')" data-group="bench" data-key="${b}">
      <span>${b}</span>
    </label>`;
  });
  document.getElementById('bench-checks').innerHTML = bhtml;
}

function toggleModel(key) {
  if (selectedModels.has(key)) selectedModels.delete(key); else selectedModels.add(key);
  updateChart();
}
function toggleBench(key) {
  if (selectedBench.has(key)) selectedBench.delete(key); else selectedBench.add(key);
  updateChart();
}
function toggleAll(group, state) {
  document.querySelectorAll(`input[data-group="${group}"]`).forEach(cb => {
    cb.checked = state;
    const key = cb.dataset.key;
    if (group === 'model') { state ? selectedModels.add(key) : selectedModels.delete(key); }
    else { state ? selectedBench.add(key) : selectedBench.delete(key); }
  });
  updateChart();
}

function switchChartType(type) {
  chartType = type;
  document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.chart-type-btn[onclick="switchChartType('${type}')"]`).classList.add('active');
  updateChart();
}

function updateChart() {
  const container = document.getElementById('chart-container');
  if (!chartInstance) chartInstance = echarts.init(container);
  chartInstance.clear();
  chartInstance.resize();

  const selModels = models.filter(m => selectedModels.has(m.key));
  const selBench = benchmarks.filter(b => selectedBench.has(b));
  if (!selModels.length || !selBench.length) {
    chartInstance.setOption({ title: { text: 'Select models and benchmarks', left: 'center', top: 'center' } });
    return;
  }

  if (chartType === 'radar') renderRadar(selModels, selBench);
  else renderBar(selModels, selBench);
}

function renderRadar(selModels, selBench) {
  const indicators = selBench.map(b => {
    let vals = selModels.map(m => scores[m.key]?.[b]).filter(v => v != null);
    let min = vals.length ? Math.min(...vals) : 0;
    let max = vals.length ? Math.max(...vals) : 100;
    let pad = (max - min) * 0.1 || 5;
    return { name: b, min: Math.max(0, Math.floor(min - pad)), max: Math.ceil(max + pad) };
  });

  chartInstance.setOption({
    tooltip: { trigger: 'item' },
    legend: { data: selModels.map(m => m.key), bottom: 0, type: 'scroll', textStyle: { fontSize: 11 } },
    radar: { indicator: indicators, radius: '60%', nameGap: 8, axisName: { fontSize: 11 } },
    series: [{ type: 'radar', data: selModels.map(m => ({
      name: m.key,
      value: selBench.map(b => scores[m.key]?.[b] ?? 0),
      lineStyle: { width: 2 },
      areaStyle: { opacity: 0.1 },
      itemStyle: { color: m.color },
    })) }],
  });
}

function renderBar(selModels, selBench) {
  chartInstance.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: selModels.map(m => m.key), bottom: 0, type: 'scroll', textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 20, bottom: 60 },
    xAxis: { type: 'category', data: selBench, axisLabel: { rotate: 30, fontSize: 11 } },
    yAxis: { type: 'value', name: 'Score' },
    series: selModels.map(m => ({
      name: m.key, type: 'bar',
      data: selBench.map(b => scores[m.key]?.[b] ?? 0),
      itemStyle: { color: m.color },
    })),
  });
}

// ── Init ──
window.addEventListener('resize', () => { if (chartInstance) chartInstance.resize(); });
renderTable();
initSidebar();
initConfigSidebar();
</script>
</body>
</html>
"""


def generate_dashboard(loader: ResultLoader, output_path: str):
    """Write a self-contained HTML dashboard to output_path."""
    data = export_data(loader)
    data_json = json.dumps(data, ensure_ascii=False)
    content = _TEMPLATE.replace('/**__DATA__**/', data_json)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Dashboard written to {output_path}')


def serve(loader: ResultLoader, port: int = 8890):
    """Start an HTTP server that regenerates the dashboard on each request."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            data = export_data(loader)
            data_json = json.dumps(data, ensure_ascii=False)
            content = _TEMPLATE.replace('/**__DATA__**/', data_json)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(('', port), Handler) as httpd:
        print(f'Dashboard serving at http://localhost:{port}  (Ctrl+C to stop)')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nShutting down.')


def main():
    """CLI entry point for standalone usage."""
    import argparse
    parser = argparse.ArgumentParser(description='Eval dashboard')
    parser.add_argument('--work-dir', '--work_dir', required=True, help='Results directory')
    parser.add_argument('-o', '--output', default=None, help='Output HTML file (default: serve mode)')
    parser.add_argument('--port', type=int, default=8890)
    args = parser.parse_args()

    loader = ResultLoader(args.work_dir)
    print(f'Discovered {len(loader.models)} models, {len(loader.benchmarks)} benchmarks')

    if args.output:
        generate_dashboard(loader, args.output)
    else:
        serve(loader, args.port)


if __name__ == '__main__':
    main()
