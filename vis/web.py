"""Self-contained interactive HTML dashboard for evaluation results."""

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


def export_data(loader: ResultLoader, include_overlap: bool = False) -> dict:
    """Build the complete JSON data payload for the dashboard."""
    df = loader.load_all_column_scores()
    models = loader.models
    table_groups = loader.table_groups
    table_columns = loader.table_columns

    scores = {}
    for model in models:
        row = {}
        for col in table_columns:
            col_id = col['id']
            val = df.loc[model, col_id] if model in df.index and col_id in df.columns else float('nan')
            row[col_id] = _sanitize(val)
        scores[model] = row

    configs = {}
    for model in models:
        cfg = loader.model_config(model)
        configs[model] = _sanitize(cfg) if cfg else None

    breakdowns = {}
    for model in models:
        per_model = {}
        for col in table_columns:
            data = loader.load_column_breakdown(model, col['id'])
            if data:
                per_model[col['id']] = _sanitize(data)
        if per_model:
            breakdowns[model] = per_model

    payload = {
        'models': [{'key': model, 'color': loader.model_color(model)} for model in models],
        'table_groups': table_groups,
        'table_columns': table_columns,
        'scores': scores,
        'configs': configs,
        'breakdowns': breakdowns,
    }
    if include_overlap:
        payload['overlap'] = loader.load_overlap_analysis(max_case_matrix=0)
    return _sanitize(payload)


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
table.score-table th { color:#fff; cursor:pointer; user-select:none; position:sticky; z-index:2; }
table.score-table thead tr.group-row th { top:0; background:#2c3e50; }
table.score-table thead tr.sub-row th { top:35px; background:#405a73; }
table.score-table th.rowspan-head { z-index:3; }
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

/* ── Overlap tab ── */
.overlap-layout { display:flex; gap:16px; min-height:calc(100vh - 120px); }
.overlap-sidebar { width:260px; flex-shrink:0; background:#fff; border-radius:8px; padding:12px; overflow-y:auto; max-height:calc(100vh - 120px); box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.overlap-main { flex:1; display:flex; flex-direction:column; gap:16px; min-width:0; }
.overlap-grid { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:16px; }
.overlap-panel { background:#fff; border-radius:8px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,0.1); min-width:0; }
.overlap-panel h3 { font-size:13px; font-weight:600; margin-bottom:10px; color:#475569; }
.overlap-chart { width:100%; height:300px; }
.select-control { width:100%; padding:6px 8px; border:1px solid #cbd5e1; border-radius:4px; background:#fff; font-size:12px; color:#334155; }
.compact { max-height:260px; overflow:auto; }
table.overlap-table { width:100%; border-collapse:collapse; font-size:12px; }
table.overlap-table th, table.overlap-table td { padding:6px 8px; border:1px solid #e2e8f0; text-align:right; white-space:nowrap; }
table.overlap-table th { background:#f1f5f9; color:#334155; font-weight:600; position:sticky; top:0; z-index:1; }
table.overlap-table td.left, table.overlap-table th.left { text-align:left; }
.delta-pos { color:#15803d; font-weight:700; }
.delta-neg { color:#b91c1c; font-weight:700; }
.empty-state { color:#94a3b8; text-align:center; padding:24px; font-size:13px; }
@media (max-width: 980px) {
  .overlap-layout { flex-direction:column; }
  .overlap-sidebar { width:100%; max-height:none; }
  .overlap-grid { grid-template-columns:1fr; }
}
</style>
</head>
<body>

<div class="header">
  <h1>Eval Dashboard</h1>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('table')">Score Table</div>
    <div class="tab" onclick="switchTab('config')">Model Config</div>
    <div class="tab" onclick="switchTab('charts')">Charts</div>
    <div class="tab overlap-tab" style="display:none" onclick="switchTab('overlap')">Overlap</div>
  </div>
</div>

<div class="content">
  <div id="tab-table">
    <div class="table-layout" style="display:flex;gap:16px;">
      <div class="table-sidebar" style="width:240px;flex-shrink:0;background:#fff;border-radius:8px;padding:12px;overflow-y:auto;max-height:calc(100vh - 120px);box-shadow:0 1px 3px rgba(0,0,0,0.1);">
        <div class="sidebar-section">
          <h3>Models <span class="btn-group"><button onclick="toggleAllTable(true)">All</button><button onclick="toggleAllTable(false)">None</button></span></h3>
          <div class="check-list" id="table-model-checks"></div>
        </div>
      </div>
      <div style="flex:1;overflow:hidden;">
        <div class="table-wrap" id="score-table-wrap"></div>
      </div>
    </div>
  </div>

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

  <div id="tab-overlap" style="display:none">
    <div class="overlap-layout">
      <div class="overlap-sidebar">
        <div class="sidebar-section">
          <h3>Baseline</h3>
          <select id="overlap-baseline" class="select-control" onchange="setOverlapBaseline(this.value)"></select>
        </div>
        <div class="sidebar-section">
          <h3>Candidate</h3>
          <select id="overlap-candidate" class="select-control" onchange="setOverlapCandidate(this.value)"></select>
        </div>
        <div class="sidebar-section">
          <h3>Dataset</h3>
          <select id="overlap-dataset" class="select-control" onchange="setOverlapDataset(this.value)"></select>
        </div>
      </div>
      <div class="overlap-main">
        <div class="overlap-panel">
          <h3>Pairwise Overlap</h3>
          <div class="table-wrap compact" id="overlap-pair-table"></div>
        </div>
        <div class="overlap-grid">
          <div class="overlap-panel">
            <h3>Dataset Delta</h3>
            <div class="overlap-chart" id="overlap-dataset-chart"></div>
          </div>
          <div class="overlap-panel">
            <h3>Subclass Delta</h3>
            <div class="overlap-chart" id="overlap-group-chart"></div>
          </div>
        </div>
        <div class="overlap-panel">
          <h3>Top Subclasses</h3>
          <div class="table-wrap compact" id="overlap-group-table"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = /**__DATA__**/;

const models = DATA.models || [];
const tableGroups = DATA.table_groups || [];
const tableColumns = DATA.table_columns || [];
const columnById = Object.fromEntries(tableColumns.map(col => [col.id, col]));
const scores = DATA.scores || {};
const configs = DATA.configs || {};
const breakdowns = DATA.breakdowns || {};
const overlap = DATA.overlap || null;

let activeTab = 'table';
let chartType = 'radar';
let selectedModels = new Set(models.map(m => m.key));
let selectedBench = new Set(tableColumns.map(col => col.id));
let selectedConfigModels = new Set(models.slice(0, Math.min(4, models.length)).map(m => m.key));
let chartInstance = null;
let overlapDatasetChart = null;
let overlapGroupChart = null;
let sortCol = null;
let sortAsc = true;
let expandedCell = null;
let selectedTableModels = new Set(models.map(m => m.key));
let overlapBaseline = overlap?.baseline || models[0]?.key || '';
let overlapCandidate = (overlap?.models || []).find(m => m !== overlapBaseline) || overlap?.models?.[0] || '';
let overlapDataset = '__all__';

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  ['table', 'config', 'charts', 'overlap'].forEach(t => {
    document.getElementById('tab-' + t).style.display = t === tab ? '' : 'none';
  });
  if (tab === 'charts') updateChart();
  if (tab === 'config') renderConfigTable();
  if (tab === 'overlap') renderOverlap();
}

function configBadges(model) {
  const cfg = configs[model];
  if (!cfg) return '';
  let badges = '';
  if (cfg.use_vllm) badges += '<span class="badge badge-vllm">vLLM</span>';
  if (cfg.dataset_adaptive) badges += '<span class="badge badge-adaptive">adaptive</span>';
  if (cfg.fps != null && !cfg.dataset_adaptive) badges += `<span class="badge badge-fps">${cfg.fps}fps</span>`;
  if (cfg.dataset_fps != null && cfg.dataset_fps > 0) badges += `<span class="badge badge-fps">ds:${cfg.dataset_fps}fps</span>`;
  if (cfg.nframe != null && cfg.nframe > 0 && !cfg.dataset_adaptive) badges += `<span class="badge badge-nframe">${cfg.nframe}f</span>`;
  return badges;
}

function sortTable(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = false; }
  renderTable();
}

function calcAvg(model) {
  let sum = 0;
  let cnt = 0;
  tableGroups.forEach(group => {
    const summaryId = group.summary_column_id || group.columns[group.columns.length - 1]?.id;
    const value = summaryId ? scores[model]?.[summaryId] : null;
    if (value != null) {
      sum += value;
      cnt++;
    }
  });
  return cnt > 0 ? sum / cnt : -Infinity;
}

function heatColor(ratio) {
  let r, g, b;
  if (ratio < 0.5) {
    const t = ratio * 2;
    r = Math.round(207 + (154 - 207) * t);
    g = Math.round(34 + (103 - 34) * t);
    b = Math.round(46 + (0 - 46) * t);
  } else {
    const t = (ratio - 0.5) * 2;
    r = Math.round(154 + (26 - 154) * t);
    g = Math.round(103 + (127 - 103) * t);
    b = Math.round(0 + (55 - 0) * t);
  }
  return `rgb(${r},${g},${b})`;
}

function renderTable() {
  let modelKeys = models.map(m => m.key).filter(m => selectedTableModels.has(m));

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

  const colMin = {};
  const colMax = {};
  tableColumns.forEach(col => {
    const vals = modelKeys.map(m => scores[m]?.[col.id]).filter(v => v != null);
    colMin[col.id] = vals.length ? Math.min(...vals) : 0;
    colMax[col.id] = vals.length ? Math.max(...vals) : 100;
  });

  const hasMultiHeader = tableGroups.some(group => group.columns.length > 1);
  let h = '<table class="score-table"><thead><tr class="group-row">';
  h += `<th class="rowspan-head" ${hasMultiHeader ? 'rowspan="2"' : ''}>Model</th>`;
  tableGroups.forEach(group => {
    const arrow = sortCol === group.summary_column_id ? (sortAsc ? ' ▲' : ' ▼') : '';
    if (group.columns.length === 1 || !hasMultiHeader) {
      h += `<th class="rowspan-head" ${hasMultiHeader ? 'rowspan="2"' : ''} onclick="sortTable('${esc(group.summary_column_id)}')">${group.label}${arrow}</th>`;
    } else {
      h += `<th colspan="${group.columns.length}" onclick="sortTable('${esc(group.summary_column_id)}')">${group.label}${arrow}</th>`;
    }
  });
  h += `<th class="rowspan-head" ${hasMultiHeader ? 'rowspan="2"' : ''} onclick="sortTable('__avg__')">Avg${sortCol === '__avg__' ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>`;
  h += '</tr>';

  if (hasMultiHeader) {
    h += '<tr class="sub-row">';
    tableGroups.forEach(group => {
      if (group.columns.length <= 1) return;
      group.columns.forEach(col => {
        const arrow = sortCol === col.id ? (sortAsc ? ' ▲' : ' ▼') : '';
        h += `<th onclick="sortTable('${esc(col.id)}')">${col.label}${arrow}</th>`;
      });
    });
    h += '</tr>';
  }
  h += '</thead><tbody>';

  modelKeys.forEach(model => {
    h += '<tr>';
    h += `<td class="model-name">${model}${configBadges(model)}</td>`;
    tableGroups.forEach(group => {
      group.columns.forEach(col => {
        const value = scores[model]?.[col.id];
        if (value != null) {
          const range = colMax[col.id] - colMin[col.id];
          const ratio = range > 0 ? (value - colMin[col.id]) / range : 0.5;
          const color = heatColor(ratio);
          const hasBreakdown = breakdowns[model]?.[col.id] != null;
          const cls = hasBreakdown ? 'clickable' : '';
          const onclick = hasBreakdown ? `onclick="toggleBreakdown('${esc(model)}','${esc(col.id)}')"` : '';
          h += `<td class="${cls}" style="color:${color};font-weight:600" ${onclick}>${value.toFixed(2)}</td>`;
        } else {
          h += '<td style="color:#ccc">—</td>';
        }
      });
    });
    const avg = calcAvg(model);
    h += `<td style="font-weight:600">${avg > -Infinity ? avg.toFixed(2) : '—'}</td>`;
    h += '</tr>';

    if (expandedCell && expandedCell.model === model) {
      const bd = breakdowns[model]?.[expandedCell.columnId];
      if (bd) {
        const label = columnById[expandedCell.columnId]?.chart_label || expandedCell.columnId;
        h += `<tr class="breakdown-row"><td colspan="${tableColumns.length + 2}"><strong>${label}</strong> breakdown for ${model}<div class="breakdown-chart" id="bd-chart-${esc(model)}-${esc(expandedCell.columnId)}"></div></td></tr>`;
      }
    }
  });

  h += '</tbody></table>';
  document.getElementById('score-table-wrap').innerHTML = h;

  if (expandedCell) {
    const bd = breakdowns[expandedCell.model]?.[expandedCell.columnId];
    if (bd) renderBreakdownChart(expandedCell.model, expandedCell.columnId, bd);
  }
}

function toggleBreakdown(model, columnId) {
  if (expandedCell && expandedCell.model === model && expandedCell.columnId === columnId) {
    expandedCell = null;
  } else {
    expandedCell = {model, columnId};
  }
  renderTable();
}

function renderBreakdownChart(model, columnId, bd) {
  const el = document.getElementById(`bd-chart-${esc(model)}-${esc(columnId)}`);
  if (!el) return;
  const keys = Object.keys(bd).filter(k => typeof bd[k] === 'number').sort();
  const vals = keys.map(k => bd[k]);
  const chart = echarts.init(el);
  chart.setOption({
    grid: { left: 200, right: 30, top: 10, bottom: 20 },
    xAxis: { type: 'value', max: Math.max(100, ...vals) },
    yAxis: { type: 'category', data: keys, axisLabel: { fontSize: 10, width: 180, overflow: 'truncate' } },
    series: [{
      type: 'bar',
      data: vals,
      itemStyle: { color: '#3498db' },
      barWidth: 14,
      label: { show: true, position: 'right', fontSize: 10, formatter: p => p.value.toFixed(2) },
    }],
    tooltip: { trigger: 'axis' },
  });
}

function esc(s) {
  return s.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function initTableSidebar() {
  let h = '';
  models.forEach(m => {
    h += `<label class="check-item">
      <input type="checkbox" checked onchange="toggleTableModel('${esc(m.key)}')" data-group="tablemodel" data-key="${m.key}">
      <span class="dot" style="background:${m.color}"></span>
      <span>${m.key}</span>
    </label>`;
  });
  document.getElementById('table-model-checks').innerHTML = h;
}

function toggleTableModel(key) {
  if (selectedTableModels.has(key)) selectedTableModels.delete(key);
  else selectedTableModels.add(key);
  renderTable();
}

function toggleAllTable(state) {
  document.querySelectorAll('input[data-group="tablemodel"]').forEach(cb => {
    cb.checked = state;
    const key = cb.dataset.key;
    state ? selectedTableModels.add(key) : selectedTableModels.delete(key);
  });
  renderTable();
}

const CONFIG_GROUPS = [
  { label: 'Inference', keys: ['model_class', 'model_path', 'use_vllm', 'use_lmdeploy'] },
  { label: 'Sampling', keys: ['fps', 'nframe', 'dataset_adaptive', 'dataset_fps', 'dataset_nframe'] },
  { label: 'Resolution', keys: ['min_pixels', 'max_pixels', 'total_pixels', 'limit_mm_per_prompt'] },
  { label: 'Generation', keys: ['system_prompt', 'generation_kwargs'] },
];

function initConfigSidebar() {
  let h = '';
  models.forEach(m => {
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

  const cfgMap = {};
  sel.forEach(m => { cfgMap[m.key] = configs[m.key] || {}; });

  let h = '';
  CONFIG_GROUPS.forEach(group => {
    h += `<div class="config-section"><h4>${group.label}</h4>`;
    h += '<table class="config-table"><thead><tr><th>Key</th>';
    sel.forEach(m => {
      h += `<th style="min-width:120px"><span class="dot" style="background:${m.color};display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px"></span>${m.key}</th>`;
    });
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
  tableColumns.forEach(col => {
    bhtml += `<label class="check-item">
      <input type="checkbox" checked onchange="toggleBench('${esc(col.id)}')" data-group="bench" data-key="${col.id}">
      <span>${col.chart_label || col.id}</span>
    </label>`;
  });
  document.getElementById('bench-checks').innerHTML = bhtml;
}

function toggleModel(key) {
  if (selectedModels.has(key)) selectedModels.delete(key);
  else selectedModels.add(key);
  updateChart();
}

function toggleBench(key) {
  if (selectedBench.has(key)) selectedBench.delete(key);
  else selectedBench.add(key);
  updateChart();
}

function toggleAll(group, state) {
  document.querySelectorAll(`input[data-group="${group}"]`).forEach(cb => {
    cb.checked = state;
    const key = cb.dataset.key;
    if (group === 'model') {
      state ? selectedModels.add(key) : selectedModels.delete(key);
    } else {
      state ? selectedBench.add(key) : selectedBench.delete(key);
    }
  });
  updateChart();
}

function switchChartType(type) {
  chartType = type;
  document.querySelectorAll('.chart-type-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelector(`.chart-type-btn[onclick="switchChartType('${type}')"]`).classList.add('active');
  updateChart();
}

function updateChart() {
  const container = document.getElementById('chart-container');
  if (!chartInstance) chartInstance = echarts.init(container);
  chartInstance.clear();
  chartInstance.resize();

  const selModels = models.filter(m => selectedModels.has(m.key));
  const selCols = tableColumns.filter(col => selectedBench.has(col.id));
  if (!selModels.length || !selCols.length) {
    chartInstance.setOption({ title: { text: 'Select models and benchmarks', left: 'center', top: 'center' } });
    return;
  }

  if (chartType === 'radar') renderRadar(selModels, selCols);
  else renderBar(selModels, selCols);
}

function renderRadar(selModels, selCols) {
  const indicators = selCols.map(col => {
    const vals = selModels.map(m => scores[m.key]?.[col.id]).filter(v => v != null);
    const min = vals.length ? Math.min(...vals) : 0;
    const max = vals.length ? Math.max(...vals) : 100;
    const pad = (max - min) * 0.1 || 5;
    return {
      name: col.chart_label || col.id,
      min: Math.max(0, Math.floor(min - pad)),
      max: Math.ceil(max + pad),
    };
  });

  chartInstance.setOption({
    tooltip: { trigger: 'item' },
    legend: { data: selModels.map(m => m.key), bottom: 0, type: 'scroll', textStyle: { fontSize: 11 } },
    radar: { indicator: indicators, radius: '60%', nameGap: 8, axisName: { fontSize: 11 } },
    series: [{
      type: 'radar',
      data: selModels.map(m => ({
        name: m.key,
        value: selCols.map(col => scores[m.key]?.[col.id] ?? null),
        lineStyle: { width: 2 },
        areaStyle: { opacity: 0.1 },
        itemStyle: { color: m.color },
      })),
    }],
  });
}

function renderBar(selModels, selCols) {
  chartInstance.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { data: selModels.map(m => m.key), bottom: 0, type: 'scroll', textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 20, top: 20, bottom: 80 },
    xAxis: {
      type: 'category',
      data: selCols.map(col => col.chart_label || col.id),
      axisLabel: { rotate: 30, fontSize: 11 },
    },
    yAxis: { type: 'value', name: 'Score' },
    series: selModels.map(m => ({
      name: m.key,
      type: 'bar',
      data: selCols.map(col => scores[m.key]?.[col.id] ?? null),
      itemStyle: { color: m.color },
    })),
  });
}

function fmtNum(value, digits = 2) {
  return value == null ? '—' : Number(value).toFixed(digits);
}

function fmtPct(value) {
  return value == null ? '—' : (Number(value) * 100).toFixed(1) + '%';
}

function deltaClass(value) {
  if (value == null || Number(value) === 0) return '';
  return Number(value) > 0 ? 'delta-pos' : 'delta-neg';
}

function initOverlapControls() {
  const overlapTab = document.querySelector('.overlap-tab');
  if (!overlap) {
    if (overlapTab) overlapTab.style.display = 'none';
    return;
  }
  if (overlapTab) overlapTab.style.display = '';
  const modelOptions = (overlap.models || models.map(m => m.key)).map(model => `<option value="${esc(model)}">${model}</option>`).join('');
  const datasetOptions = ['<option value="__all__">All</option>']
    .concat((overlap.datasets || []).map(dataset => `<option value="${esc(dataset)}">${dataset}</option>`))
    .join('');
  document.getElementById('overlap-baseline').innerHTML = modelOptions;
  document.getElementById('overlap-candidate').innerHTML = modelOptions;
  document.getElementById('overlap-dataset').innerHTML = datasetOptions;
  document.getElementById('overlap-baseline').value = overlapBaseline;
  document.getElementById('overlap-candidate').value = overlapCandidate;
  document.getElementById('overlap-dataset').value = overlapDataset;
}

function setOverlapBaseline(value) {
  overlapBaseline = value;
  if (overlapCandidate === overlapBaseline) {
    overlapCandidate = (overlap?.models || []).find(m => m !== overlapBaseline) || overlapCandidate;
    const candEl = document.getElementById('overlap-candidate');
    if (candEl) candEl.value = overlapCandidate;
  }
  renderOverlap();
}

function setOverlapCandidate(value) {
  overlapCandidate = value;
  renderOverlap();
}

function setOverlapDataset(value) {
  overlapDataset = value;
  renderOverlap();
}

function overlapDatasetFilter(row) {
  return overlapDataset === '__all__' || row.dataset === overlapDataset;
}

function selectedPair(row) {
  return (
    (row.model_a === overlapBaseline && row.model_b === overlapCandidate) ||
    (row.model_a === overlapCandidate && row.model_b === overlapBaseline)
  );
}

function renderOverlap() {
  if (!overlap || !(overlap.pairwise_overlap || []).length) {
    document.getElementById('tab-overlap').innerHTML = '<div class="empty-state">No row-level overlap data</div>';
    return;
  }
  renderOverlapPairTable();
  renderOverlapDatasetChart();
  renderOverlapGroupChart();
  renderOverlapGroupTable();
}

function renderOverlapPairTable() {
  const rows = (overlap.pairwise_overlap || [])
    .filter(overlapDatasetFilter)
    .filter(row => selectedPair(row))
    .sort((a, b) => (b.disagreement_rate || 0) - (a.disagreement_rate || 0));
  const visible = rows.length ? rows : (overlap.pairwise_overlap || []).filter(overlapDatasetFilter).slice(0, 50);
  if (!visible.length) {
    document.getElementById('overlap-pair-table').innerHTML = '<div class="empty-state">No overlap rows</div>';
    return;
  }

  let h = '<table class="overlap-table"><thead><tr>';
  ['Dataset', 'Model A', 'Model B', 'Shared', 'Both ✓', 'A only', 'B only', 'Both ×', 'Disagree', 'Δ B-A'].forEach((label, idx) => {
    h += `<th class="${idx < 3 ? 'left' : ''}">${label}</th>`;
  });
  h += '</tr></thead><tbody>';
  visible.forEach(row => {
    h += '<tr>';
    h += `<td class="left">${row.dataset}</td>`;
    h += `<td class="left">${row.model_a}</td>`;
    h += `<td class="left">${row.model_b}</td>`;
    h += `<td>${row.shared_cases}</td>`;
    h += `<td>${row.both_correct}</td>`;
    h += `<td>${row.model_a_only}</td>`;
    h += `<td>${row.model_b_only}</td>`;
    h += `<td>${row.both_wrong}</td>`;
    h += `<td>${fmtPct(row.disagreement_rate)}</td>`;
    h += `<td class="${deltaClass(row.delta_b_minus_a)}">${fmtNum(row.delta_b_minus_a)}</td>`;
    h += '</tr>';
  });
  h += '</tbody></table>';
  document.getElementById('overlap-pair-table').innerHTML = h;
}

function selectedDatasetDeltas() {
  return (overlap.dataset_deltas || [])
    .filter(row => row.baseline_model === overlapBaseline)
    .filter(row => !overlapCandidate || row.candidate_model === overlapCandidate)
    .filter(overlapDatasetFilter);
}

function renderOverlapDatasetChart() {
  const el = document.getElementById('overlap-dataset-chart');
  if (!overlapDatasetChart) overlapDatasetChart = echarts.init(el);
  overlapDatasetChart.clear();
  const rows = selectedDatasetDeltas().sort((a, b) => a.delta - b.delta);
  if (!rows.length) {
    overlapDatasetChart.setOption({ title: { text: 'No dataset deltas', left: 'center', top: 'center', textStyle: { color: '#94a3b8', fontSize: 13 } } });
    return;
  }
  overlapDatasetChart.setOption({
    grid: { left: 140, right: 32, top: 12, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', name: 'Δ' },
    yAxis: {
      type: 'category',
      data: rows.map(row => row.dataset),
      axisLabel: { fontSize: 10, width: 130, overflow: 'truncate' },
    },
    series: [{
      type: 'bar',
      data: rows.map(row => row.delta),
      itemStyle: { color: p => p.value >= 0 ? '#55A868' : '#C44E52' },
      label: { show: true, position: 'right', fontSize: 10, formatter: p => Number(p.value).toFixed(2) },
    }],
  });
}

function selectedGroupDeltas() {
  return (overlap.group_deltas || [])
    .filter(row => row.baseline_model === overlapBaseline)
    .filter(row => !overlapCandidate || row.candidate_model === overlapCandidate)
    .filter(overlapDatasetFilter)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
}

function renderOverlapGroupChart() {
  const el = document.getElementById('overlap-group-chart');
  if (!overlapGroupChart) overlapGroupChart = echarts.init(el);
  overlapGroupChart.clear();
  const rows = selectedGroupDeltas().slice(0, 12).sort((a, b) => a.delta - b.delta);
  if (!rows.length) {
    overlapGroupChart.setOption({ title: { text: 'No subclass deltas', left: 'center', top: 'center', textStyle: { color: '#94a3b8', fontSize: 13 } } });
    return;
  }
  overlapGroupChart.setOption({
    grid: { left: 170, right: 32, top: 12, bottom: 24 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', name: 'Δ' },
    yAxis: {
      type: 'category',
      data: rows.map(row => `${row.dataset}/${row.label}`),
      axisLabel: { fontSize: 10, width: 160, overflow: 'truncate' },
    },
    series: [{
      type: 'bar',
      data: rows.map(row => row.delta),
      itemStyle: { color: p => p.value >= 0 ? '#55A868' : '#C44E52' },
      label: { show: true, position: 'right', fontSize: 10, formatter: p => Number(p.value).toFixed(2) },
    }],
  });
}

function renderOverlapGroupTable() {
  const rows = selectedGroupDeltas().slice(0, 80);
  if (!rows.length) {
    document.getElementById('overlap-group-table').innerHTML = '<div class="empty-state">No subclass rows</div>';
    return;
  }
  let h = '<table class="overlap-table"><thead><tr>';
  ['Dataset', 'Candidate', 'Group', 'N', 'Base', 'Cand', 'Δ', 'Fix', 'Drop'].forEach((label, idx) => {
    h += `<th class="${idx < 3 ? 'left' : ''}">${label}</th>`;
  });
  h += '</tr></thead><tbody>';
  rows.forEach(row => {
    h += '<tr>';
    h += `<td class="left">${row.dataset}</td>`;
    h += `<td class="left">${row.candidate_model}</td>`;
    h += `<td class="left">${row.label}</td>`;
    h += `<td>${row.sample_count}</td>`;
    h += `<td>${fmtNum(row.baseline_score)}</td>`;
    h += `<td>${fmtNum(row.candidate_score)}</td>`;
    h += `<td class="${deltaClass(row.delta)}">${fmtNum(row.delta)}</td>`;
    h += `<td>${row.fixes}</td>`;
    h += `<td>${row.drops}</td>`;
    h += '</tr>';
  });
  h += '</tbody></table>';
  document.getElementById('overlap-group-table').innerHTML = h;
}

window.addEventListener('resize', () => {
  if (chartInstance) chartInstance.resize();
  if (overlapDatasetChart) overlapDatasetChart.resize();
  if (overlapGroupChart) overlapGroupChart.resize();
});
initTableSidebar();
initSidebar();
initConfigSidebar();
initOverlapControls();
renderTable();
</script>
</body>
</html>
"""


def generate_dashboard(loader: ResultLoader, output_path: str, include_overlap: bool = False):
    """Write a self-contained HTML dashboard to output_path."""
    data = export_data(loader, include_overlap=include_overlap)
    data_json = json.dumps(data, ensure_ascii=False)
    content = _TEMPLATE.replace('/**__DATA__**/', data_json)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Dashboard written to {output_path}')


def serve(loader: ResultLoader, port: int = 8890, include_overlap: bool = False):
    """Start an HTTP server that regenerates the dashboard on each request."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            data = export_data(loader, include_overlap=include_overlap)
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
    parser.add_argument('--include-overlap', action='store_true', help='Include slow row-level overlap analysis')
    args = parser.parse_args()

    loader = ResultLoader(args.work_dir)
    print(f'Discovered {len(loader.models)} models, {len(loader.benchmarks)} benchmarks')

    if args.output:
        generate_dashboard(loader, args.output, include_overlap=args.include_overlap)
    else:
        serve(loader, args.port, include_overlap=args.include_overlap)


if __name__ == '__main__':
    main()
