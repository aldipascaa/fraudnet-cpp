/* FraudNet console - vanilla JS, no dependencies. All dynamic text goes through textContent. */
(() => {
  'use strict';

  /* ------------------------------------------------------------------ helpers */
  const $ = (sel, root = document) => root.querySelector(sel);
  const SVG_NS = 'http://www.w3.org/2000/svg';

  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  function s(tag, attrs, ...children) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, v);
    for (const c of children.flat()) if (c) el.append(c.nodeType ? c : document.createTextNode(String(c)));
    return el;
  }

  const fmtPct = (x, d = 1) => (x * 100).toFixed(d) + '%';
  const fmt4 = (x) => Number(x).toFixed(4);
  const fmtInt = (x) => Number(x).toLocaleString('en-US');
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  let toastTimer = 0;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
  }

  async function api(path, body) {
    let res;
    try {
      res = await fetch(path, body === undefined ? {} : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
    } catch (e) {
      const err = new Error('Cannot reach the FraudNet server. Is `fraudnet serve` still running?');
      err.network = true;
      throw err;
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error */ }
    if (!res.ok) {
      const err = new Error((data && data.message) || `Request failed (${res.status})`);
      err.details = (data && data.details) || [];
      err.status = res.status;
      throw err;
    }
    return data;
  }

  /* ------------------------------------------------------------------ theme */
  const root = document.documentElement;
  function effectiveTheme() {
    const t = root.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function syncThemeButton() {
    const dark = effectiveTheme() === 'dark';
    const btn = $('#theme-toggle');
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  }
  try {
    const saved = localStorage.getItem('fraudnet-theme');
    if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved);
  } catch { /* storage unavailable */ }
  $('#theme-toggle').addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('fraudnet-theme', next); } catch { /* ignore */ }
    syncThemeButton();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', syncThemeButton);
  syncThemeButton();

  /* ------------------------------------------------------------------ constants */
  const LABELS = {
    amount: 'Amount', hour: 'Hour of day', merchant_category: 'Merchant category',
    distance_from_home_km: 'Distance from home', distance_from_last_txn_km: 'Distance from last txn',
    txn_count_1h: 'Txns in last hour', txn_count_24h: 'Txns in last 24 h', amount_vs_30d_avg: 'Amount vs 30-day avg',
    avg_amount_30d: '30-day average amount', card_present: 'Card present', is_foreign: 'Foreign merchant',
    account_age_days: 'Account age',
  };
  function fmtValue(feature, v) {
    switch (feature) {
      case 'amount': case 'avg_amount_30d': return '$' + Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      case 'hour': return String(v).padStart(2, '0') + ':00';
      case 'distance_from_home_km': case 'distance_from_last_txn_km': return Number(v).toLocaleString('en-US', { maximumFractionDigits: 1 }) + ' km';
      case 'amount_vs_30d_avg': return Number(v).toFixed(2) + 'x avg';
      case 'account_age_days': return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 }) + ' days';
      case 'card_present': case 'is_foreign': return v ? 'yes' : 'no';
      case 'merchant_category': return String(v).replace('_', ' ');
      default: return String(v);
    }
  }

  const DEFAULTS = {
    amount: 60, hour: 14, merchant_category: 'grocery', distance_from_home_km: 5, distance_from_last_txn_km: 3,
    txn_count_1h: 0, txn_count_24h: 2, avg_amount_30d: 60, card_present: true, is_foreign: false, account_age_days: 700,
  };
  const PRESETS = [
    { name: 'Everyday grocery', tx: { amount: 54.2, hour: 17, merchant_category: 'grocery', distance_from_home_km: 3, distance_from_last_txn_km: 2, txn_count_1h: 0, txn_count_24h: 2, avg_amount_30d: 62, card_present: true, is_foreign: false, account_age_days: 900 } },
    { name: 'Holiday flight', tx: { amount: 780, hour: 11, merchant_category: 'travel', distance_from_home_km: 900, distance_from_last_txn_km: 12, txn_count_1h: 0, txn_count_24h: 1, avg_amount_30d: 140, card_present: true, is_foreign: true, account_age_days: 1200 } },
    { name: 'Late-night electronics', tx: { amount: 1450, hour: 3, merchant_category: 'electronics', distance_from_home_km: 640, distance_from_last_txn_km: 380, txn_count_1h: 3, txn_count_24h: 6, avg_amount_30d: 70, card_present: false, is_foreign: true, account_age_days: 25 } },
    { name: 'Card testing burst', tx: { amount: 2.4, hour: 2, merchant_category: 'online_services', distance_from_home_km: 12, distance_from_last_txn_km: 9, txn_count_1h: 6, txn_count_24h: 9, avg_amount_30d: 55, card_present: false, is_foreign: false, account_age_days: 200 } },
  ];

  /* slider config: `log` sliders map 0..1000 to 0..max on a log scale so small values stay adjustable */
  const FIELD_UI = [
    { key: 'amount', unit: '$', smax: 5000, log: true, decimals: 2, full: true, hint: 'transaction value' },
    { key: 'hour', smax: 23, step: 1, hint: '0-23' },
    { key: 'merchant_category', type: 'select' },
    { key: 'distance_from_home_km', unit: 'km', smax: 5000, log: true, decimals: 1 },
    { key: 'distance_from_last_txn_km', unit: 'km', smax: 5000, log: true, decimals: 1 },
    { key: 'txn_count_1h', smax: 15, step: 1, hint: 'previous txns' },
    { key: 'txn_count_24h', smax: 40, step: 1, hint: 'previous txns' },
    { key: 'avg_amount_30d', unit: '$', smax: 3000, log: true, decimals: 2 },
    { key: 'account_age_days', unit: 'days', smax: 7300, log: true, decimals: 0 },
    { key: 'card_present', type: 'bool', label: 'Card physically present' },
    { key: 'is_foreign', type: 'bool', label: 'Foreign merchant' },
  ];

  let schema = [];          // from /api/v1/model
  let modelInfo = null;
  const state = { ...DEFAULTS };
  const inputs = {};        // key -> {num, range, err}

  /* ------------------------------------------------------------------ analyze view */
  function logToValue(sv, ui) {
    const v = Math.expm1((sv / 1000) * Math.log1p(ui.smax));
    const f = Math.pow(10, ui.decimals);
    return Math.round(v * f) / f;
  }
  function valueToLog(v, ui) { return Math.round((Math.log1p(Math.max(0, v)) / Math.log1p(ui.smax)) * 1000); }

  function fieldSpec(key) { return schema.find((f) => f.name === key) || {}; }

  function buildForm() {
    const host = $('#fields');
    host.replaceChildren();
    for (const ui of FIELD_UI) {
      const spec = fieldSpec(ui.key);
      const id = 'f-' + ui.key;
      const label = ui.label || LABELS[ui.key];
      if (ui.type === 'select') {
        const sel = h('select', { id, 'aria-describedby': id + '-err' },
          (spec.values || ['grocery']).map((v) => h('option', { value: v, text: v.replace('_', ' ') })));
        sel.addEventListener('change', () => { state[ui.key] = sel.value; schedule(); });
        inputs[ui.key] = { sel };
        host.append(h('div', { class: 'field' }, h('label', { for: id }, label), sel));
      } else if (ui.type === 'bool') {
        const cb = h('input', { type: 'checkbox', id, role: 'switch' });
        cb.addEventListener('change', () => { state[ui.key] = cb.checked; schedule(); });
        inputs[ui.key] = { cb };
        host.append(h('div', { class: 'field' }, h('label', { class: 'switch-row', for: id }, h('span', { text: label }), h('span', { class: 'switch' }, cb, h('i')))));
      } else {
        const min = spec.min ?? 0, max = spec.max ?? 1e7;
        const num = h('input', { type: 'number', id, inputmode: 'decimal', min, max, step: ui.step ?? 'any', 'aria-describedby': id + '-err' });
        const range = ui.log
          ? h('input', { type: 'range', min: 0, max: 1000, step: 1, 'aria-label': label + ' slider', tabindex: '-1' })
          : h('input', { type: 'range', min, max: ui.smax, step: ui.step ?? 1, 'aria-label': label + ' slider', tabindex: '-1' });
        range.removeAttribute('tabindex');
        const err = h('div', { class: 'field-error', id: id + '-err', 'aria-live': 'polite' });
        num.addEventListener('input', () => {
          const raw = num.value.trim();
          const v = Number(raw);
          state[ui.key] = raw === '' ? NaN : v;
          range.value = ui.log ? valueToLog(v, ui) : v;
          schedule();
        });
        range.addEventListener('input', () => {
          const v = ui.log ? logToValue(Number(range.value), ui) : Number(range.value);
          state[ui.key] = v;
          num.value = v;
          schedule();
        });
        inputs[ui.key] = { num, range, err };
        host.append(h('div', { class: 'field' + (ui.full ? ' full' : '') },
          h('label', { for: id }, h('span', { text: label + (ui.unit ? ` (${ui.unit})` : '') }), ui.hint ? h('span', { class: 'hint', text: ui.hint }) : null),
          h('div', { class: 'field-row' }, range, num), err));
      }
    }
    syncInputs();
  }

  function syncInputs() {
    for (const ui of FIELD_UI) {
      const el = inputs[ui.key];
      const v = state[ui.key];
      if (el.sel) el.sel.value = v;
      else if (el.cb) el.cb.checked = !!v;
      else {
        el.num.value = Number.isNaN(v) ? '' : v;
        el.range.value = ui.log ? valueToLog(v, ui) : v;
      }
    }
  }

  function validateClient() {
    let ok = true;
    for (const ui of FIELD_UI) {
      const el = inputs[ui.key];
      if (!el.num) continue;
      const spec = fieldSpec(ui.key);
      const v = state[ui.key];
      let msg = '';
      if (!Number.isFinite(v)) msg = 'Enter a number';
      else if (spec.min !== undefined && v < spec.min) msg = `Must be at least ${spec.min}`;
      else if (spec.max !== undefined && v > spec.max) msg = `Must be at most ${fmtInt(spec.max)}`;
      else if (spec.type === 'integer' && !Number.isInteger(v)) msg = 'Must be a whole number';
      el.num.setAttribute('aria-invalid', msg ? 'true' : 'false');
      el.err.textContent = msg;
      if (msg) ok = false;
    }
    return ok;
  }

  let timer = 0, seq = 0;
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(predictNow, 120);
  }

  async function predictNow() {
    if (!validateClient()) { setIdle('Fix the highlighted fields'); return; }
    const mine = ++seq;
    try {
      const r = await api('/api/v1/predict', { ...state });
      if (mine !== seq) return;
      $('#global-error').hidden = true;
      renderResult(r);
    } catch (e) {
      if (mine !== seq) return;
      if (e.network) showGlobalError(e.message);
      else setIdle(e.details && e.details.length ? e.details[0] : e.message);
    }
  }

  function showGlobalError(msg) {
    const b = $('#global-error');
    b.textContent = msg;
    b.hidden = false;
  }

  function setIdle(msg) {
    $('#verdict').dataset.state = 'idle';
    $('#verdict-text').textContent = msg;
  }

  function placeThreshold(thr) {
    const g = $('#gauge-threshold');
    g.setAttribute('transform', `translate(130 130) rotate(${-90 + thr * 180})`);
  }

  function renderResult(r) {
    const p = r.probability;
    const pct = p * 100;
    $('#gauge-value').setAttribute('stroke-dasharray', `${Math.max(p * 100, p > 0 ? 0.6 : 0)} 100`);
    const text = pct < 0.1 ? '<0.1%' : pct > 99.9 ? '>99.9%' : pct.toFixed(1) + '%';
    $('#gauge-text').textContent = text;
    $('#gauge-sub').textContent = `decision threshold ${fmtPct(r.threshold, 0)}`;
    const gauge = $('#gauge');
    gauge.setAttribute('aria-valuenow', pct.toFixed(1));
    gauge.setAttribute('aria-valuetext', `${text} fraud probability, ${r.label}`);
    placeThreshold(r.threshold);

    const v = $('#verdict');
    v.dataset.state = r.is_fraud ? 'fraud' : 'legit';
    $('#verdict-text').textContent = r.is_fraud ? 'Flagged as likely fraud' : 'Looks legitimate';
    const risk = { high: 'High risk: above the decision threshold.', medium: 'Elevated: below the threshold but worth a glance.', low: 'Low risk.' }[r.risk_level] || '';
    $('#verdict-detail').textContent = risk;

    const maxAbs = Math.max(1, ...r.top_features.map((f) => Math.abs(f.contribution)));
    const list = $('#explain');
    list.replaceChildren(...r.top_features.map((f) => {
      const pos = f.contribution >= 0;
      const w = clamp(Math.abs(f.contribution) / maxAbs, 0, 1) * 50;
      const bar = h('span', { class: 'ex-bar ' + (pos ? 'pos' : 'neg'), style: pos ? `width:${w}%` : `width:${w}%;left:${50 - w}%` });
      return h('li', { class: 'ex-row' },
        h('div', { class: 'ex-name' }, LABELS[f.feature] || f.feature, h('small', { text: fmtValue(f.feature, f.value) })),
        h('div', { class: 'ex-track', 'aria-hidden': 'true' }, bar),
        h('div', { class: 'ex-val ' + (pos ? 'pos' : 'neg'), 'aria-label': `${pos ? 'raises' : 'lowers'} risk by ${Math.abs(f.contribution).toFixed(2)}`, text: (pos ? '+' : '-') + Math.abs(f.contribution).toFixed(2) }));
    }));
  }

  function applyPreset(tx) {
    Object.assign(state, tx);
    syncInputs();
    predictNow();
  }

  function initAnalyze() {
    const row = $('#preset-row');
    for (const p of PRESETS) row.append(h('button', { type: 'button', class: 'chip', onclick: () => applyPreset(p.tx), text: p.name }));
    $('#reset-btn').addEventListener('click', () => applyPreset({ ...DEFAULTS }));
    $('#copy-json-btn').addEventListener('click', async () => {
      const text = JSON.stringify(state, null, 2);
      try { await navigator.clipboard.writeText(text); toast('Copied JSON to clipboard'); }
      catch {
        const ta = h('textarea', { 'aria-hidden': 'true', style: 'position:fixed;opacity:0' });
        ta.value = text; document.body.append(ta); ta.select();
        try { document.execCommand('copy'); toast('Copied JSON to clipboard'); } catch { toast('Copy failed'); }
        ta.remove();
      }
    });
    $('#tx-form').addEventListener('submit', (e) => e.preventDefault());
    buildForm();
    $('#csv-cols').textContent = schema.map((f) => f.name).join(', ');
    placeThreshold(modelInfo ? modelInfo.threshold : 0.5);
    predictNow();
  }

  /* ------------------------------------------------------------------ charts */
  function niceTicks(max, n = 4) {
    const step = max / n;
    return Array.from({ length: n + 1 }, (_, i) => +(step * i).toFixed(4));
  }

  function lineChart(o) {
    const W = 400, H = 300, m = { l: 48, r: 14, t: 12, b: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const [x0, x1] = o.xDomain, [y0, y1] = o.yDomain;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * iw;
    const Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.label });
    svg.append(s('title', {}, o.label));
    for (const t of o.yTicks) {
      svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }));
      svg.append(s('text', { class: 'axis-text', x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, o.yFmt(t)));
    }
    for (const t of o.xTicks) {
      svg.append(s('line', { class: 'grid', x1: X(t), x2: X(t), y1: m.t, y2: m.t + ih }));
      svg.append(s('text', { class: 'axis-text', x: X(t), y: m.t + ih + 16, 'text-anchor': 'middle' }, o.xFmt(t)));
    }
    svg.append(s('text', { class: 'axis-title', x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, o.xTitle));
    svg.append(s('text', { class: 'axis-title', transform: `translate(13 ${m.t + ih / 2}) rotate(-90)`, 'text-anchor': 'middle' }, o.yTitle));
    if (o.diag) svg.append(s('line', { class: 'diag', x1: X(0), y1: Y(0), x2: X(1), y2: Y(1) }));
    if (o.hline !== undefined) svg.append(s('line', { class: 'diag', x1: X(x0), x2: X(x1), y1: Y(o.hline), y2: Y(o.hline) }));
    o.series.forEach((ser) => {
      const pts = ser.x.map((xv, i) => `${X(xv).toFixed(1)},${Y(ser.y[i]).toFixed(1)}`);
      if (ser.area) svg.append(s('polygon', { class: 'area', points: `${X(ser.x[0])},${Y(y0)} ${pts.join(' ')} ${X(ser.x[ser.x.length - 1])},${Y(y0)}` }));
      svg.append(s('polyline', { class: 'line' + (ser.alt ? ' alt' : ''), points: pts.join(' ') }));
    });
    if (o.marker) svg.append(s('circle', { class: 'dot', cx: X(o.marker.x), cy: Y(o.marker.y), r: 5 }, s('title', {}, o.marker.title)));
    return svg;
  }

  /* ------------------------------------------------------------------ model view */
  function statBox(k, v, sub) {
    return h('div', { class: 'stat' }, h('div', { class: 'k', text: k }), h('div', { class: 'v', text: v }), sub ? h('div', { class: 's', text: sub }) : null);
  }

  function renderConfusion(cm) {
    const total = cm.tp + cm.fp + cm.tn + cm.fn;
    const pos = cm.tp + cm.fn, neg = cm.tn + cm.fp;
    const cell = (name, n, rowTotal, good, desc) => {
      const frac = rowTotal ? n / rowTotal : 0;
      const a = good ? 0.12 + 0.4 * frac : Math.min(0.6, 0.08 + 2.2 * frac);
      return h('div', { class: 'cm-cell ' + (good ? 'cm-good' : 'cm-bad'), style: `--a:${a.toFixed(2)}`, role: 'group', 'aria-label': `${name}: ${n} transactions, ${desc}` },
        h('em', { text: name }), h('b', { text: fmtInt(n) }), h('span', { text: fmtPct(rowTotal ? n / rowTotal : 0, 1) + ' of ' + desc }));
    };
    $('#cm').replaceChildren(h('div', { class: 'cm' },
      h('div'), h('div', { class: 'hd', text: 'Predicted legit' }), h('div', { class: 'hd', text: 'Predicted fraud' }),
      h('div', { class: 'rl', text: 'Actual legit' }), cell('True negative', cm.tn, neg, true, 'legit'), cell('False positive', cm.fp, neg, false, 'legit'),
      h('div', { class: 'rl', text: 'Actual fraud' }), cell('False negative', cm.fn, pos, false, 'fraud'), cell('True positive', cm.tp, pos, true, 'fraud')),
    h('p', { class: 'muted small', text: `${fmtInt(total)} held-out transactions (${fmtInt(pos)} fraud).` }));
  }

  function renderModel() {
    const info = modelInfo;
    const t = info.metrics.test;
    $('#model-lede').textContent = `Evaluated on ${fmtInt(t.samples)} held-out transactions never used for training or threshold tuning.`;
    $('#stat-grid').replaceChildren(
      statBox('ROC-AUC', fmt4(t.roc_auc)), statBox('PR-AUC', fmt4(t.pr_auc), 'avg. precision'),
      statBox('F1 score', fmt4(t.f1), 'at tuned threshold'), statBox('Precision', fmtPct(t.precision), 'flags that are fraud'),
      statBox('Recall', fmtPct(t.recall), 'fraud caught'), statBox('Threshold', t.threshold.toFixed(3), 'max-F1 on validation'));
    renderConfusion(t.confusion_matrix);

    const roc = info.metrics.roc_curve;
    const cm = t.confusion_matrix;
    const fpr = cm.fp / Math.max(1, cm.fp + cm.tn);
    $('#roc').replaceChildren(
      lineChart({ label: `ROC curve, area under curve ${fmt4(t.roc_auc)}`, xDomain: [0, 1], yDomain: [0, 1], xTicks: niceTicks(1), yTicks: niceTicks(1), xFmt: (v) => v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') || '0', yFmt: (v) => v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') || '0', xTitle: 'False positive rate', yTitle: 'True positive rate', diag: true, series: [{ x: roc.fpr, y: roc.tpr, area: true }], marker: { x: fpr, y: t.recall, title: `Operating point: FPR ${fmtPct(fpr, 2)}, TPR ${fmtPct(t.recall)}` } }),
      h('p', { class: 'caption', text: `AUC ${fmt4(t.roc_auc)}. Red dot = deployed threshold ${t.threshold.toFixed(3)}.` }));

    const pr = info.metrics.pr_curve;
    const prevalence = t.positives / t.samples;
    $('#pr').replaceChildren(
      lineChart({ label: `Precision-recall curve, average precision ${fmt4(t.pr_auc)}`, xDomain: [0, 1], yDomain: [0, 1], xTicks: niceTicks(1), yTicks: niceTicks(1), xFmt: (v) => v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') || '0', yFmt: (v) => v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '') || '0', xTitle: 'Recall', yTitle: 'Precision', hline: prevalence, series: [{ x: pr.recall, y: pr.precision, area: true }], marker: { x: t.recall, y: t.precision, title: `Operating point: recall ${fmtPct(t.recall)}, precision ${fmtPct(t.precision)}` } }),
      h('p', { class: 'caption', text: `Average precision ${fmt4(t.pr_auc)}; dashed line = no-skill baseline (${fmtPct(prevalence, 2)} fraud).` }));

    const hist = info.training.history || [];
    if (hist.length > 1) {
      const maxL = Math.max(...hist.map((e) => Math.max(e.train_loss, e.val_loss)));
      const top = Math.ceil(maxL * 10) / 10;
      const ep = hist.map((e) => e.epoch);
      const best = info.training.best_epoch;
      const nTicks = Math.min(hist.length, 6);
      const xTicks = Array.from({ length: nTicks }, (_, i) => Math.round(1 + (i * (hist.length - 1)) / Math.max(1, nTicks - 1)));
      $('#loss').replaceChildren(
        lineChart({ label: 'Training and validation loss per epoch', xDomain: [1, hist.length], yDomain: [0, top], xTicks, yTicks: niceTicks(top), xFmt: (v) => String(v), yFmt: (v) => v.toFixed(2), xTitle: 'Epoch', yTitle: 'Weighted BCE loss', series: [{ x: ep, y: hist.map((e) => e.train_loss) }, { x: ep, y: hist.map((e) => e.val_loss), alt: true }], marker: hist[best - 1] ? { x: best, y: hist[best - 1].val_loss, title: `Best epoch ${best} (early stopping restores these weights)` } : null }),
        h('div', { class: 'legend-line' }, h('span', {}, h('i'), 'train'), h('span', {}, h('i', { class: 'alt' }), 'validation'), h('span', { text: `best epoch ${best} of ${hist.length}` })));
    }

    const base = info.metrics.baseline_logreg_test;
    const rows = [['ROC-AUC', 'roc_auc', fmt4], ['PR-AUC', 'pr_auc', fmt4], ['F1', 'f1', fmt4], ['Precision', 'precision', fmt4], ['Recall', 'recall', fmt4]];
    $('#cmp-table').replaceChildren(
      h('thead', {}, h('tr', {}, h('th', { scope: 'col', text: 'Metric' }), h('th', { scope: 'col', class: 'num', text: 'Logistic regression' }), h('th', { scope: 'col', class: 'num', text: 'Neural network' }), h('th', { scope: 'col', class: 'num', text: 'Delta' }))),
      h('tbody', {}, rows.map(([name, k, f]) => {
        const d = t[k] - base[k];
        return h('tr', {}, h('th', { scope: 'row', text: name }), h('td', { class: 'num', text: f(base[k]) }), h('td', { class: 'num', text: f(t[k]) }), h('td', { class: 'num', text: (d >= 0 ? '+' : '') + d.toFixed(4) }));
      })));

    const tr = info.training;
    const kv = [
      ['Model', `${info.name} v${info.version}`], ['Trained', info.created_at], ['Architecture', info.network.layer_sizes.join(' - ') + ' (ReLU, sigmoid output)'],
      ['Parameters', fmtInt(info.network.parameters)], ['Optimizer', `${tr.optimizer}, lr ${tr.learning_rate}, batch ${tr.batch_size}, L2 ${tr.l2}`],
      ['Class weight (fraud)', Number(tr.positive_class_weight).toFixed(2)], ['Training data', `${fmtInt(tr.rows)} rows, ${fmtPct(tr.fraud_rate, 2)} fraud`],
      ['Split', `${fmtInt(tr.train_rows)} train / ${fmtInt(tr.val_rows)} val / ${fmtInt(tr.test_rows)} test`], ['Epochs', `${tr.epochs_run} run, best ${tr.best_epoch} (early stopping)`],
      ['Features', `${info.features.length} engineered columns from ${info.feature_groups.length} inputs`],
    ];
    $('#model-info').replaceChildren(...kv.map(([k, v]) => h('div', {}, h('dt', { text: k }), h('dd', { text: v }))));
  }

  /* ------------------------------------------------------------------ batch view */
  const batch = { rows: [], sortKey: 'probability', sortDir: -1, page: 0, pageSize: 25, flaggedOnly: false, hasLabels: false };

  function parseCsv(text) {
    const rows = [];
    let row = [], field = '', inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; } else field += c;
      } else if (c === '"') inQ = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        if (row.some((x) => x.trim() !== '')) rows.push(row);
        row = [];
      } else field += c;
    }
    row.push(field);
    if (row.some((x) => x.trim() !== '')) rows.push(row);
    return rows;
  }

  function csvToTransactions(text) {
    const table = parseCsv(text.replace(/^﻿/, ''));
    if (table.length < 2) throw Object.assign(new Error('The CSV needs a header row and at least one data row.'), { details: [] });
    const header = table[0].map((x) => x.trim());
    const missing = schema.map((f) => f.name).filter((n) => !header.includes(n));
    if (missing.length) throw Object.assign(new Error('The CSV is missing required columns.'), { details: missing.map((m) => 'Missing column: ' + m) });
    const col = Object.fromEntries(header.map((n, i) => [n, i]));
    const problems = [], txs = [], labels = [];
    const hasLabel = 'is_fraud' in col;
    table.slice(1).forEach((r, i) => {
      const line = i + 2;
      const tx = {};
      for (const f of schema) {
        const raw = (r[col[f.name]] ?? '').trim();
        if (f.type === 'category') tx[f.name] = raw;
        else if (f.type === 'boolean') {
          if (['1', 'true', 'True', 'TRUE'].includes(raw)) tx[f.name] = true;
          else if (['0', 'false', 'False', 'FALSE'].includes(raw)) tx[f.name] = false;
          else if (problems.length < 12) problems.push(`Line ${line}: ${f.name} must be 0/1 (got "${raw}")`);
        } else {
          const n = raw === '' ? NaN : Number(raw);
          if (!Number.isFinite(n)) { if (problems.length < 12) problems.push(`Line ${line}: ${f.name} is not a number ("${raw}")`); } else tx[f.name] = n;
        }
      }
      txs.push(tx);
      if (hasLabel) labels.push(['1', 'true', 'True'].includes((r[col.is_fraud] ?? '').trim()) ? 1 : 0);
    });
    if (problems.length) throw Object.assign(new Error('Some CSV cells could not be read.'), { details: problems });
    return { txs, labels: hasLabel ? labels : null };
  }

  function showBatchError(e) {
    const box = $('#batch-error');
    box.replaceChildren(h('strong', { text: e.message }));
    if (e.details && e.details.length) box.append(h('ul', {}, e.details.slice(0, 12).map((d) => h('li', { text: d }))));
    box.hidden = false;
  }

  async function scoreBatch(txs, labels, name) {
    $('#batch-error').hidden = true;
    const card = $('#batch-results');
    try {
      const out = [];
      for (let off = 0; off < txs.length; off += 10000) {
        const chunk = txs.slice(off, off + 10000);
        const r = await api('/api/v1/predict/batch', { transactions: chunk });
        r.results.forEach((x) => out.push({ ...x, index: off + x.index }));
      }
      batch.rows = out.map((r, i) => ({ n: i + 1, tx: txs[i], probability: r.probability, is_fraud: r.is_fraud, top: r.top_features[0], actual: labels ? labels[i] : null }));
      batch.hasLabels = !!labels;
      batch.page = 0;
      batch.name = name;
      card.hidden = false;
      renderBatch();
      card.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    } catch (e) {
      if (e.network) showGlobalError(e.message); else showBatchError(e);
    }
  }

  function renderBatch() {
    const all = batch.rows;
    const flagged = all.filter((r) => r.is_fraud).length;
    const avg = all.reduce((a, r) => a + r.probability, 0) / Math.max(1, all.length);
    const stats = [statBox('Transactions', fmtInt(all.length)), statBox('Flagged', fmtInt(flagged), fmtPct(flagged / Math.max(1, all.length), 2) + ' of rows'), statBox('Mean probability', fmtPct(avg, 2))];
    if (batch.hasLabels) {
      const tp = all.filter((r) => r.is_fraud && r.actual === 1).length;
      const fp = all.filter((r) => r.is_fraud && r.actual === 0).length;
      const fn = all.filter((r) => !r.is_fraud && r.actual === 1).length;
      stats.push(statBox('Precision', fmtPct(tp / Math.max(1, tp + fp), 1), `${tp} true / ${fp} false alarms`), statBox('Recall', fmtPct(tp / Math.max(1, tp + fn), 1), `${fn} fraud missed`));
    }
    $('#batch-stats').replaceChildren(...stats);

    let view = batch.flaggedOnly ? all.filter((r) => r.is_fraud) : all.slice();
    const get = { n: (r) => r.n, amount: (r) => r.tx.amount, probability: (r) => r.probability, category: (r) => r.tx.merchant_category };
    const k = get[batch.sortKey] || get.probability;
    view.sort((a, b) => { const x = k(a), y = k(b); return (x < y ? -1 : x > y ? 1 : 0) * batch.sortDir; });
    const pages = Math.max(1, Math.ceil(view.length / batch.pageSize));
    batch.page = clamp(batch.page, 0, pages - 1);
    const slice = view.slice(batch.page * batch.pageSize, (batch.page + 1) * batch.pageSize);

    const th = (key, label, num) => {
      const active = batch.sortKey === key;
      const arrow = active ? (batch.sortDir < 0 ? ' ▼' : ' ▲') : '';
      return h('th', { scope: 'col', class: num ? 'num' : '', 'aria-sort': active ? (batch.sortDir < 0 ? 'descending' : 'ascending') : 'none' },
        h('button', { type: 'button', onclick: () => { if (batch.sortKey === key) batch.sortDir *= -1; else { batch.sortKey = key; batch.sortDir = key === 'category' ? 1 : -1; } renderBatch(); } }, label + arrow));
    };
    const head = h('tr', {}, th('n', '#', true), th('amount', 'Amount', true), th('category', 'Category'), h('th', { scope: 'col', class: 'num', text: 'Hour' }), h('th', { scope: 'col', class: 'num', text: 'Home km' }),
      th('probability', 'Fraud probability'), h('th', { scope: 'col', text: 'Verdict' }), h('th', { scope: 'col', text: 'Top signal' }), batch.hasLabels ? h('th', { scope: 'col', text: 'Actual' }) : null);
    const body = slice.map((r) => h('tr', { class: r.is_fraud ? 'flag' : '' },
      h('td', { class: 'num', text: r.n }), h('td', { class: 'num', text: r.tx.amount.toFixed(2) }), h('td', { text: r.tx.merchant_category.replace('_', ' ') }),
      h('td', { class: 'num', text: r.tx.hour }), h('td', { class: 'num', text: r.tx.distance_from_home_km.toFixed(1) }),
      h('td', {}, h('span', { class: 'mini-bar', 'aria-hidden': 'true' }, h('i', { style: `width:${(r.probability * 100).toFixed(1)}%` })), fmtPct(r.probability, 2)),
      h('td', {}, h('span', { class: 'tag ' + (r.is_fraud ? 'tag-fraud' : 'tag-legit'), text: r.is_fraud ? 'Fraud' : 'Legit' })),
      h('td', { text: r.top ? `${LABELS[r.top.feature] || r.top.feature} (${r.top.contribution >= 0 ? '+' : '-'}${Math.abs(r.top.contribution).toFixed(1)})` : '' }),
      batch.hasLabels ? h('td', {}, h('span', { class: 'tag ' + (r.actual ? 'tag-fraud' : 'tag-legit'), text: r.actual ? 'Fraud' : 'Legit' })) : null));
    $('#batch-table').replaceChildren(h('caption', { class: 'sr-only', text: 'Scored transactions' }), h('thead', {}, head), h('tbody', {}, body.length ? body : [h('tr', {}, h('td', { colspan: 9, text: 'No rows to show.' }))]));

    const prev = h('button', { type: 'button', class: 'btn btn-ghost', text: 'Previous', disabled: batch.page === 0, onclick: () => { batch.page--; renderBatch(); } });
    const next = h('button', { type: 'button', class: 'btn btn-ghost', text: 'Next', disabled: batch.page >= pages - 1, onclick: () => { batch.page++; renderBatch(); } });
    $('#pager').replaceChildren(h('span', { 'aria-live': 'polite', text: `Page ${batch.page + 1} of ${pages} (${fmtInt(view.length)} rows)` }), prev, next);
  }

  function downloadText(name, text, type = 'text/csv') {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = h('a', { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exampleRows() {
    let seed = 7;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
    const out = [];
    for (let rep = 0; rep < 4; rep++) {
      for (const p of PRESETS) {
        const tx = { ...p.tx };
        tx.amount = +(tx.amount * (0.7 + rnd() * 0.6)).toFixed(2);
        tx.hour = clamp(Math.round(tx.hour + (rnd() - 0.5) * 4), 0, 23);
        tx.distance_from_home_km = +(tx.distance_from_home_km * (0.6 + rnd() * 0.8)).toFixed(1);
        out.push(tx);
      }
    }
    return out;
  }
  const toCsv = (txs) => {
    const cols = schema.map((f) => f.name);
    const cell = (v) => (typeof v === 'boolean' ? (v ? 1 : 0) : v);
    return [cols.join(','), ...txs.map((t) => cols.map((c) => cell(t[c])).join(','))].join('\n') + '\n';
  };

  function initBatch() {
    const dz = $('#dropzone'), input = $('#csv-input');
    const handleFile = async (file) => {
      if (!file) return;
      if (file.size > 6 * 1024 * 1024) { showBatchError(new Error('File is larger than 6 MB.')); return; }
      try {
        const { txs, labels } = csvToTransactions(await file.text());
        if (txs.length > 50000) throw new Error('Please upload at most 50,000 rows.');
        await scoreBatch(txs, labels, file.name);
      } catch (e) { showBatchError(e); }
    };
    input.addEventListener('change', () => { handleFile(input.files[0]); input.value = ''; });
    ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
    dz.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
    $('#example-btn').addEventListener('click', () => scoreBatch(exampleRows(), null, 'example'));
    $('#template-btn').addEventListener('click', () => downloadText('fraudnet_template.csv', toCsv(exampleRows().slice(0, 3))));
    $('#flagged-only').addEventListener('change', (e) => { batch.flaggedOnly = e.target.checked; batch.page = 0; renderBatch(); });
    $('#download-btn').addEventListener('click', () => {
      const cols = schema.map((f) => f.name);
      const lines = [[...cols, 'fraud_probability', 'predicted_label'].join(',')];
      for (const r of batch.rows) lines.push([...cols.map((c) => (typeof r.tx[c] === 'boolean' ? +r.tx[c] : r.tx[c])), r.probability.toFixed(6), r.is_fraud ? 'fraud' : 'legit'].join(','));
      downloadText('fraudnet_results.csv', lines.join('\n') + '\n');
    });
  }

  /* ------------------------------------------------------------------ routing */
  const VIEWS = ['analyze', 'model', 'batch'];
  const TITLES = { analyze: 'Analyze', model: 'Model', batch: 'Batch' };
  function route() {
    const m = /^#\/(\w+)/.exec(location.hash);
    const view = m && VIEWS.includes(m[1]) ? m[1] : 'analyze';
    for (const v of VIEWS) {
      $('#view-' + v).hidden = v !== view;
      const tab = $('#tab-' + v);
      tab.setAttribute('aria-selected', v === view ? 'true' : 'false');
      tab.tabIndex = v === view ? 0 : -1;
    }
    document.title = `${TITLES[view]} - FraudNet`;
    if (view === 'batch' && location.hash.startsWith('#/batch/example') && schema.length && !batch.rows.length) scoreBatch(exampleRows(), null, 'example');
  }
  window.addEventListener('hashchange', route);
  $('.tabs').addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const idx = VIEWS.findIndex((v) => $('#tab-' + v).getAttribute('aria-selected') === 'true');
    const next = VIEWS[(idx + (e.key === 'ArrowRight' ? 1 : VIEWS.length - 1)) % VIEWS.length];
    location.hash = '#/' + next;
    $('#tab-' + next).focus();
    e.preventDefault();
  });

  /* ------------------------------------------------------------------ boot */
  async function boot() {
    route();
    try {
      const [health, info] = await Promise.all([api('/api/v1/health'), api('/api/v1/model')]);
      modelInfo = info;
      schema = info.input_schema;
      const pill = $('#version-pill');
      pill.textContent = 'v' + health.version;
      pill.hidden = false;
      initAnalyze();
      initBatch();
      renderModel();
      route();
    } catch (e) {
      showGlobalError(e.message);
    }
  }
  boot();
})();
