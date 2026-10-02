// Campfire: app wiring — rigid bodies (CPU), wood thermal model (CPU),
// gas + radiation (GPU), rendering and UI.
import { RigidWorld, Body } from './rigid.js';
import { WoodPiece } from './wood.js';
import { exchangeContacts } from './contact.js';
import { FluidSim, MAX_PIECES, PIECE_FLOATS, ACC_GLOBAL, E_SCALE, M_SCALE, G_SCALE, N_FLOOR } from './fluid.js';
import { Renderer, VIEWS, GROUND } from './render.js';
import { MODES, PHYS, SCENE_WIDTH, CONTROL_DEFAULTS } from './config.js';

const $ = id => document.getElementById(id);
const canvas = $('view');

// ---------------- controls ----------------
const CONTROLS = [
  { group: 'Gas', key: 'viscMult', label: 'Viscosity', min: 0.25, max: 4, log: true },
  { group: 'Gas', key: 'diffMult', label: 'Heat & species diffusivity', min: 0.25, max: 4, log: true },
  { group: 'Combustion', key: 'reactMult', label: 'Gas reaction rate', min: 0.01, max: 100, log: true },
  { group: 'Combustion', key: 'sootMult', label: 'Soot formation', min: 0.1, max: 10, log: true },
  { group: 'Wood', key: 'pyroMult', label: 'Pyrolysis rate', min: 0.1, max: 10, log: true },
  { group: 'Wood', key: 'charMult', label: 'Char burning rate', min: 0.1, max: 10, log: true },
  { group: 'Wood', key: 'charYield', label: 'Char yield', min: 0.1, max: 0.4, step: 0.01, fmt: v => Math.round(v * 100) + '%' },
  { group: 'Wood', key: 'woodDensity', label: 'Density of new wood', min: 300, max: 800, step: 10, fmt: v => v + ' kg/m³' },
  { group: 'Wood', key: 'friction', label: 'Friction', min: 0.2, max: 1.0, step: 0.05, fmt: v => 'μ ' + v.toFixed(2) },
  { group: 'Air', key: 'ambientO2', label: 'Ambient O₂', min: 12, max: 30, step: 0.1, fmt: v => v.toFixed(1) + '%' },
  { group: 'Air', key: 'depth', label: 'Fire depth', min: 0.2, max: 3, step: 0.05, fmt: v => v.toFixed(2) + ' m' },
];
const ctl = { ...CONTROL_DEFAULTS };

function buildControls() {
  const host = $('controls');
  let group = '';
  for (const c of CONTROLS) {
    if (c.group !== group) {
      group = c.group;
      const h = document.createElement('h3'); h.textContent = group; host.appendChild(h);
    }
    const row = document.createElement('div'); row.className = 'ctl';
    const name = document.createElement('span'); name.className = 'name'; name.textContent = c.label;
    const val = document.createElement('span'); val.className = 'val';
    const inp = document.createElement('input'); inp.type = 'range';
    if (c.log) { inp.min = Math.log10(c.min); inp.max = Math.log10(c.max); inp.step = 0.01; }
    else { inp.min = c.min; inp.max = c.max; inp.step = c.step; }
    const fmt = c.fmt || (v => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toPrecision(2)) + '×');
    c.sync = () => {
      inp.value = c.log ? Math.log10(ctl[c.key]) : ctl[c.key];
      val.textContent = fmt(ctl[c.key]);
    };
    inp.addEventListener('input', () => {
      ctl[c.key] = c.log ? Math.pow(10, +inp.value) : +inp.value;
      val.textContent = fmt(ctl[c.key]);
      applyControls();
    });
    row.append(name, val, inp);
    host.appendChild(row);
    c.sync();
  }
}

// ---------------- state ----------------
let device, sim, ren;
const world = new RigidWorld({ width: SCENE_WIDTH });
let pieces = [];                         // {body, wood, slot, qConv[4], qInc[4], vis[4], charPending[4]}
const slots = new Array(MAX_PIECES).fill(null);
const slotFree = new Array(MAX_PIECES).fill(0);   // frame index when the slot may be reused
let frameNo = 0;
let lastRefuel = 0;
const stats = { hrr: 0, Tmax: 0, simRatio: 1 };
const view = { mode: 'photo', exposure: 0 };
// camera (debug/inspection): ?zoom=3&cx=0.35&cy=0.1 centres the view on (cx, cy)
const camera = (() => {
  const q = new URLSearchParams(location.search);
  const zoom = +(q.get('zoom') || 1);
  return { zoom, cx: +(q.get('cx') || SCENE_WIDTH / 2), cy: q.get('cy') };
})();
function cam() {
  const H = sim.height + GROUND;
  if (camera.zoom === 1) return { x: 0, y: -GROUND, zoom: 1 };
  const cy = camera.cy !== null ? +camera.cy : H / 2;
  return { x: camera.cx - SCENE_WIDTH / 2 / camera.zoom, y: cy - H / 2 / camera.zoom, zoom: camera.zoom };
}

function applyControls() {
  if (!sim) return;
  const X = ctl.ambientO2 / 100;
  const YO = 32 * X / (32 * X + 28.01 * (1 - X));
  sim.setParam('YOamb', YO);
  sim.setParam('viscMult', ctl.viscMult);
  sim.setParam('diffMult', ctl.diffMult);
  sim.setParam('reactB', PHYS.reactB * ctl.reactMult);
  sim.setParam('sootFormA', PHYS.sootFormA * ctl.sootMult);
  sim.setParam('charA', PHYS.charA * ctl.charMult);
  sim.setParam('depth', ctl.depth);
  world.p.muStatic = ctl.friction;
  world.p.muKinetic = 0.75 * ctl.friction;
}

// ---------------- pieces ----------------
function allocSlot() {
  for (let s = 0; s < MAX_PIECES; s++) if (!slots[s] && slotFree[s] <= frameNo) return s;
  return -1;
}

function addPiece(x, y, side, opts = {}) {
  const slot = allocSlot();
  if (slot < 0) return null;
  const wood = new WoodPiece(side, opts.density || ctl.woodDensity);
  if (opts.preheat) wood.preheat(...opts.preheat, ctl.charYield);
  if (opts.faces) opts.faces.forEach((T, f) => { if (T) { wood.Ts[f] = T; wood.Tp[f] = Math.max(wood.Tp[f], 0.5 * (T + wood.Tc)); } });
  const body = new Body(x, y, wood.half, (wood.mv + wood.mc) / (wood.side * wood.side), opts.angle || 0);
  world.add(body);
  const p = {
    body, wood, slot,
    qConv: [0, 0, 0, 0], qInc: [0, 0, 0, 0].map(() => PHYS.sigma * PHYS.Tamb ** 4),
    vis: [0, 0, 0, 0], charPending: [0, 0, 0, 0],
  };
  body.userData = p;
  slots[slot] = p;
  pieces.push(p);
  return p;
}

function removePiece(p) {
  world.remove(p.body);
  slots[p.slot] = null;
  slotFree[p.slot] = frameNo + 6;     // let in-flight readbacks for this slot drain
  pieces = pieces.filter(q => q !== p);
}

function clearWood() { for (const p of [...pieces]) removePiece(p); }

// Pre-lit fire: a bed of glowing coals, two logs flanking it, sticks piled on it.
function defaultScene() {
  clearWood();
  const cx = SCENE_WIDTH / 2, S = 0.0125;
  const coal = (x, y, sz, T) => {
    const p = addPiece(x, y, sz, { preheat: [T, sz, T - 30] });
    const w = p.wood;
    w.mv = 0; w.mc = w.rhoC * sz * sz; w.updateSide();
    w.d = [sz / 2, sz / 2, sz / 2, sz / 2]; w.charCov = [1, 1, 1, 1];
    w.Tp = [T - 20, T - 20, T - 20, T - 20];
    p.body.density = w.mc / (sz * sz);
    world.resize(p.body, w.half);
  };
  const lit = (x, y, sz) => {
    const p = addPiece(x, y, sz, { preheat: [880, sz > 0.03 ? 0.003 : 0.002, 380] });
    p.wood.Tp = [590, 590, 590, 590];
  };
  let x = cx - 0.06;
  for (const sz of [0.016, 0.02, 0.015, 0.018, 0.02, 0.016, 0.017]) { coal(x + sz / 2, sz / 2, sz, 1050); x += sz + 0.0015; }
  lit(cx - 0.093, 0.025, 0.05); lit(cx + 0.093, 0.025, 0.05);
  const y1 = 0.02 + S + 0.001;
  for (const dx of [-0.045, -0.016, 0.013, 0.042]) lit(cx + dx, y1, 0.025);
  for (const dx of [-0.03, 0.0, 0.028]) lit(cx + dx, y1 + 2 * S + 0.002, 0.025);
  lastRefuel = sim ? sim.time : 0;
}

// ---------------- GPU ----------------
async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Use a recent Chrome or Edge.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  if (adapter.limits.maxStorageBuffersPerShaderStage < 10) throw new Error('This GPU exposes too few storage buffers (need 10).');
  device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(12, adapter.limits.maxStorageBuffersPerShaderStage) },
  });
  device.lost.then(info => showMsg('The GPU device was lost (' + info.message + '). Reload the page.'));
  device.addEventListener('uncapturederror', e => console.error('GPU', e.error.message));
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  ren = new Renderer(device, ctx, format);
}

async function buildSim(quality) {
  const mode = MODES[quality];
  const s = new FluidSim(device, mode);
  await s.init();
  sim = s;
  applyControls();
  await ren.init(sim);
  resize();
}

function showMsg(t) { const m = $('msg'); m.textContent = t; m.hidden = !t; }

// ---------------- frame ----------------
const pieceData = new Float32Array(MAX_PIECES * PIECE_FLOATS);
const woodData = new Float32Array(MAX_PIECES * 12);

function uploadPieces() {
  pieceData.fill(0);
  woodData.fill(0);
  let n = 0;
  for (const p of pieces) {
    const b = p.body, w = p.wood, o = p.slot * PIECE_FLOATS;
    pieceData.set([b.x, b.y, b.a, b.half, b.vx, b.vy, b.w, 1], o);
    pieceData.set(w.Ts, o + 8);
    pieceData.set(w.mflux, o + 12);
    pieceData.set(w.charCov, o + 16);
    const r = p.slot * 12;
    woodData.set(w.Tp, r);
    woodData.set(w.d, r + 4);
    woodData.set([w.charFraction(), w.Tc, (p.vis[0] + p.vis[1] + p.vis[2] + p.vis[3]) / 4, w.mv > 0 ? 1 : 0], r + 8);
    n = Math.max(n, p.slot + 1);
  }
  sim.uploadPieces(pieceData, n);
  ren.writeWood(woodData);
}

function onReadback({ acc, rad, faceW }, tag) {
  if (tag.simDt <= 0) return;
  for (let s = 0; s < tag.owners.length; s++) {
    const p = tag.owners[s];
    if (!p || slots[s] !== p) continue;
    for (let f = 0; f < 4; f++) {
      // fraction of the face open to the gas (0 where pressed against wood or floor)
      p.wood.exposure[f] = Math.min(1, faceW[s * 4 + f] / 1e8 / Math.max(p.wood.side, 1e-4));
      const k = (s * 4 + f) * 2;
      p.qConv[f] = acc[k] / E_SCALE / tag.simDt;
      p.charPending[f] += acc[k + 1] / M_SCALE;
      p.qInc[f] = rad[(s * 4 + f) * 4];
      p.vis[f] = rad[(s * 4 + f) * 4 + 1];
    }
  }
  const hrr = acc[ACC_GLOBAL] / G_SCALE / tag.simDt;      // W per m depth
  const radl = acc[ACC_GLOBAL + 1] / G_SCALE / tag.simDt;
  stats.hrr = 0.8 * stats.hrr + 0.2 * hrr;
  stats.rad = 0.8 * (stats.rad || 0) + 0.2 * radl;
  stats.Tmax = acc[ACC_GLOBAL + 2];
}

function stepWood(dt) {
  if (dt <= 0) return;
  for (const p of [...pieces]) {
    const w = p.wood;
    w.step(dt, p.qConv, p.qInc, p.charPending, ctl);
    p.charPending = [0, 0, 0, 0];
    if (w.dead) { removePiece(p); continue; }
    const b = p.body;
    b.density = (w.mv + w.mc) / (w.side * w.side);
    world.resize(b, w.half);
  }
}

let lastT = performance.now(), simAcc = 0, ratioAvg = 1;
let busy = false;
function frame(now) {
  requestAnimationFrame(frame);
  if (!sim || busy) return;
  const realDt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  frameNo++;
  // gas steps for this frame (capped: if the GPU cannot keep up the sim slows down)
  simAcc += realDt * ctl.speed;
  const maxSteps = Math.ceil(0.034 / sim.dt);
  let n = Math.floor(simAcc / sim.dt);
  if (n > maxSteps) { n = maxSteps; simAcc = 0; } else simAcc -= n * sim.dt;
  const simDt = n * sim.dt;
  if (realDt > 0 && ctl.speed > 0) ratioAvg = 0.97 * ratioAvg + 0.03 * (simDt / realDt / ctl.speed);

  // pieces stay movable while paused: the rigid bodies then run on real time
  world.advance(ctl.speed > 0 ? simDt : realDt);
  exchangeContacts(world, simDt);
  stepWood(simDt);
  if (drag && drag.body && !world.bodies.includes(drag.body)) endDrag();   // burnt out while held
  uploadPieces();
  sim.writeParams();
  ren.writeParams({
    view: VIEWS[view.mode], nPieces: sim.params.nPieces,
    exposure: Math.pow(2, view.exposure), time: sim.time, tmax: 1800, cam: cam(),
  });
  const enc = device.createCommandEncoder();
  sim.clearGeometry(enc);
  const pass = enc.beginComputePass();
  sim.encodeGeometry(pass);
  for (let s = 0; s < n; s++) sim.encodeStep(pass);
  sim.encodeRadiation(pass);
  pass.end();
  sim.encodeReadback(enc);
  ren.encode(enc);
  device.queue.submit([enc.finish()]);
  sim.frame++;
  sim.afterSubmit(onReadback, { simDt, owners: slots.slice() });
  updateStats(simDt);
}

let statT = 0;
function updateStats(simDt) {
  statT += simDt;
  if (performance.now() - (updateStats.last || 0) < 250) return;
  updateStats.last = performance.now();
  const D = ctl.depth;
  let mass = 0, burn = 0;
  for (const p of pieces) { mass += p.wood.mv + p.wood.mc; burn += p.wood.gasRate; }
  $('sHRR').textContent = (stats.hrr * D / 1000).toFixed(1) + ' kW';
  $('sBurn').textContent = (burn * D * 1000).toFixed(2) + ' g/s';
  $('sTmax').textContent = stats.Tmax ? (stats.Tmax - 273.15).toFixed(0) + ' °C' : '–';
  $('sRad').textContent = stats.hrr > 100 ? Math.round(100 * stats.rad / stats.hrr) + '%' : '–';
  $('sMass').textContent = (mass * D).toFixed(2) + ' kg';
  const since = sim.time - lastRefuel;
  $('sRefuel').textContent = Math.floor(since / 60) + ':' + String(Math.floor(since % 60)).padStart(2, '0');
  $('sSpeed').textContent = ctl.speed === 0 ? 'paused' : (ratioAvg * 100).toFixed(0) + '% real time';
}

// ---------------- canvas & input ----------------
function resize() {
  if (!sim) return;
  const H = sim.height + GROUND, W = SCENE_WIDTH;
  const stage = $('stage');
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const scale = Math.min(sw / W, sh / H);
  const cssW = Math.floor(W * scale), cssH = Math.floor(H * scale);
  canvas.style.width = cssW + 'px'; canvas.style.height = cssH + 'px';
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(cssW * dpr); canvas.height = Math.floor(cssH * dpr);
}
window.addEventListener('resize', resize);
new ResizeObserver(resize).observe($('stage'));

function toWorld(ev) {
  const r = canvas.getBoundingClientRect();
  const H = sim.height + GROUND, c = cam();
  return {
    x: (ev.clientX - r.left) / r.width * SCENE_WIDTH / c.zoom + c.x,
    y: (1 - (ev.clientY - r.top) / r.height) * H / c.zoom + c.y,
    inside: ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom,
    pxPerM: r.width / SCENE_WIDTH * c.zoom,
  };
}

// ---- dragging: one active drag, tracked on the window so it always ends ----
let drag = null;   // {body, isNew, size}
const ghost = $('ghost');

function endDrag() {
  if (!drag) return;
  if (drag.body) {
    world.endDrag(drag.body);
    if (drag.isNew) lastRefuel = sim.time;
  }
  drag = null;
  ghost.hidden = true;
  canvas.className = '';
}

function showGhost(ev, w) {
  const px = drag.size * w.pxPerM;
  Object.assign(ghost.style, { left: ev.clientX - px / 2 + 'px', top: ev.clientY - px / 2 + 'px', width: px + 'px', height: px + 'px' });
  ghost.hidden = false;
}

function onDragMove(ev) {
  if (!drag || !sim) return;
  const w = toWorld(ev);
  if (!drag.isNew) { world.moveDrag(drag.body, w.x, w.y); return; }
  // new piece from the palette: it exists only while the pointer is over the scene
  if (w.inside) {
    ghost.hidden = true;
    if (!drag.body) {
      const half = drag.size / 2;
      const x = Math.min(SCENE_WIDTH - half, Math.max(half, w.x)), y = Math.max(half, w.y);
      const p = addPiece(x, y, drag.size);       // created right under the pointer
      if (!p) return;
      drag.body = p.body;
      world.startDrag(p.body, 0, 0, w.x, w.y);
    } else world.moveDrag(drag.body, w.x, w.y);
  } else {
    if (drag.body) { removePiece(drag.body.userData); drag.body = null; }
    showGhost(ev, w);
  }
}

window.addEventListener('pointermove', onDragMove);
window.addEventListener('pointerup', endDrag);
window.addEventListener('pointercancel', endDrag);
window.addEventListener('blur', endDrag);

canvas.addEventListener('pointerdown', ev => {
  if (!sim || ev.button > 0) return;
  endDrag();
  const w = toWorld(ev);
  const hit = world.pick(w.x, w.y);
  if (!hit) return;
  ev.preventDefault();
  world.startDrag(hit.body, hit.lx, hit.ly, w.x, w.y);
  drag = { body: hit.body, isNew: false };
  canvas.className = 'grabbing';
});
canvas.addEventListener('pointermove', ev => {
  if (!sim || drag) return;
  const w = toWorld(ev);
  canvas.className = world.pick(w.x, w.y) ? 'grab' : '';
});

// palette: drag a new piece into the scene
for (const el of document.querySelectorAll('.piece')) {
  el.addEventListener('pointerdown', ev => {
    if (!sim || ev.button > 0) return;
    ev.preventDefault();
    endDrag();
    drag = { body: null, isNew: true, size: +el.dataset.size };
    onDragMove(ev);
  });
}

// ---------------- UI wiring ----------------
function seg(id, attr, fn) {
  const host = $(id);
  host.addEventListener('click', ev => {
    const b = ev.target.closest('button'); if (!b) return;
    for (const x of host.children) x.classList.toggle('on', x === b);
    fn(b.dataset[attr]);
  });
  return host;
}
seg('view-seg', 'view', v => { view.mode = v; $('diag').value = ''; });
$('diag').addEventListener('change', e => {
  if (!e.target.value) return;
  view.mode = e.target.value;
  for (const x of $('view-seg').children) x.classList.remove('on');
});
$('exposure').addEventListener('input', e => { view.exposure = +e.target.value; });
seg('speed-seg', 'speed', v => { ctl.speed = +v; });
seg('quality-seg', 'q', async q => {
  if (q === ctl.quality) return;
  ctl.quality = q;
  busy = true;
  showMsg('Rebuilding the grid…');
  await device.queue.onSubmittedWorkDone();
  await buildSim(q);
  showMsg('');
  busy = false;
});
$('resetScene').addEventListener('click', () => { sim.reset(); defaultScene(); });
$('clearWood').addEventListener('click', clearWood);
$('resetDefaults').addEventListener('click', () => {
  for (const c of CONTROLS) { ctl[c.key] = CONTROL_DEFAULTS[c.key]; c.sync(); }
  applyControls();
});

// ---------------- start ----------------
buildControls();
try {
  await initGPU();
  await buildSim(ctl.quality);
  defaultScene();
  requestAnimationFrame(frame);
} catch (e) {
  console.error(e);
  showMsg(e.message);
}

// debugging handle
window.campfire = { get sim() { return sim; }, world, get pieces() { return pieces; }, ctl, stats, addPiece, defaultScene, view, camera };
