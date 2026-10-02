// Headless rigid-body stability tests:  node tests/rigid_test.mjs
import { RigidWorld as RW, Body } from '../js/rigid.js';
const OVR = process.env.RIGID ? JSON.parse(process.env.RIGID) : {};
class RigidWorld extends RW { constructor(o = {}) { super({ ...OVR, ...o }); } }

const RHO = 500, STICK = 0.0125, LOG = 0.025;
let failures = 0;
const fmt = (x, d = 3) => x.toExponential(d);

function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!ok) failures++;
}

function run(world, seconds, onStep) {
  const n = Math.round(seconds / world.p.dt);
  for (let i = 0; i < n; i++) { world.step(world.p.dt); onStep && onStep(i); }
}

function snapshot(world) { return world.bodies.map(b => [b.x, b.y, b.a]); }
function drift(world, snap) {
  let dp = 0, da = 0;
  world.bodies.forEach((b, i) => {
    dp = Math.max(dp, Math.hypot(b.x - snap[i][0], b.y - snap[i][1]));
    da = Math.max(da, Math.abs(b.a - snap[i][2]));
  });
  return { dp, da };
}
function restMetrics(world, seconds) {
  let vmax = 0, pen = 0;
  run(world, seconds, () => {
    for (const b of world.bodies) vmax = Math.max(vmax, Math.hypot(b.vx, b.vy) + Math.abs(b.w) * b.half);
    pen = Math.max(pen, world.maxPenetration());
  });
  return { vmax, pen };
}

// 1. Tall tower of sticks, slightly misaligned
{
  const w = new RigidWorld();
  for (let i = 0; i < 12; i++) w.add(new Body(0.35 + (i % 2 ? 0.002 : -0.002), STICK + i * (2 * STICK + 0.0005), STICK, RHO));
  run(w, 3);
  const s = snapshot(w);
  const m = restMetrics(w, 30);
  const d = drift(w, s);
  check('tower12 rest', d.dp < 1e-4 && d.da < 1e-3 && m.vmax < 1e-3 && m.pen < 1e-3,
    `drift=${fmt(d.dp)}m rot=${fmt(d.da)} vmax=${fmt(m.vmax)} pen=${fmt(m.pen)}`);
  check('tower12 standing', Math.abs(w.bodies[11].x - 0.35) < 0.01, `top x=${w.bodies[11].x.toFixed(4)}`);
}

// 2. Pyramid of logs 4-3-2-1
{
  const w = new RigidWorld();
  let y = LOG;
  for (let row = 4; row >= 1; row--) {
    const x0 = 0.35 - (row - 1) * (LOG + 0.0005);
    for (let k = 0; k < row; k++) w.add(new Body(x0 + k * (2 * LOG + 0.001), y, LOG, RHO));
    y += 2 * LOG + 0.0005;
  }
  run(w, 3);
  const s = snapshot(w);
  const m = restMetrics(w, 30);
  const d = drift(w, s);
  check('pyramid rest', d.dp < 1e-4 && m.vmax < 1e-3 && m.pen < 1e-3,
    `drift=${fmt(d.dp)} vmax=${fmt(m.vmax)} pen=${fmt(m.pen)}`);
}

// 3. Random pile: 30 pieces dropped one at a time from 0.5-0.9 m
{
  const w = new RigidWorld();
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  let pen = 0, dropped = 0;
  const every = Math.round(0.3 / w.p.dt);
  run(w, 30 * 0.3 + 3, i => {
    if (i % every === 0 && dropped < 30) {
      const h = rnd() < 0.5 ? STICK : LOG;
      w.add(new Body(0.25 + 0.2 * rnd(), 0.5 + 0.4 * rnd(), h, RHO, rnd() * 3));
      dropped++;
    }
    pen = Math.max(pen, w.maxPenetration());
  });
  const s = snapshot(w);
  const m = restMetrics(w, 30);
  const d = drift(w, s);
  check('random pile impact', pen < 2e-3, `max pen during drops=${fmt(pen)}`);
  check('random pile rest', d.dp < 2e-4 && m.vmax < 2e-3 && m.pen < 1e-3,
    `drift=${fmt(d.dp)} rot=${fmt(d.da)} vmax=${fmt(m.vmax)} pen=${fmt(m.pen)}`);
}

// 4. Friction on incline (tilted gravity)
for (const deg of [30, 34, 40]) {
  const th = deg * Math.PI / 180;
  const w = new RigidWorld({ gravity: [9.81 * Math.sin(th), -9.81 * Math.cos(th)] });
  const b = w.add(new Body(0.1, LOG, LOG, RHO));
  const b2 = deg <= 30 ? w.add(new Body(0.1, 2 * LOG + STICK + 0.0005, STICK, RHO)) : b; // stick on log
  run(w, 0.5);
  const x0 = b.x, x20 = b2.x - b.x;
  run(w, 0.3);
  const dx = b.x - x0, rel = (b2.x - b.x) - x20;
  if (Math.tan(th) < 0.7) {
    check(`incline ${deg}° holds`, Math.abs(dx) < 1e-5 && Math.abs(rel) < 1e-5, `dx=${fmt(dx)} rel=${fmt(rel)}`);
  } else {
    const aExp = 9.81 * (Math.sin(th) - 0.5 * Math.cos(th));
    const v = b.vx;
    check(`incline ${deg}° slides`, dx > 0.01, `dx=${dx.toFixed(4)} v=${v.toFixed(3)} expected a=${aExp.toFixed(2)}`);
  }
}

// 5. Overhang: stick on a log, centre inside vs outside the edge
for (const [off, expectFall] of [[0.020, false], [0.023, false], [0.028, true]]) {
  const w = new RigidWorld();
  const base = w.add(new Body(0.35, LOG, LOG, RHO));
  const top = w.add(new Body(0.35 + off, 2 * LOG + STICK + 0.0003, STICK, RHO));
  run(w, 4);
  const fell = top.y < 2 * LOG;
  check(`overhang off=${off} ${expectFall ? 'tips' : 'stays'}`, fell === expectFall,
    `top y=${top.y.toFixed(4)} angle=${top.a.toFixed(3)} base moved=${fmt(Math.abs(base.x - 0.35))}`);
}

// 6. Shrinking: self-similar supports, shrinkage much faster than burning.
//    Settling must be smooth (speed ~ shrink rate, no accelerations/jumps).
function shrinkTest(name, build, fmin, T) {
  const w = new RigidWorld();
  build(w);
  run(w, 3);
  const h0 = w.bodies.map(b => b.half);
  let pen = 0, jerk = 0, jerkAt = '';
  const spf = Math.round(1 / 60 / w.p.dt);           // rigid steps per rendered frame
  let prev = snapshot(w), prevD = null;
  const n = Math.round(T / w.p.dt);
  for (let i = 0; i < n; i++) {
    const f = 1 - (1 - fmin) * (i + 1) / n;
    w.bodies.forEach((b, k) => w.resize(b, h0[k] * f));
    w.step(w.p.dt);
    pen = Math.max(pen, w.maxPenetration());
    if ((i + 1) % spf === 0) {
      // visible jerkiness: change of per-frame displacement between frames
      const cur = snapshot(w);
      const d = cur.map((c, k) => [c[0] - prev[k][0], c[1] - prev[k][1]]);
      if (prevD) d.forEach((dk, k) => {
        const j = Math.hypot(dk[0] - prevD[k][0], dk[1] - prevD[k][1]);
        if (j > jerk) { jerk = j; jerkAt = `body ${k} t=${(i * w.p.dt).toFixed(2)}s f=${f.toFixed(3)}`; }
      });
      prev = cur; prevD = d;
    }
  }
  check(`shrink ${name}`, jerk < 5e-5 && pen < 1e-3, `max frame-to-frame jerk=${fmt(jerk)}m (${jerkAt}) pen=${fmt(pen)}`);
}
shrinkTest('tower8 to 40%', w => {
  for (let i = 0; i < 8; i++) w.add(new Body(0.35, STICK + i * (2 * STICK + 0.0003), STICK, RHO));
}, 0.4, 30);
shrinkTest('pyramid to 60%', w => {
  let y = LOG;
  for (let row = 3; row >= 1; row--) {
    const x0 = 0.35 - (row - 1) * (LOG + 0.0005);
    for (let k = 0; k < row; k++) w.add(new Body(x0 + k * (2 * LOG + 0.001), y, LOG, RHO));
    y += 2 * LOG + 0.0005;
  }
}, 0.6, 30);
shrinkTest('cabin to 80%', w => {
  const add = (x, y, h) => w.add(new Body(x, y, h, RHO));
  add(0.32, LOG, LOG); add(0.38, LOG, LOG);
  add(0.33, 2 * LOG + STICK + 0.0003, STICK); add(0.37, 2 * LOG + STICK + 0.0003, STICK);
  add(0.35, 2 * LOG + 2 * STICK + LOG + 0.0006, LOG);
  for (let i = 0; i < 3; i++) add(0.35, 4 * LOG + 2 * STICK + 0.001 + i * (2 * STICK + 0.0003) + STICK, STICK);
}, 0.8, 30);

// 7. Drop a log from 0.6 m onto a stick pile: impact, no bounce
{
  const w = new RigidWorld();
  for (let i = 0; i < 5; i++) w.add(new Body(0.28 + i * (2 * STICK + 0.001), STICK, STICK, RHO));
  run(w, 1);
  const lg = w.add(new Body(0.33, 0.6, LOG, RHO, 0));
  let pen = 0, landed = false, peak = 0, yLand = 0;
  run(w, 4, () => {
    pen = Math.max(pen, w.maxPenetration());
    if (!landed && lg.vy > -0.01 && lg.y < 0.2) { landed = true; yLand = lg.y; }
    if (landed) peak = Math.max(peak, lg.y - yLand);
  });
  check('log drop', pen < 3e-3 && peak < 0.003, `pen=${fmt(pen)} rebound=${fmt(peak)} final y=${lg.y.toFixed(4)}`);
}

// 8. Dragging (hand grip)
{
  // lift a log off the floor to 0.4 m and hold it there
  const w = new RigidWorld();
  const b = w.add(new Body(0.3, LOG, LOG, RHO));
  run(w, 0.5);
  w.startDrag(b, 0.01, 0.01, b.x + 0.01, b.y + 0.01);   // grabbed off-centre
  const steps = Math.round(1.5 / w.p.dt);
  for (let i = 0; i < steps; i++) { const t = (i + 1) / steps; w.moveDrag(b, 0.31, LOG + 0.01 + t * 0.4); w.step(w.p.dt); }
  run(w, 3);
  const err = Math.hypot(b.x + 0.01 - 0.31, b.y + 0.01 - (LOG + 0.41));
  check('drag lift & hold', err < 0.004 && Math.abs(b.a) < 0.01, `pos err=${fmt(err)} angle=${b.a.toFixed(4)}`);
  w.endDrag(b);
  run(w, 2);
  check('release drops straight', Math.abs(b.x - 0.3) < 0.002 && b.y < LOG + 0.001 && Math.abs(b.a) < 0.01, `x=${b.x.toFixed(4)} y=${b.y.toFixed(4)}`);
}
{
  // a stick carried over a pile and placed on top, then left there
  const w = new RigidWorld();
  for (let i = 0; i < 3; i++) w.add(new Body(0.3 + i * 0.051, LOG, LOG, RHO));
  run(w, 1);
  const b = w.add(new Body(0.15, 0.2, STICK, RHO));
  w.startDrag(b, 0, 0, 0.15, 0.2);
  const steps = Math.round(1 / w.p.dt);
  for (let i = 0; i < steps; i++) { w.moveDrag(b, 0.15 + (i + 1) / steps * 0.2, 0.2); w.step(w.p.dt); }
  run(w, 1);
  const held = Math.hypot(b.x - 0.35, b.y - 0.2);
  w.moveDrag(b, 0.35, 2 * LOG + STICK + 0.002); run(w, 1); w.endDrag(b);
  run(w, 1);
  const s = snapshot(w);
  run(w, 3);
  const m = restMetrics(w, 5);
  const d = drift(w, s);
  check('drag carry & place', held < 0.004 && b.y > 2 * LOG && d.dp < 2e-3 && m.vmax < 1e-3,
    `held err=${fmt(held)} final y=${b.y.toFixed(4)} drift=${fmt(d.dp)}`);
}
{
  // hand strength: pushing a stick hard into the base of a 6-stick tower does
  // not bulldoze the tower; the held stick is blocked and lags behind the hand
  const w = new RigidWorld();
  for (let i = 0; i < 6; i++) w.add(new Body(0.35, STICK + i * (2 * STICK + 0.0003), STICK, RHO));
  run(w, 2);
  const top = w.bodies[5];
  const b = w.add(new Body(0.25, STICK, STICK, RHO));
  w.startDrag(b, 0, 0, 0.25, STICK);
  const steps = Math.round(1 / w.p.dt);
  for (let i = 0; i < steps; i++) { w.moveDrag(b, 0.25 + (i + 1) / steps * 0.2, STICK); w.step(w.p.dt); }
  run(w, 1);
  w.endDrag(b);
  run(w, 2);
  const lag = 0.45 - b.x;
  check('hand strength (moderate)', lag > 0.02, `tower top dx=${(top.x - 0.35).toFixed(4)} held stick lag=${lag.toFixed(3)} m`);
}

// 9. Performance: 60 bodies
{
  const w = new RigidWorld();
  for (let i = 0; i < 60; i++) w.add(new Body(0.1 + (i % 10) * 0.055, 0.05 + Math.floor(i / 10) * 0.06, i % 3 ? STICK : LOG, RHO));
  run(w, 3);
  const t0 = performance.now();
  run(w, 2);
  const ms = (performance.now() - t0) / (2 * 60);
  check('perf 60 bodies', ms < 4, `${ms.toFixed(2)} ms per 1/60 s frame`);
}

console.log(failures ? `\n${failures} FAILED` : '\nALL PASSED');
process.exit(failures ? 1 : 0);
