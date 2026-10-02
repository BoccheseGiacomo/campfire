// Char smouldering and contact conduction:  node tests/char_test.mjs
import { WoodPiece, WOOD } from '../js/wood.js';
import { RigidWorld, Body } from '../js/rigid.js';
import { exchangeContacts } from '../js/contact.js';
import { PHYS } from '../js/config.js';

let failures = 0;
const check = (name, ok, detail) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); if (!ok) failures++; };
const sig = WOOD.sigma, Tamb = 293.15, ctl = { charYield: 0.25, pyroMult: 1 };

function charPiece(side, T) {
  const w = new WoodPiece(side, 500);
  w.mv = 0; w.mc = w.rhoC * side * side; w.updateSide();
  w.d = [side / 2, side / 2, side / 2, side / 2]; w.charCov = [1, 1, 1, 1];
  w.Ts = [T, T, T, T]; w.Tp = [T, T, T, T]; w.Tc = T;
  return w;
}
// surface oxidation as on the GPU: m_O2'' = A exp(-Ta/Ts) rho_gas YO2, gas film at (Ts+Tamb)/2
function charBurnt(w, f, dt) {
  if (w.exposure[f] <= 0) return 0;
  const Tg = 0.5 * (w.Ts[f] + Tamb), rho = PHYS.rhoAmb * PHYS.Tamb / Tg;
  const mO2 = PHYS.charA * Math.exp(-PHYS.charTa / w.Ts[f]) * rho * PHYS.YO2air;
  return mO2 / PHYS.charO2 * w.side * w.exposure[f] * dt;
}

// 1. smouldering rate near 700 K
{
  const w = charPiece(0.02, 700);
  const rate = charBurnt(w, 1, 1) / w.side * 1000;
  check('smoulder rate at 700 K', rate > 0.05 && rate < 0.2, `${rate.toFixed(3)} g/m2/s`);
}

// 2. a lone ember in open air cools and goes out
{
  const w = charPiece(0.02, 750);
  w.exposure = [1, 1, 1, 0];                         // resting on the floor
  const dt = 1 / 60, m0 = w.mc;
  for (let t = 0; t < 120; t += dt) {
    const qInc = [0, 1, 2, 3].map(() => sig * Tamb ** 4);
    const qConv = w.Ts.map((T, f) => 10 * w.side * (Tamb - T) * w.exposure[f]);
    w.step(dt, qConv, qInc, [0, 1, 2, 3].map(f => charBurnt(w, f, dt)), ctl);
  }
  check('lone ember dies', Math.max(...w.Ts) < 550, `Ts=${w.Ts.map(t => t.toFixed(0))} mass lost=${((1 - w.mc / m0) * 100).toFixed(1)}%`);
}

// 3. an ember surrounded by hot embers (sees 750 K on all sides) keeps smouldering
{
  const w = charPiece(0.02, 750);
  const dt = 1 / 60, m0 = w.mc;
  for (let t = 0; t < 300; t += dt) {
    const qInc = [0, 1, 2, 3].map(() => sig * 750 ** 4);
    const qConv = w.Ts.map(T => 10 * w.side * (700 - T));
    w.step(dt, qConv, qInc, [0, 1, 2, 3].map(f => charBurnt(w, f, dt)), ctl);
  }
  const lost = 1 - w.mc / m0;
  check('ember in a pile sustains', Math.min(...w.Ts) > 750 && lost > 0.05,
    `Ts=${w.Ts.map(t => t.toFixed(0))} mass lost in 5 min=${(lost * 100).toFixed(0)}%`);
}

// 4. contact conduction: hot coal resting on a cold stick heats its top face
{
  const world = new RigidWorld();
  const stick = new WoodPiece(0.025, 500), coal = charPiece(0.02, 1000);
  const bs = world.add(new Body(0.35, 0.0125, stick.half, 500)); bs.userData = { wood: stick };
  const bc = world.add(new Body(0.35, 0.025 + 0.01 + 0.0002, coal.half, 150)); bc.userData = { wood: coal };
  world.advance(0.5);
  const energy = () => [stick, coal].reduce((e, w) => e + [0, 1, 2, 3].reduce((a, f) => a + w.surfaceCap(f) * w.Ts[f], 0), 0);
  const E0 = energy();
  for (let t = 0; t < 1; t += 1 / 60) { world.advance(1 / 60); exchangeContacts(world, 1 / 60); }
  const dE = Math.abs(energy() - E0) / E0;
  check('contact exchange conserves energy', dE < 1e-9, `relative change=${dE.toExponential(2)}`);
  const ctlz = { charYield: 0.25, pyroMult: 1 };
  stick.exposure = [1, 0, 1, 0]; coal.exposure = [1, 1, 1, 0];
  for (let t = 0; t < 30; t += 1 / 60) {
    world.advance(1 / 60);
    exchangeContacts(world, 1 / 60);
    const none = [0, 0, 0, 0];
    stick.step(1 / 60, none, [0, 1, 2, 3].map(() => sig * Tamb ** 4), none, ctlz);
    // the coal is part of a sustained bed: held at bed temperature
    coal.Ts = [1000, 1000, 1000, 1000]; coal.Tp = [1000, 1000, 1000, 1000];
  }
  check('coal bed heats stick through contact', stick.Ts[1] > 600,
    `stick top=${stick.Ts[1].toFixed(0)} K, other faces=${[0, 2, 3].map(f => stick.Ts[f].toFixed(0))}, coal bottom=${coal.Ts[3].toFixed(0)} K`);
}

console.log(failures ? `\n${failures} FAILED` : '\nALL PASSED');
process.exit(failures ? 1 : 0);
