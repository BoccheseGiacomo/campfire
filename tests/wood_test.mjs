// Wood model sanity:  node tests/wood_test.mjs
import { WoodPiece } from '../js/wood.js';
const ctl = { charYield: 0.25, pyroMult: 1 };
const sig = 5.670374e-8;
function run(side, qFlux, Tgas, h, seconds, label, charOx = 0) {
  const w = new WoodPiece(side, 500);
  const dt = 1 / 60; let tIgn = null, peak = 0, out = [];
  for (let t = 0; t < seconds && !w.dead; t += dt) {
    const s = w.side;
    const qConv = w.Ts.map(Ts => h * s * (Tgas - Ts));
    const qInc = [0, 1, 2, 3].map(() => qFlux + sig * 293.15 ** 4);
    // crude char oxidation: kinetic rate with ambient O2 at the surface
    const burnt = w.Ts.map((Ts, f) => charOx * w.charCov[f] * 470 * Math.exp(-12000 / Ts) * 0.27 * s * dt / 2.667);
    w.step(dt, qConv, qInc, burnt, ctl);
    const mf = w.mflux.reduce((a, b) => a + b) / 4;
    peak = Math.max(peak, mf);
    if (tIgn === null && mf > 0.0025) tIgn = t;
    if (Math.abs(t % 30) < dt) out.push(`t=${t.toFixed(0)}s side=${(w.side * 1000).toFixed(1)}mm Ts=${w.Ts[1].toFixed(0)} Tp=${w.Tp[1].toFixed(0)} Tc=${w.Tc.toFixed(0)} d=${(w.d[1] * 1000).toFixed(1)}mm mflux=${(mf * 1000).toFixed(1)}g/m2s mv=${(w.mv * 1000).toFixed(1)} mc=${(w.mc * 1000).toFixed(1)}g/m`);
  }
  console.log(`== ${label}: ignition (2.5 g/m2s) at ${tIgn?.toFixed(1)} s, peak flux ${(peak * 1000).toFixed(1)} g/m2s`);
  console.log(out.slice(0, 14).join('\n'));
}
run(0.025, 30000, 1100, 25, 400, 'stick 2.5cm in 30 kW/m2 + 1100K gas');
run(0.05, 30000, 1100, 25, 400, 'log 5cm in 30 kW/m2 + 1100K gas');
run(0.025, 0, 293, 10, 120, 'stick in cold air (no heating)');
