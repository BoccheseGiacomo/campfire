// 1D freely propagating premixed flame with the simulator's gas model
// (rho ~ 1/T, Sutherland mu(T), unity Lewis number, cp const, one-step Arrhenius).
// Measures the flame speed for a given grid spacing and thickening factor.
//   node tests/flame1d.mjs
const P = {
  Tamb: 293.15, rhoAmb: 1.2, cp: 1300, Pr: 0.7, dHc: 15e6, sO2: 15e6 / 13.1e6, YO: 0.232,
};
const mu = T => 1.716e-5 * Math.pow(T / 273.15, 1.5) * 383.55 / (T + 110.4);

// Lagrangian-free approach: solve in the lab frame with the unburnt gas at rest
// ahead of the flame; burnt gas expands backwards (open end behind). Continuity
// in 1D gives u(x) from the dilatation; scalars advected upwind.
function flameSpeed({ B, Ta, dx, F = 1, phi = 1, L = 0.12, tEnd = 0.25 }) {
  const n = Math.round(L / dx);
  const YFst = P.YO / P.sO2;          // fuel per unit O2 at stoichiometric
  // mixture: fuel/air with equivalence ratio phi
  const YF0 = phi * YFst / (1 + phi * YFst) * 1, YO0 = P.YO / (1 + phi * YFst);
  let T = new Float64Array(n).fill(P.Tamb), YF = new Float64Array(n).fill(YF0), YO = new Float64Array(n).fill(YO0);
  // ignite the left end (closed wall at x=0, flame moves right into fresh gas)
  for (let i = 0; i < Math.round(0.01 / dx); i++) { T[i] = 2000; YF[i] = 0; YO[i] = Math.max(0, YO0 - P.sO2 * YF0); }
  const alphaMax = mu(2300) / P.Pr * F / (P.rhoAmb * P.Tamb / 2300);
  const dt = Math.min(0.2 * dx * dx / alphaMax, 0.2 * dx / 6);
  let t = 0;
  const track = [];
  let Tb = 0;
  const u = new Float64Array(n + 1);   // velocity at faces, u[0] = 0 (closed end)
  while (t < tEnd) {
    // sources: reaction (divided by F) and diffusion (times F)
    const S = new Float64Array(n);
    const Tn = T.slice(), Fn = YF.slice(), On = YO.slice();
    for (let i = 0; i < n; i++) {
      const rho = P.rhoAmb * P.Tamb / T[i];
      let dT = 0, dF = 0, dO = 0;
      for (const j of [i - 1, i + 1]) {
        if (j < 0 || j >= n) continue;
        const a = mu(0.5 * (T[i] + T[j])) / P.Pr * F;   // rho D = k/cp
        dT += a * (T[j] - T[i]); dF += a * (YF[j] - YF[i]); dO += a * (YO[j] - YO[i]);
      }
      const c = dt / (rho * dx * dx);
      Tn[i] += c * dT; Fn[i] += c * dF; On[i] += c * dO;
      const w = B * rho * Fn[i] * On[i] * Math.exp(-Ta / Tn[i]) / F;
      const d = Math.min(w * dt, Fn[i], On[i] / P.sO2);
      Fn[i] -= d; On[i] -= P.sO2 * d; Tn[i] += d * P.dHc / P.cp;
      S[i] = Math.log(Tn[i] / T[i]) / dt;
    }
    T = Tn; YF = Fn; YO = On;
    // continuity: du/dx = S
    for (let i = 0; i < n; i++) u[i + 1] = u[i] + S[i] * dx;
    // upwind advection
    const Ta2 = T.slice(), F2 = YF.slice(), O2 = YO.slice();
    for (let i = 0; i < n; i++) {
      const uc = 0.5 * (u[i] + u[i + 1]);
      const j = uc > 0 ? i - 1 : i + 1;
      const k = Math.min(Math.max(j, 0), n - 1);
      const cfl = Math.abs(uc) * dt / dx;
      Ta2[i] = T[i] + cfl * (T[k] - T[i]); F2[i] = YF[i] + cfl * (YF[k] - YF[i]); O2[i] = YO[i] + cfl * (YO[k] - YO[i]);
    }
    T = Ta2; YF = F2; YO = O2;
    t += dt;
    // front position: furthest point hotter than 1200 K
    let xf = 0; for (let i = 0; i < n; i++) if (T[i] > 1200) xf = (i + 0.5) * dx;
    for (let i = 0; i < n; i++) Tb = Math.max(Tb, T[i]);
    track.push([t, xf]);
    if (xf > L * 0.8) break;
  }
  // lab-frame front speed between 30% and 75% of the domain; the closed end keeps
  // the burnt gas at rest, so S_L = lab speed * rho_b / rho_u = lab speed * Tu / Tb
  const a = track.find(([, x]) => x > 0.3 * L), b = track.find(([, x]) => x > 0.75 * L) || track[track.length - 1];
  if (!a || b[0] <= a[0]) return { SL: 0, Tb };
  return { SL: (b[1] - a[1]) / (b[0] - a[0]) * P.Tamb / Tb, Tb };
}

const args = Object.fromEntries(process.argv.slice(2).map(a => a.split('=')).map(([k, v]) => [k, +v]));
const Ta = args.Ta || 15000;
if (args.calibrate) {
  for (const B of [3e9, 1e10, 2e10, 4e10]) {
    const r = flameSpeed({ B, Ta, dx: 5e-5, L: 0.03, tEnd: 1 });
    console.log(`B=${B.toExponential(1)} Ta=${Ta} resolved: S_L=${r.SL.toFixed(3)} m/s Tb=${r.Tb.toFixed(0)}`);
  }
} else {
  const B = args.B || 1e9;
  console.log(`B=${B} Ta=${Ta}`);
  for (const dx of [5e-5, 2e-4, 1e-3, 3.125e-3]) {
    for (const F of dx > 5e-5 ? [1, Math.max(1, 3 * dx / 5e-4)] : [1]) {
      const L = Math.max(0.03, 60 * dx);
      const r = flameSpeed({ B, Ta, dx, F, L, tEnd: 3 });
      console.log(`  dx=${(dx * 1000).toFixed(3)}mm F=${F.toFixed(1)}: S=${r.SL.toFixed(3)} m/s`);
    }
  }
}
