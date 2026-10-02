// Pointwise sources, applied after advection and diffusion:
//  - wood/gas coupling: conduction to wood faces, pyrolysis gas injection,
//    char surface oxidation (smeared over a one-cell interface band)
//  - finite-rate one-step combustion  F + s O2 -> (1+s) P
//  - soot formation (from fuel) and oxidation (as condensed fuel)
//  - optically thin radiative loss
//  - entrainment of ambient air through the front/back faces (finite depth)
// and the low-Mach divergence source  S = mdot/rho + (1/T) DT/Dt.
//#include common

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> T: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4f>;
@group(0) @binding(3) var<storage, read> Tadv: array<f32>;
@group(0) @binding(4) var<storage, read> U: array<f32>;
@group(0) @binding(5) var<storage, read> V: array<f32>;
@group(0) @binding(6) var<storage, read> cellG: array<vec2f>;
@group(0) @binding(7) var<storage, read> pieces: array<Piece>;
@group(0) @binding(8) var<storage, read_write> SQ: array<vec2f>;
@group(0) @binding(9) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read> faceW: array<i32>;

// accumulator layout (fixed point): per face [heat to wood J/m * 1e6, char kg/m * 1e12],
// then globals (J/m * 1e3): [heat release, gas radiation loss, wood conduction]
const ACC_GLOBAL: u32 = 512u;
const E_SCALE: f32 = 1.0e6;
const M_SCALE: f32 = 1.0e12;
const G_SCALE: f32 = 1.0e3;

var<workgroup> redQ: array<f32, 64>;
var<workgroup> redR: array<f32, 64>;
var<workgroup> redT: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = i32(id.x); let j = i32(id.y);
  let inside = i < P.nx && j < P.ny;
  var qrel = 0.0;   // heat release W/m^3
  var qrad = 0.0;   // radiative loss W/m^3
  var tpeak = 0.0;
  if (inside) {
    let k = idxC(i, j);
    var Tc = T[k];
    var Yc = Y[k];
    let g = cellG[k];
    let sdf = g.x;
    let sid = bitcast<u32>(g.y);
    var massSrc = 0.0;
    let cellA = P.dx * P.dx;
    if (sdf < 0.0 && sid != SOLID_NONE) {
      // inside wood: hold the surface temperature (no-slip wall value for sampling)
      let pc = pieces[sid >> 2u];
      T[k] = pc.Ts[sid & 3u];
      SQ[k] = vec2f(0.0);
    } else {
      // ---------------- wood coupling ----------------
      if (sid != SOLID_NONE && sdf < 2.0 * P.dx) {
        let pi = sid >> 2u; let fi = sid & 3u;
        let pc = pieces[pi];
        let Ts = pc.Ts[fi];
        // interface length per area (1/m), normalised so the band integrates to the face length
        let band = f32(faceW[sid]) / 1.0e8;
        let w = max(0.0, 1.0 - abs(sdf - P.dx) / P.dx) / P.dx * (2.0 * pc.half) / max(band, 1e-6);
        let rho = rhoOf(Tc);
        // conduction across the boundary layer between cell centre and surface
        let kg = rhoDOf(0.5 * (Tc + Ts)) * P.cp;
        let dist = max(sdf, 0.5 * P.dx);
        let rate = kg * w / (dist * rho * P.cp);
        let dTc = (Ts - Tc) * (1.0 - exp(-rate * P.dt));
        Tc += dTc;
        let toWood = -rho * P.cp * dTc * cellA;          // J per m depth this step
        atomicAdd(&acc[(pi * 4u + fi) * 2u], i32(round(toWood * E_SCALE)));
        // pyrolysis gas (pure fuel at the surface temperature)
        let mf = pc.mflux[fi] * w;
        if (mf > 0.0) {
          let madd = mf * P.dt;
          let r = rhoOf(Tc);
          Yc = (r * Yc + vec4f(madd, 0.0, 0.0, 0.0)) / (r + madd);
          Tc = (r * Tc + madd * Ts) / (r + madd);
          massSrc += mf;
        }
        // char surface oxidation C + O2 -> CO2; heat goes to the wood surface
        let cov = pc.charCov[fi];
        if (cov > 0.0 && Yc.y > 0.0) {
          let r = rhoOf(Tc);
          let kc = P.charA * exp(-P.charTa / Ts) * cov;
          let mO2 = min(kc * r * Yc.y * w * P.dt, 0.9 * r * Yc.y);
          let mC = mO2 / P.charO2;
          Yc = (r * Yc + vec4f(0.0, -mO2, mC + mO2, 0.0)) / (r + mC);
          Tc = (r * Tc + mC * Ts) / (r + mC);
          massSrc += mC / P.dt;
          atomicAdd(&acc[(pi * 4u + fi) * 2u + 1u], i32(round(mC * cellA * M_SCALE)));
        }
      }
      // ---------------- test sources ----------------
      let x = (f32(i) + 0.5) * P.dx; let y = (f32(j) + 0.5) * P.dx;
      if (P.heaterT > 0.0 && x > P.heaterX0 && x < P.heaterX1 && y < P.heaterY1) {
        Tc = P.heaterT;
      }
      if (P.burnerFlux > 0.0 && j == 0 && x > P.burnerX0 && x < P.burnerX1) {
        let mf = P.burnerFlux / P.dx;
        let madd = mf * P.dt;
        let r = rhoOf(Tc);
        Yc = (r * Yc + vec4f(madd, 0.0, 0.0, 0.0)) / (r + madd);
        Tc = (r * Tc + madd * P.burnerT) / (r + madd);
        massSrc += mf;
      }
      // ---------------- gas-phase combustion ----------------
      let rho0 = rhoOf(Tc);
      let NS = 8;
      let h = P.dt / f32(NS);
      var burned = 0.0;
      for (var s = 0; s < NS; s++) {
        // w = B rho^2 YF YO exp(-Ta/T) [kg/m^3/s]  ->  dYF/dt = w/rho
        let wr = P.reactB * rho0 * Yc.x * Yc.y * exp(-P.reactTa / Tc);
        let dF = min(wr * h, min(Yc.x, Yc.y / P.sO2));
        Yc += vec4f(-dF, -P.sO2 * dF, (1.0 + P.sO2) * dF, 0.0);
        Tc += dF * P.dHc / P.cp;
        burned += dF;
      }
      // ---------------- soot ----------------
      let dSf = min(P.sootFormA * exp(-P.sootFormTa / Tc) * Yc.x * P.dt, Yc.x);
      Yc += vec4f(-dSf, 0.0, 0.0, dSf);
      let dSo = min(P.sootOxA * Yc.w * Yc.y * exp(-P.sootOxTa / Tc) * P.dt, min(Yc.w, Yc.y / P.sO2));
      Yc += vec4f(0.0, -P.sO2 * dSo, (1.0 + P.sO2) * dSo, -dSo);
      Tc += dSo * P.dHc / P.cp;
      burned += dSo;
      qrel = rho0 * burned * P.dHc / P.dt;
      // ---------------- radiation (optically thin, linearised implicit) ----------------
      let r1 = rhoOf(Tc);
      let fv = r1 * Yc.w / P.sootDensity;
      let kap = P.kappaSootC * fv * Tc + P.kappaGas * Yc.z;
      let T3 = Tc * Tc * Tc;
      qrad = 4.0 * kap * P.sigma * (T3 * Tc - pow(P.Tamb, 4.0));
      let dTr = qrad * P.dt / (r1 * P.cp) / (1.0 + 16.0 * kap * P.sigma * T3 * P.dt / (r1 * P.cp));
      Tc -= dTr;
      // ---------------- entrainment through the front/back faces ----------------
      // Morton-Taylor-Turner: ambient air enters both faces of the finite-depth
      // fire at alpha*|u|, i.e. a mass source 2 alpha |u| rho_amb / depth.
      let uc = 0.5 * vec2f(U[idxU(i, j)] + U[idxU(i + 1, j)], V[idxV(i, j)] + V[idxV(i, j + 1)]);
      let me = 2.0 * P.exchAlpha * length(uc) * P.rhoAmb / P.depth;
      if (me > 0.0) {
        let madd = me * P.dt;
        let r = rhoOf(Tc);
        Yc = (r * Yc + madd * ambY()) / (r + madd);
        Tc = (r * Tc + madd * P.Tamb) / (r + madd);
        massSrc += me;
      }
      // ---------------- write ----------------
      Tc = max(Tc, 0.5 * P.Tamb);
      Yc = max(Yc, vec4f(0.0));
      let S = massSrc / rhoOf(Tc) + log(Tc / Tadv[k]) / P.dt;
      T[k] = Tc;
      Y[k] = Yc;
      tpeak = Tc;
      SQ[k] = vec2f(S, qrel);
    }
  }
  // workgroup reduction of global energy rates
  redQ[li] = qrel * P.dx * P.dx * P.dt;
  redR[li] = qrad * P.dx * P.dx * P.dt;
  redT[li] = tpeak;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { redQ[li] += redQ[li + s]; redR[li] += redR[li + s]; redT[li] = max(redT[li], redT[li + s]); }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicAdd(&acc[ACC_GLOBAL], i32(round(redQ[0] * G_SCALE)));
    atomicAdd(&acc[ACC_GLOBAL + 1u], i32(round(redR[0] * G_SCALE)));
    atomicMax(&acc[ACC_GLOBAL + 2u], i32(redT[0]));
  }
}
