// Thermal and pyrolysis model of a wood piece (per unit depth along its axis).
//
// Each square piece has, per face, a thin surface node (Ts) and a pyrolysis
// front node (Tp) below the char layer of thickness d; all faces share a core
// node (Tc). Virgin wood decomposes with a single-step Arrhenius rate
//   r = A exp(-Ta/T) m_virgin   ->  (1 - Y_char) gas + Y_char char,
// mostly at the front (heated through the insulating char), and at the core
// late in life. Char burns at the surface with O2 from the gas (computed on the
// GPU) and that heat goes to the surface node. The piece's side follows from
// the remaining virgin and char volumes, so it shrinks as it burns.

export const WOOD = {
  cv: 1500,      // J/kg/K virgin wood
  kv: 0.15,      // W/m/K
  cc: 1000,      // char
  kc: 0.08,      // char solid-phase conductivity; plus radiation across pores (below)
  pore: 0.001,   // m, effective char pore/crack size for radiative conductivity
  charDensityRatio: 0.3,      // rho_char / rho_virgin
  eps: 0.9,
  pyroA: 2.0e7,  // 1/s
  pyroTa: 12000, // K (100 kJ/mol, hemicellulose-dominated onset ~ 250-300 C)
  dHpyro: 3.0e5, // J/kg endothermic
  dHchar: 32.8e6,// J/kg C (C + O2 -> CO2)
  cpGas: 1150,
  deltaS: 0.0015,  // surface node thickness (m)
  deltaP: 0.002,   // front node thickness (m)
  minDepth: 0.0004,
  sigma: 5.670374e-8,
  minHalf: 0.0015, // pieces smaller than this have burnt out
};

export class WoodPiece {
  constructor(side, rhoV, T0 = 293.15) {
    this.rhoV = rhoV;
    this.rhoC = rhoV * WOOD.charDensityRatio;
    this.mv = rhoV * side * side;     // virgin mass, kg/m
    this.mc = 0;                      // char mass, kg/m
    this.Ts = [T0, T0, T0, T0];
    this.Tp = [T0, T0, T0, T0];
    this.d = [0, 0, 0, 0];            // char depth per face
    this.Tc = T0;
    this.mflux = [0, 0, 0, 0];        // pyrolysis gas flux per face, kg/m^2/s
    this.charCov = [0, 0, 0, 0];
    this.gasRate = 0;                 // kg/s/m total volatiles released
    this.exposure = [1, 1, 1, 1];     // fraction of each face open to the gas
    this.dead = false;
    this.side = side;
  }

  get half() { return 0.5 * this.side; }

  // heat capacity of a face's surface node (J/K per m), as used in step()
  surfaceCap(f) {
    const W = WOOD, s = this.side;
    const dS = Math.min(W.deltaS, s / 8);
    const phi = this.mv <= 1e-9 ? 1 : Math.min(1, this.d[f] / dS);
    return (this.rhoV * W.cv * (1 - phi) + this.rhoC * W.cc * phi) * dS * s;
  }
  updateSide() { this.side = Math.sqrt(this.mv / this.rhoV + this.mc / this.rhoC); }
  charFraction() { return this.mc / this.rhoC / (this.mv / this.rhoV + this.mc / this.rhoC + 1e-12); }

  // Pre-ignite: give it a char layer and a hot surface (used for the starting fire).
  preheat(Tsurf, charDepth, Tcore, charYield) {
    const s = this.side;
    const dv = Math.min(charDepth, 0.45 * s);
    // virgin consumed by the char layer on all faces (thin-shell approximation)
    const vVol = 4 * dv * s - 4 * dv * dv;
    const consumed = Math.min(this.mv * 0.98, vVol * this.rhoV);
    this.mv -= consumed;
    this.mc += consumed * charYield;
    this.d = [dv, dv, dv, dv];
    this.Ts = [Tsurf, Tsurf, Tsurf, Tsurf];
    this.Tp = this.Ts.map(t => 0.5 * (t + Tcore));
    this.Tc = Tcore;
    this.updateSide();
  }

  // Advance by dt with heat inputs per face (W per m depth):
  //   qConv[f]: convective/conductive heat from the gas
  //   qInc[f]:  incident radiative flux (W/m^2)
  //   charBurnt[f]: char consumed by surface oxidation this interval (kg/m)
  step(dt, qConv, qInc, charBurnt, ctl) {
    if (this.dead) return;
    const W = WOOD, Yc = ctl.charYield;
    // char oxidation: remove mass, heat the surface
    const qChar = [0, 0, 0, 0];
    for (let f = 0; f < 4; f++) {
      const m = Math.min(charBurnt[f], this.mc);
      this.mc -= m;
      qChar[f] = m * W.dHchar / dt;
      this.d[f] = Math.max(0, this.d[f] - m / (this.rhoC * Math.max(this.side, 1e-3)));
    }
    this.updateSide();
    const nsub = Math.max(1, Math.ceil(dt / 0.004));
    const h = dt / nsub;
    let gasTotal = 0;
    const gasFace = [0, 0, 0, 0];
    for (let n = 0; n < nsub; n++) {
      const s = this.side;
      const dS = Math.min(W.deltaS, s / 8), dP = Math.min(W.deltaP, s / 8);
      const allChar = this.mv <= 1e-9;
      // capacities (J/K per m)
      const rcv = this.rhoV * W.cv, rcc = this.rhoC * W.cc;
      const Cs = [], Cp = [];
      let Csum = 0;
      for (let f = 0; f < 4; f++) {
        const phi = allChar ? 1 : Math.min(1, this.d[f] / dS);
        Cs[f] = (rcv * (1 - phi) + rcc * phi) * dS * s;
        Cp[f] = (allChar ? rcc : rcv) * dP * s;
        Csum += Cs[f] + Cp[f];
      }
      const Ctot = this.mv * W.cv + this.mc * W.cc;
      const Cc = Math.max(0.1 * Ctot, Ctot - Csum);
      // pyrolysis
      const A = W.pyroA * ctl.pyroMult;
      const mFrontCap = this.rhoV * dP * s;
      let mFrontSum = 0;
      const rf = [0, 0, 0, 0];
      for (let f = 0; f < 4; f++) {
        const mFront = Math.min(mFrontCap, this.mv / 5);
        mFrontSum += mFront;
        rf[f] = A * Math.exp(-W.pyroTa / this.Tp[f]) * mFront;
      }
      const rCore = A * Math.exp(-W.pyroTa / this.Tc) * Math.max(0, this.mv - mFrontSum);
      let rTot = rf[0] + rf[1] + rf[2] + rf[3] + rCore;
      if (rTot * h > this.mv) {               // do not consume more than exists
        const k = this.mv / (rTot * h);
        for (let f = 0; f < 4; f++) rf[f] *= k;
        rTot *= k;
      }
      const rC = rTot - (rf[0] + rf[1] + rf[2] + rf[3]);
      // conductances
      const kv = allChar ? W.kc : W.kv;
      let coreIn = 0;
      for (let f = 0; f < 4; f++) {
        const d = this.d[f];
        // char conductivity with radiation across pores: k = k0 + 4 eps sigma T^3 d_pore
        const Tm = 0.5 * (this.Ts[f] + this.Tp[f]);
        const kcEff = W.kc + 4 * W.eps * W.sigma * Tm * Tm * Tm * W.pore;
        const Gsp = s / (d / kcEff + (dS + dP) / (2 * kv));
        const Lpc = Math.max(0.5 * (0.5 * s - d), 0.001);
        const Gpc = kv * s / Lpc;
        const gas = (1 - Yc) * (rf[f] + 0.25 * rC);       // kg/s/m leaving this face
        gasFace[f] += gas * h;
        // surface node: convection, radiation, char heat, conduction, transpiration
        const Ts = this.Ts[f], Tp = this.Tp[f];
        const eps = W.eps;
        // radiation only through the exposed part of the face; covered parts exchange
        // heat with what they touch through the contact model (floor: adiabatic)
        const ex = this.exposure[f];
        const qRad = eps * s * ex * (qInc[f] - W.sigma * Ts * Ts * Ts * Ts);
        const dRad = 4 * eps * W.sigma * Ts * Ts * Ts * s * ex;   // linearised for stability
        const qs = qConv[f] + qRad + qChar[f] - Gsp * (Ts - Tp) - gas * W.cpGas * (Ts - Tp);
        this.Ts[f] = Ts + h * qs / (Cs[f] + h * (dRad + Gsp));
        const qp = Gsp * (Ts - Tp) - Gpc * (Tp - this.Tc) - W.dHpyro * rf[f];
        this.Tp[f] = Tp + h * qp / (Cp[f] + h * (Gsp + Gpc));
        coreIn += Gpc * (Tp - this.Tc);
        // char layer grows into the virgin wood
        if (!allChar) this.d[f] = Math.min(0.5 * s, this.d[f] + rf[f] * h / (this.rhoV * s));
      }
      this.Tc += h * (coreIn - W.dHpyro * rC) / Cc;
      this.mv -= rTot * h;
      this.mc += Yc * rTot * h;
      if (this.mv < 1e-9) this.mv = 0;
      gasTotal += (1 - Yc) * rTot * h;
      this.updateSide();
    }
    this.gasRate = gasTotal / dt;
    const s = Math.max(this.side, 1e-4);
    // volatiles leave through the open faces: gas from covered faces is routed
    // (through the porous char) to the exposed ones
    const ex = this.exposure, exSum = ex[0] + ex[1] + ex[2] + ex[3];
    let blocked = 0;
    for (let f = 0; f < 4; f++) blocked += gasFace[f] * (1 - ex[f]);
    for (let f = 0; f < 4; f++) {
      const g = gasFace[f] * ex[f] + (exSum > 0.05 ? blocked * ex[f] / exSum : 0);
      // per unit exposed area, so the GPU (which normalises over the exposed band) injects g
      this.mflux[f] = ex[f] > 0.02 ? g / dt / s : 0;
      this.charCov[f] = this.mv <= 0 ? 1 : Math.min(1, this.d[f] / WOOD.minDepth);
    }
    if (this.half < WOOD.minHalf || (this.mv + this.mc) < 1e-7) this.dead = true;
  }
}
