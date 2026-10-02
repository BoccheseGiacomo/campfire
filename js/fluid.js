// GPU gas solver: low-Mach variable-density equations on a MAC grid.
import { Kernel, loadWGSL, makeBuffer } from './gpu.js';
import { PHYS, SCENE_WIDTH } from './config.js';

export const MAX_PIECES = 64;
export const PIECE_FLOATS = 20;
export const ACC_GLOBAL = 512;
export const ACC_LEN = ACC_GLOBAL + 16;
export const E_SCALE = 1e6, M_SCALE = 1e12, G_SCALE = 1e3;
export const N_FLOOR = 64;
export const RAD_LEN = MAX_PIECES * 4 + N_FLOOR;   // RadOut entries (4 floats)

const PARAM_FIELDS = [
  ['nx', 'i'], ['ny', 'i'], ['dx', 'f'], ['dt', 'f'],
  ['Tamb', 'f'], ['rhoAmb', 'f'], ['YOamb', 'f'], ['grav', 'f'],
  ['cp', 'f'], ['Pr', 'f'], ['viscMult', 'f'], ['diffMult', 'f'],
  ['reactB', 'f'], ['reactTa', 'f'], ['dHc', 'f'], ['sO2', 'f'],
  ['sootFormA', 'f'], ['sootFormTa', 'f'], ['sootOxA', 'f'], ['sootOxTa', 'f'],
  ['kappaSootC', 'f'], ['kappaGas', 'f'], ['exchAlpha', 'f'], ['depth', 'f'],
  ['charA', 'f'], ['charTa', 'f'], ['nPieces', 'i'], ['frame', 'u'],
  ['heaterX0', 'f'], ['heaterX1', 'f'], ['heaterY1', 'f'], ['heaterT', 'f'],
  ['burnerX0', 'f'], ['burnerX1', 'f'], ['burnerFlux', 'f'], ['burnerT', 'f'],
  ['sootDensity', 'f'], ['sigma', 'f'], ['charO2', 'f'], ['time', 'f'],
];

export class FluidSim {
  constructor(device, mode) {
    this.device = device;
    this.nx = mode.nx; this.ny = mode.ny;
    this.dx = SCENE_WIDTH / mode.nx;
    this.height = this.dx * mode.ny;
    this.dt = mode.dt;
    this.time = 0;
    this.frame = 0;
    this.params = {
      Tamb: PHYS.Tamb, rhoAmb: PHYS.rhoAmb, YOamb: PHYS.YO2air, grav: PHYS.g,
      cp: PHYS.cp, Pr: PHYS.Pr, viscMult: 1, diffMult: 1,
      reactB: PHYS.reactB, reactTa: PHYS.reactTa, dHc: PHYS.dHc, sO2: PHYS.sO2,
      sootFormA: PHYS.sootFormA, sootFormTa: PHYS.sootFormTa, sootOxA: PHYS.sootOxA, sootOxTa: PHYS.sootOxTa,
      kappaSootC: PHYS.kappaSootC, kappaGas: PHYS.kappaGas, exchAlpha: PHYS.exchAlpha, depth: 0.8,
      charA: PHYS.charA, charTa: PHYS.charTa, nPieces: 0, frame: 0,
      heaterX0: 0, heaterX1: 0, heaterY1: 0, heaterT: 0,
      burnerX0: 0, burnerX1: 0, burnerFlux: 0, burnerT: 600,
      sootDensity: PHYS.sootDensity, sigma: PHYS.sigma, charO2: PHYS.charO2, time: 0,
    };
    this.diffSub = 1;
  }

  async init() {
    const d = this.device, nx = this.nx, ny = this.ny;
    const nC = nx * ny, nU = (nx + 1) * ny, nV = nx * (ny + 1);
    const B = this.buf = {};
    const f32 = 4;
    for (const n of ['U', 'Uh', 'U2']) B[n] = makeBuffer(d, n, nU * f32);
    for (const n of ['V', 'Vh', 'V2']) B[n] = makeBuffer(d, n, nV * f32);
    for (const n of ['T', 'Th', 'T2']) B[n] = makeBuffer(d, n, nC * f32);
    for (const n of ['Y', 'Yh', 'Y2']) B[n] = makeBuffer(d, n, nC * 16);
    B.SQ = makeBuffer(d, 'SQ', nC * 8);
    B.cellG = makeBuffer(d, 'cellG', nC * 8);
    B.fx = makeBuffer(d, 'fx', nU * f32);
    B.fy = makeBuffer(d, 'fy', nV * f32);
    B.pieces = makeBuffer(d, 'pieces', MAX_PIECES * PIECE_FLOATS * 4);
    B.acc = makeBuffer(d, 'acc', ACC_LEN * 4);
    B.rad = makeBuffer(d, 'rad', RAD_LEN * 16);
    B.faceW = makeBuffer(d, 'faceW', MAX_PIECES * 4 * 4);
    B.P = makeBuffer(d, 'params', PARAM_FIELDS.length * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.diffU = [];
    for (let s = 0; s < 3; s++) this.diffU.push(makeBuffer(d, 'diffParams', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
    // multigrid levels
    this.levels = [];
    let lx = nx, ly = ny, h = this.dx;
    for (let l = 0; l < 6; l++) {
      const L = {
        nx: lx, ny: ly, h,
        p: makeBuffer(d, 'p' + l, lx * ly * f32),
        rhs: makeBuffer(d, 'rhs' + l, lx * ly * f32),
        ax: makeBuffer(d, 'ax' + l, (lx + 1) * ly * f32),
        ay: makeBuffer(d, 'ay' + l, lx * (ly + 1) * f32),
        u: [0, 1].map(c => {
          const b = makeBuffer(d, `lvl${l}c${c}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          const dv = new DataView(new ArrayBuffer(16));
          dv.setInt32(0, lx, true); dv.setInt32(4, ly, true); dv.setFloat32(8, h * h, true); dv.setInt32(12, c, true);
          d.queue.writeBuffer(b, 0, dv.buffer);
          return b;
        }),
      };
      this.levels.push(L);
      if (lx % 2 || ly % 2 || lx * ly <= 256) break;
      lx /= 2; ly /= 2; h *= 2;
    }
    if (this.levels.at(-1).nx * this.levels.at(-1).ny > 256) throw new Error('coarsest level too large');

    // kernels
    const adv = await loadWGSL('shaders/advect.wgsl');
    const dif = await loadWGSL('shaders/diffuse.wgsl');
    const src = await loadWGSL('shaders/sources.wgsl');
    const frc = await loadWGSL('shaders/forces.wgsl');
    const prs = await loadWGSL('shaders/pressure.wgsl');
    const geo = await loadWGSL('shaders/geometry.wgsl');
    const rad = await loadWGSL('shaders/radiation.wgsl');
    const K = this.k = {
      velForward: new Kernel(d, adv, 'velForward'),
      velCorrect: new Kernel(d, adv, 'velCorrect'),
      scalForward: new Kernel(d, adv, 'scalForward'),
      scalCorrect: new Kernel(d, adv, 'scalCorrect'),
      diffuse: new Kernel(d, dif, 'main', 'diffuse'),
      sources: new Kernel(d, src, 'main', 'sources'),
      forces: new Kernel(d, frc, 'main', 'forces'),
      coef: new Kernel(d, prs, 'coef'),
      rhs0: new Kernel(d, prs, 'rhs0'),
      coefRestrict: new Kernel(d, prs, 'coefRestrict'),
      smooth: new Kernel(d, prs, 'relax'),
      restrictRes: new Kernel(d, prs, 'restrictRes'),
      prolong: new Kernel(d, prs, 'prolong'),
      coarsest: new Kernel(d, prs, 'coarsest'),
      project: new Kernel(d, prs, 'project'),
      geometry: new Kernel(d, geo, 'main', 'geometry'),
      radiation: new Kernel(d, rad, 'main', 'radiation'),
    };
    const L0 = this.levels[0];
    const base = { ...B, p: L0.p, rhs: L0.rhs, ax: L0.ax, ay: L0.ay };
    const G = this.bg = {};
    G.velForward = K.velForward.bind(base);
    G.velCorrect = K.velCorrect.bind(base);
    G.scalForward = K.scalForward.bind(base);
    G.scalCorrect = K.scalCorrect.bind(base);
    G.diff = [
      K.diffuse.bind({ P: B.P, D: this.diffU[0], Tin: B.T2, Yin: B.Y2, Tout: B.T, Yout: B.Y, fx: B.fx, fy: B.fy }),
      K.diffuse.bind({ P: B.P, D: this.diffU[1], Tin: B.T, Yin: B.Y, Tout: B.Th, Yout: B.Yh, fx: B.fx, fy: B.fy }),
      K.diffuse.bind({ P: B.P, D: this.diffU[2], Tin: B.Th, Yin: B.Yh, Tout: B.T, Yout: B.Y, fx: B.fx, fy: B.fy }),
    ];
    G.sources = K.sources.bind({ ...B, Tadv: B.T2 });
    G.forces = K.forces.bind(base);
    G.coef = K.coef.bind(base);
    G.rhs0 = K.rhs0.bind(base);
    G.project = K.project.bind(base);
    G.geometry = K.geometry.bind(B);
    G.radiation = K.radiation.bind(B);
    G.lv = this.levels.map((L, l) => {
      const C = this.levels[l + 1];
      const m = { P: B.P, p: L.p, rhs: L.rhs, ax: L.ax, ay: L.ay };
      const g = {
        smooth: [0, 1].map(c => K.smooth.bind({ ...m, L: L.u[c] })),
      };
      if (C) {
        const mc = { ...m, L: L.u[0], pc: C.p, rhsc: C.rhs, axc: C.ax, ayc: C.ay };
        g.coefRestrict = K.coefRestrict.bind(mc);
        g.restrictRes = K.restrictRes.bind(mc);
        g.prolong = K.prolong.bind(mc);
      } else {
        g.coarsest = K.coarsest.bind({ ...m, L: L.u[0] });
      }
      return g;
    });
    this.reset();
    // staging ring for accumulator readback
    this.staging = [0, 1, 2].map(i => d.createBuffer({ label: 'stage' + i, size: ACC_LEN * 4 + RAD_LEN * 16 + MAX_PIECES * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    this.stageBusy = [false, false, false];
    this.stageIdx = 0;
  }

  reset() {
    const d = this.device, nC = this.nx * this.ny;
    const T = new Float32Array(nC).fill(PHYS.Tamb);
    d.queue.writeBuffer(this.buf.T, 0, T);
    d.queue.writeBuffer(this.buf.T2, 0, T);
    d.queue.writeBuffer(this.buf.Th, 0, T);
    const Y = new Float32Array(nC * 4);
    for (let i = 0; i < nC; i++) Y[i * 4 + 1] = this.params.YOamb;
    for (const n of ['Y', 'Y2', 'Yh']) d.queue.writeBuffer(this.buf[n], 0, Y);
    for (const n of ['U', 'Uh', 'U2', 'V', 'Vh', 'V2', 'SQ', 'acc', 'rad']) {
      d.queue.writeBuffer(this.buf[n], 0, new Float32Array(this.buf[n].size / 4));
    }
    for (const L of this.levels) d.queue.writeBuffer(L.p, 0, new Float32Array(L.p.size / 4));
    this.time = 0;
  }

  setParam(k, v) { this.params[k] = v; }

  writeParams() {
    const dv = new DataView(new ArrayBuffer(PARAM_FIELDS.length * 4));
    const p = this.params;
    p.nx = this.nx; p.ny = this.ny; p.dx = this.dx; p.dt = this.dt; p.time = this.time; p.frame = this.frame;
    PARAM_FIELDS.forEach(([k, t], i) => {
      if (t === 'f') dv.setFloat32(i * 4, p[k], true);
      else if (t === 'i') dv.setInt32(i * 4, p[k], true);
      else dv.setUint32(i * 4, p[k] >>> 0, true);
    });
    this.device.queue.writeBuffer(this.buf.P, 0, dv.buffer);
    // explicit diffusion stability: dt_sub <= dx^2 / (4.5 alpha_max)
    const Tmax = 2200, mu = 1.716e-5 * Math.pow(Tmax / 273.15, 1.5) * 383.55 / (Tmax + 110.4);
    const alpha = mu / PHYS.Pr * p.diffMult / (PHYS.rhoAmb * PHYS.Tamb / Tmax);
    let n = Math.ceil(this.dt / (this.dx * this.dx / (4.5 * alpha)));
    n = Math.min(5, n % 2 ? n : n + 1);       // odd so the result lands in T/Y
    this.diffSub = n;
    for (let s = 0; s < 3; s++) {
      this.device.queue.writeBuffer(this.diffU[s], 0, new Float32Array([this.dt / n, 0, 0, 0]));
    }
  }

  uploadPieces(data, n) {
    this.params.nPieces = n;
    this.device.queue.writeBuffer(this.buf.pieces, 0, data);
  }

  dispatch(pass, kernel, bg, wx, wy = 1) {
    pass.setPipeline(kernel.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wx, wy);
  }

  // must be called on the encoder before the pass that runs encodeGeometry
  clearGeometry(encoder) { encoder.clearBuffer(this.buf.faceW); }

  encodeGeometry(pass) {
    const g = [Math.ceil((this.nx + 1) / 8), Math.ceil((this.ny + 1) / 8)];
    this.dispatch(pass, this.k.geometry, this.bg.geometry, ...g);
  }

  encodeRadiation(pass) {
    this.dispatch(pass, this.k.radiation, this.bg.radiation, this.params.nPieces * 4 + N_FLOOR);
  }

  // one time step of the gas
  encodeStep(pass) {
    const K = this.k, G = this.bg;
    const gF = [Math.ceil((this.nx + 1) / 8), Math.ceil((this.ny + 1) / 8)];
    const gC = [Math.ceil(this.nx / 8), Math.ceil(this.ny / 8)];
    // advection (all with the old velocity)
    this.dispatch(pass, K.scalForward, G.scalForward, ...gC);
    this.dispatch(pass, K.scalCorrect, G.scalCorrect, ...gC);
    this.dispatch(pass, K.velForward, G.velForward, ...gF);
    this.dispatch(pass, K.velCorrect, G.velCorrect, ...gF);
    // diffusion T2 -> T (odd number of sub-steps)
    this.dispatch(pass, K.diffuse, G.diff[0], ...gC);
    for (let s = 1; s < this.diffSub; s += 2) {
      this.dispatch(pass, K.diffuse, G.diff[1], ...gC);
      this.dispatch(pass, K.diffuse, G.diff[2], ...gC);
    }
    this.dispatch(pass, K.sources, G.sources, ...gC);
    this.dispatch(pass, K.forces, G.forces, ...gF);
    // projection
    this.dispatch(pass, K.coef, G.coef, ...gF);
    this.dispatch(pass, K.rhs0, G.rhs0, ...gC);
    const Ls = this.levels;
    for (let l = 0; l < Ls.length - 1; l++) {
      const L = Ls[l];
      this.dispatch(pass, K.coefRestrict, G.lv[l].coefRestrict, Math.ceil((L.nx / 2 + 1) / 8), Math.ceil((L.ny / 2 + 1) / 8));
    }
    this.vcycle(pass);
    this.dispatch(pass, K.project, G.project, ...gF);
    this.time += this.dt;
  }

  // Profiling: same work as encodeStep, one compute pass per stage with timestamps.
  encodeStepProfiled(encoder, qset, base) {
    const K = this.k, G = this.bg;
    const gF = [Math.ceil((this.nx + 1) / 8), Math.ceil((this.ny + 1) / 8)];
    const gC = [Math.ceil(this.nx / 8), Math.ceil(this.ny / 8)];
    let q = base;
    const stage = fn => {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: qset, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } });
      fn(pass); pass.end(); q += 2;
    };
    stage(p => {
      this.dispatch(p, K.scalForward, G.scalForward, ...gC);
      this.dispatch(p, K.scalCorrect, G.scalCorrect, ...gC);
      this.dispatch(p, K.velForward, G.velForward, ...gF);
      this.dispatch(p, K.velCorrect, G.velCorrect, ...gF);
    });
    stage(p => {
      this.dispatch(p, K.diffuse, G.diff[0], ...gC);
      for (let s = 1; s < this.diffSub; s += 2) {
        this.dispatch(p, K.diffuse, G.diff[1], ...gC);
        this.dispatch(p, K.diffuse, G.diff[2], ...gC);
      }
    });
    stage(p => this.dispatch(p, K.sources, G.sources, ...gC));
    stage(p => this.dispatch(p, K.forces, G.forces, ...gF));
    stage(p => {
      this.dispatch(p, K.coef, G.coef, ...gF);
      this.dispatch(p, K.rhs0, G.rhs0, ...gC);
      for (let l = 0; l < this.levels.length - 1; l++) {
        const L = this.levels[l];
        this.dispatch(p, K.coefRestrict, G.lv[l].coefRestrict, Math.ceil((L.nx / 2 + 1) / 8), Math.ceil((L.ny / 2 + 1) / 8));
      }
    });
    stage(p => this.vcycle(p));
    stage(p => this.dispatch(p, K.project, G.project, ...gF));
    this.time += this.dt;
    return ['advect', 'diffuse', 'sources', 'forces', 'coef/rhs', 'vcycle', 'project'];
  }

  vcycle(pass, nu = 2) {
    const K = this.k, G = this.bg, Ls = this.levels;
    const g = L => [Math.ceil(L.nx / 8), Math.ceil(L.ny / 8)];
    for (let l = 0; l < Ls.length - 1; l++) {
      for (let s = 0; s < nu; s++) {
        this.dispatch(pass, K.smooth, G.lv[l].smooth[0], ...g(Ls[l]));
        this.dispatch(pass, K.smooth, G.lv[l].smooth[1], ...g(Ls[l]));
      }
      this.dispatch(pass, K.restrictRes, G.lv[l].restrictRes, ...g(Ls[l + 1]));
    }
    this.dispatch(pass, K.coarsest, G.lv[Ls.length - 1].coarsest, 1);
    for (let l = Ls.length - 2; l >= 0; l--) {
      this.dispatch(pass, K.prolong, G.lv[l].prolong, ...g(Ls[l]));
      for (let s = 0; s < nu; s++) {
        this.dispatch(pass, K.smooth, G.lv[l].smooth[1], ...g(Ls[l]));
        this.dispatch(pass, K.smooth, G.lv[l].smooth[0], ...g(Ls[l]));
      }
    }
  }

  // copy accumulators to a free staging buffer and clear them
  encodeReadback(encoder) {
    const i = this.stageIdx;
    if (this.stageBusy[i]) { this.pendingStage = -1; return; }
    encoder.copyBufferToBuffer(this.buf.acc, 0, this.staging[i], 0, ACC_LEN * 4);
    encoder.copyBufferToBuffer(this.buf.rad, 0, this.staging[i], ACC_LEN * 4, RAD_LEN * 16);
    encoder.copyBufferToBuffer(this.buf.faceW, 0, this.staging[i], ACC_LEN * 4 + RAD_LEN * 16, MAX_PIECES * 16);
    encoder.clearBuffer(this.buf.acc);
    this.pendingStage = i;
    this.stageIdx = (i + 1) % 3;
  }

  // after submit: map the staging buffer; callback gets ({acc: Int32Array, rad: Float32Array}, tag)
  afterSubmit(cb, tag) {
    const i = this.pendingStage;
    if (i < 0) return;
    this.stageBusy[i] = true;
    const buf = this.staging[i];
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const all = buf.getMappedRange().slice(0);
      buf.unmap();
      this.stageBusy[i] = false;
      cb({
        acc: new Int32Array(all, 0, ACC_LEN),
        rad: new Float32Array(all, ACC_LEN * 4, RAD_LEN * 4),
        faceW: new Int32Array(all, ACC_LEN * 4 + RAD_LEN * 16, MAX_PIECES * 4),
      }, tag);
    }).catch(() => { this.stageBusy[i] = false; });
  }

  // debug: read a buffer back as Float32Array
  async readBuffer(name) {
    const src = this.buf[name] || this.levels[0][name];
    const st = this.device.createBuffer({ size: src.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(src, 0, st, 0, src.size);
    this.device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }
}
