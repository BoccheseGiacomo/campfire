// 2D rigid-body dynamics for square wood pieces (per unit depth).
//
// Soft-step solver (sub-stepping, soft contacts, relax pass, warm starting),
// two-point box manifolds by SAT + clipping, speculative contacts, and
// Coulomb friction with distinct static/kinetic coefficients. Friction acts on
// the whole contact face (in 2D all manifold points share one contact line, so
// one tangential constraint is exact). Static friction is enforced with
// persistent material anchors so resting contacts cannot creep.

const TAU = Math.PI * 2;

export const RIGID_DEFAULTS = {
  gravity: [0, -9.81],
  width: 0.7,           // wood-confining side walls at x = 0 and x = width
  dt: 1 / 480,          // collision step
  substeps: 4,
  slop: 0.0002,         // m, geometric tolerance
  contactHertz: 240,    // soft contact stiffness (stiff relative to cm-sized pieces)
  contactDamping: 10,   // damping ratio (heavily overdamped: no bounce)
  maxPushout: 0.25,     // m/s, max velocity used to resolve overlap
  muStatic: 0.7,
  muKinetic: 0.5,
  stickSpeed: 0.002,    // m/s, slip speed below which a sliding contact re-sticks
  maxSpeed: 8,          // m/s safety clamp
  dragHertz: 8,         // hand spring stiffness (critically damped)
  dragStrength: 2.5,    // hand force limit beyond carrying its weight, in multiples of the weight
};

function makeSoft(hertz, zeta, h) {
  if (hertz === 0) return { biasRate: 0, massScale: 1, impulseScale: 0 };
  const omega = TAU * hertz;
  const a1 = 2 * zeta + h * omega;
  const a2 = h * omega * a1;
  const a3 = 1 / (1 + a2);
  return { biasRate: omega / a1, massScale: a2 * a3, impulseScale: a3 };
}

let nextBodyId = 1;

export class Body {
  constructor(x, y, half, density, angle = 0) {
    this.id = nextBodyId++;
    this.x = x; this.y = y; this.a = angle;
    this.vx = 0; this.vy = 0; this.w = 0;
    this.density = density;
    this.half = 0;
    this.setHalf(half);
    this.drag = null;    // mouse joint {lx, ly, tx, ty, px, py}
    this.userData = null;
  }
  setHalf(half) {
    const s = 2 * half;
    this.half = half;
    this.mass = this.density * s * s;
    this.invMass = 1 / this.mass;
    this.I = this.mass * s * s / 6;
    this.invI = 1 / this.I;
  }
}

const STATIC = { id: 0, x: 0, y: 0, a: 0, vx: 0, vy: 0, w: 0, invMass: 0, invI: 0, half: 0, isStatic: true };

// ---------- geometry helpers ----------
const LOCAL_V = [[-1, -1], [1, -1], [1, 1], [-1, 1]];   // CCW
const LOCAL_N = [[0, -1], [1, 0], [0, 1], [-1, 0]];     // edge i: v[i] -> v[i+1]

function boxGeom(b, out) {
  const c = Math.cos(b.a), s = Math.sin(b.a), h = b.half;
  for (let i = 0; i < 4; i++) {
    const lx = LOCAL_V[i][0] * h, ly = LOCAL_V[i][1] * h;
    out.v[i][0] = b.x + c * lx - s * ly;
    out.v[i][1] = b.y + s * lx + c * ly;
    const nx = LOCAL_N[i][0], ny = LOCAL_N[i][1];
    out.n[i][0] = c * nx - s * ny;
    out.n[i][1] = s * nx + c * ny;
  }
  return out;
}
const mkGeom = () => ({ v: [[0, 0], [0, 0], [0, 0], [0, 0]], n: [[0, 0], [0, 0], [0, 0], [0, 0]] });
const GA = mkGeom(), GB = mkGeom();

function maxSeparation(ga, gb) {
  let best = -Infinity, edge = 0;
  for (let i = 0; i < 4; i++) {
    const nx = ga.n[i][0], ny = ga.n[i][1];
    const ox = ga.v[i][0], oy = ga.v[i][1];
    let m = Infinity;
    for (let j = 0; j < 4; j++) {
      const d = nx * (gb.v[j][0] - ox) + ny * (gb.v[j][1] - oy);
      if (d < m) m = d;
    }
    if (m > best) { best = m; edge = i; }
  }
  return [best, edge];
}

// Keep the part of segment where dot(n,p) <= off.
function clipSegment(pts, nx, ny, off, clipId) {
  if (pts.length < 2) return pts;
  const [p0, p1] = pts;
  const d0 = nx * p0.x + ny * p0.y - off;
  const d1 = nx * p1.x + ny * p1.y - off;
  const out = [];
  if (d0 <= 0) out.push(p0);
  if (d1 <= 0) out.push(p1);
  if (d0 * d1 < 0) {
    const t = d0 / (d0 - d1);
    out.push({ x: p0.x + t * (p1.x - p0.x), y: p0.y + t * (p1.y - p0.y), id: clipId });
  }
  return out;
}

// Box-box manifold. Normal points from A to B. Returns array of {x,y,sep,id}.
function collideBoxes(A, B, spec, slop) {
  boxGeom(A, GA); boxGeom(B, GB);
  const [sepA, eA] = maxSeparation(GA, GB);
  if (sepA > spec) return null;
  const [sepB, eB] = maxSeparation(GB, GA);
  if (sepB > spec) return null;
  let flip, R, I, e;
  if (sepB > sepA + 0.1 * slop) { flip = 1; R = GB; I = GA; e = eB; }
  else { flip = 0; R = GA; I = GB; e = eA; }
  const nx = R.n[e][0], ny = R.n[e][1];
  // incident edge: most anti-parallel normal
  let j = 0, md = Infinity;
  for (let k = 0; k < 4; k++) {
    const d = I.n[k][0] * nx + I.n[k][1] * ny;
    if (d < md) { md = d; j = k; }
  }
  const j2 = (j + 1) & 3;
  const r1 = R.v[e], r2 = R.v[(e + 1) & 3];
  let tx = r2[0] - r1[0], ty = r2[1] - r1[1];
  const tl = Math.hypot(tx, ty); tx /= tl; ty /= tl;
  const base = (flip << 6) | (e << 3);
  let pts = [
    { x: I.v[j][0], y: I.v[j][1], id: base | j },
    { x: I.v[j2][0], y: I.v[j2][1], id: base | j2 },
  ];
  pts = clipSegment(pts, -tx, -ty, -(tx * r1[0] + ty * r1[1]), base | 4);
  pts = clipSegment(pts, tx, ty, tx * r2[0] + ty * r2[1], base | 5);
  const out = [];
  for (const p of pts) {
    const sep = nx * (p.x - r1[0]) + ny * (p.y - r1[1]);
    if (sep <= spec) out.push({ x: p.x - 0.5 * sep * nx, y: p.y - 0.5 * sep * ny, sep, id: p.id });
  }
  if (!out.length) return null;
  return { nx: flip ? -nx : nx, ny: flip ? -ny : ny, points: out };
}

// Box vs half-plane {p : dot(n,p) >= off} (static A). Normal from plane to box.
function collidePlane(B, nx, ny, off, spec) {
  boxGeom(B, GB);
  const out = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = GB.v[i];
    const sep = nx * x + ny * y - off;
    if (sep <= spec) out.push({ x: x - 0.5 * sep * nx, y: y - 0.5 * sep * ny, sep, id: i });
  }
  return out.length ? { nx, ny, points: out } : null;
}

// ---------- world ----------
export class RigidWorld {
  constructor(opts = {}) {
    this.p = { ...RIGID_DEFAULTS, ...opts };
    this.bodies = [];
    this.manifolds = new Map();  // key -> manifold
    this.time = 0;
    this.accum = 0;
  }

  add(body) { this.bodies.push(body); return body; }
  remove(body) {
    const i = this.bodies.indexOf(body);
    if (i >= 0) this.bodies.splice(i, 1);
    for (const [k, m] of this.manifolds) if (m.A === body || m.B === body) this.manifolds.delete(k);
  }

  // Shrink/grow a body about its centre; friction anchors are material points
  // and are scaled with it so shrinkage itself creates no tangential drift.
  resize(body, half) {
    const f = half / body.half;
    if (f === 1) return;
    for (const m of this.manifolds.values()) {
      if (m.A === body) { m.aAx *= f; m.aAy *= f; }
      if (m.B === body) { m.aBx *= f; m.aBy *= f; }
    }
    body.setHalf(half);
  }

  // Advance by real time dt using fixed collision steps.
  advance(dt) {
    this.accum += dt;
    const h = this.p.dt;
    let n = 0;
    while (this.accum >= h && n < 64) { this.step(h); this.accum -= h; n++; }
    if (n === 64) this.accum = 0;
  }

  collide(dt) {
    const P = this.p, B = this.bodies;
    const old = this.manifolds;
    const fresh = new Map();
    const spec0 = 4 * P.slop;
    const take = (key, A, Bb, man) => {
      const prev = old.get(key);
      const m = {
        A, B: Bb, nx: man.nx, ny: man.ny, points: [],
        cx: 0, cy: 0, Pt: 0, sticking: true, aAx: 0, aAy: 0, aBx: 0, aBy: 0,
      };
      const used = [];
      for (const p of man.points) {
        const c = { id: p.id, x: p.x, y: p.y, sep: p.sep, Pn: 0 };
        if (prev) {
          // warm start: same feature, else the nearest previous point (clip
          // points and vertices of equal-sized faces swap ids under tiny shifts)
          let q = prev.points.find(q => q.id === p.id && !used.includes(q));
          if (!q) {
            let best = 0.25 * Math.min(A.half || Bb.half, Bb.half);
            for (const r of prev.points) {
              const d = Math.hypot(r.x - p.x, r.y - p.y);
              if (d < best && !used.includes(r)) { best = d; q = r; }
            }
          }
          if (q) { c.Pn = q.Pn; used.push(q); }
        }
        m.points.push(c);
        m.cx += p.x / man.points.length; m.cy += p.y / man.points.length;
      }
      if (prev && prev.nx * m.nx + prev.ny * m.ny > 0.99) {
        m.Pt = prev.Pt; m.sticking = prev.sticking;
        m.aAx = prev.aAx; m.aAy = prev.aAy; m.aBx = prev.aBx; m.aBy = prev.aBy;
      } else {
        this.setAnchors(A, Bb, m, m.cx, m.cy);
      }
      fresh.set(key, m);
    };
    const W = P.width;
    for (let i = 0; i < B.length; i++) {
      const b = B[i];
      const speed = Math.hypot(b.vx, b.vy) + Math.abs(b.w) * b.half * 1.5;
      const spec = spec0 + speed * dt;
      let m;
      if ((m = collidePlane(b, 0, 1, 0, spec))) take('f' + b.id, STATIC, b, m);
      if ((m = collidePlane(b, 1, 0, 0, spec))) take('l' + b.id, STATIC, b, m);
      if ((m = collidePlane(b, -1, 0, -W, spec))) take('r' + b.id, STATIC, b, m);
    }
    for (let i = 0; i < B.length; i++) {
      for (let j = i + 1; j < B.length; j++) {
        let a = B[i], b = B[j];
        if (a.id > b.id) { const t = a; a = b; b = t; }
        const rel = Math.hypot(b.vx - a.vx, b.vy - a.vy) + (Math.abs(a.w) * a.half + Math.abs(b.w) * b.half) * 1.5;
        const spec = spec0 + rel * dt;
        const reach = (a.half + b.half) * 1.4143 + spec;
        if (Math.abs(a.x - b.x) > reach || Math.abs(a.y - b.y) > reach) continue;
        const m = collideBoxes(a, b, spec, P.slop);
        if (m) take(a.id + '_' + b.id, a, b, m);
      }
    }
    this.manifolds = fresh;
  }

  // Friction anchors: the contact point expressed as a material point of each body.
  setAnchors(A, B, c, px = c.x, py = c.y) {
    let dx = px - A.x, dy = py - A.y, ca = Math.cos(A.a), sa = Math.sin(A.a);
    c.aAx = ca * dx + sa * dy; c.aAy = -sa * dx + ca * dy;
    dx = px - B.x; dy = py - B.y; ca = Math.cos(B.a); sa = Math.sin(B.a);
    c.aBx = ca * dx + sa * dy; c.aBy = -sa * dx + ca * dy;
  }

  step(dt) {
    const P = this.p;
    this.collide(dt);
    const ns = P.substeps, h = dt / ns, inv_h = 1 / h;
    const soft = makeSoft(Math.min(P.contactHertz, 0.25 / h), P.contactDamping, h);
    const dragSoft = makeSoft(P.dragHertz, 1, h);
    const [gx, gy] = P.gravity;

    // prepare
    const cons = [];
    for (const m of this.manifolds.values()) {
      const A = m.A, B = m.B, nx = m.nx, ny = m.ny, tx = ny, ty = -nx;
      const mA = A.invMass, mB = B.invMass, iA = A.invI, iB = B.invI;
      for (const c of m.points) {
        const rAx = c.x - A.x, rAy = c.y - A.y, rBx = c.x - B.x, rBy = c.y - B.y;
        const ca = Math.cos(A.a), sa = Math.sin(A.a), cb = Math.cos(B.a), sb = Math.sin(B.a);
        c.lAx = ca * rAx + sa * rAy; c.lAy = -sa * rAx + ca * rAy;
        c.lBx = cb * rBx + sb * rBy; c.lBy = -sb * rBx + cb * rBy;
        c.rAx = rAx; c.rAy = rAy; c.rBx = rBx; c.rBy = rBy;
        const rnA = rAx * ny - rAy * nx, rnB = rBx * ny - rBy * nx;
        c.nMass = 1 / (mA + mB + iA * rnA * rnA + iB * rnB * rnB);
      }
      m.rAx = m.cx - A.x; m.rAy = m.cy - A.y; m.rBx = m.cx - B.x; m.rBy = m.cy - B.y;
      const rtA = m.rAx * ty - m.rAy * tx, rtB = m.rBx * ty - m.rBy * tx;
      m.tMass = 1 / (mA + mB + iA * rtA * rtA + iB * rtB * rtB);
      cons.push(m);
    }
    const dragged = this.bodies.filter(b => b.drag);

    for (let s = 0; s < ns; s++) {
      // integrate velocities
      for (const b of this.bodies) {
        // a held piece's weight is carried by the hand
        if (b.drag) continue;
        b.vx += gx * h; b.vy += gy * h;
      }
      // warm start
      for (const m of cons) {
        const A = m.A, B = m.B, tx = m.ny, ty = -m.nx;
        for (const c of m.points) applyImpulse(A, B, c, c.Pn * m.nx, c.Pn * m.ny);
        applyImpulse(A, B, m, m.Pt * tx, m.Pt * ty);
      }
      this.solveDrag(dragged, dragSoft, h, true);
      this.solveContacts(cons, soft, h, inv_h, true);
      // integrate positions
      const vmax = P.maxSpeed;
      for (const b of this.bodies) {
        const sp = Math.hypot(b.vx, b.vy);
        if (sp > vmax) { b.vx *= vmax / sp; b.vy *= vmax / sp; }
        b.x += b.vx * h; b.y += b.vy * h; b.a += b.w * h;
      }
      // relax (the hand is a soft spring: it is not re-solved rigidly here)
      this.solveContacts(cons, soft, h, inv_h, false);
    }

    // stick/slip state update and anchor maintenance
    for (const m of cons) {
      const A = m.A, B = m.B, tx = m.ny, ty = -m.nx;
      let Pn = 0;
      for (const c of m.points) Pn += c.Pn;
      if (m.sticking) {
        // static friction saturated over the whole step -> the face breaks loose
        if (Pn > 0 && Math.abs(m.Pt) >= 0.999 * P.muStatic * Pn) m.sticking = false;
        continue;
      }
      const dvx = B.vx - B.w * m.rBy - A.vx + A.w * m.rAy;
      const dvy = B.vy + B.w * m.rBx - A.vy - A.w * m.rAx;
      if (Math.abs(dvx * tx + dvy * ty) < P.stickSpeed) {
        // slip has stopped: re-anchor at the current contact centre
        m.sticking = true;
        this.setAnchors(A, B, m, m.cx, m.cy);
      }
    }
    this.time += dt;
  }

  // Hand grip: a soft, critically damped spring pulling the piece's centre to
  // the target and holding its angle, with a hand-strength force/torque limit
  // applied per sub-step (so it can hold a piece indefinitely).
  solveDrag(dragged, soft, h) {
    for (const b of dragged) {
      const d = b.drag;
      const mi = b.invMass, ii = b.invI;
      // linear
      const Cx = b.x - d.cx, Cy = b.y - d.cy;
      let ix = -soft.massScale * b.mass * (b.vx + soft.biasRate * Cx);
      let iy = -soft.massScale * b.mass * (b.vy + soft.biasRate * Cy);
      const maxI = d.maxForce * h, L = Math.hypot(ix, iy);
      if (L > maxI) { ix *= maxI / L; iy *= maxI / L; }
      b.vx += mi * ix; b.vy += mi * iy;
      // angular: keep the angle it was picked up with
      let da = b.a - d.a0;
      da -= 2 * Math.PI * Math.round(da / (2 * Math.PI));
      let ia = -soft.massScale * b.I * (b.w + soft.biasRate * da);
      const maxA = d.maxTorque * h;
      ia = Math.max(-maxA, Math.min(maxA, ia));
      b.w += ii * ia;
    }
  }

  solveContacts(cons, soft, h, inv_h, useBias) {
    const P = this.p;
    for (const m of cons) {
      const A = m.A, B = m.B, nx = m.nx, ny = m.ny, tx = ny, ty = -nx;
      const cA = Math.cos(A.a), sA = Math.sin(A.a), cB = Math.cos(B.a), sB = Math.sin(B.a);
      // friction on the whole face (uses previous normal impulses), then non-penetration
      {
        let Pn = 0;
        for (const c of m.points) Pn += c.Pn;
        const dvx = B.vx - B.w * m.rBy - A.vx + A.w * m.rAy;
        const dvy = B.vy + B.w * m.rBx - A.vy - A.w * m.rAx;
        const vt = dvx * tx + dvy * ty;
        let newP;
        if (m.sticking) {
          let bias = 0, ms = 1, is = 0;
          if (useBias) {
            const pAx = A.x + cA * m.aAx - sA * m.aAy, pAy = A.y + sA * m.aAx + cA * m.aAy;
            const pBx = B.x + cB * m.aBx - sB * m.aBy, pBy = B.y + sB * m.aBx + cB * m.aBy;
            const drift = (pBx - pAx) * tx + (pBy - pAy) * ty;
            bias = Math.max(-P.maxPushout, Math.min(P.maxPushout, soft.biasRate * drift));
            ms = soft.massScale; is = soft.impulseScale;
          }
          const maxS = P.muStatic * Pn;
          newP = m.Pt - m.tMass * ms * (vt + bias) - is * m.Pt;
          newP = Math.max(-maxS, Math.min(maxS, newP));
        } else {
          const maxK = P.muKinetic * Pn;
          newP = Math.max(-maxK, Math.min(maxK, m.Pt - m.tMass * vt));
        }
        const imp = newP - m.Pt; m.Pt = newP;
        applyImpulse(A, B, m, imp * tx, imp * ty);
      }
      for (const c of m.points) {
        // current separation from current poses
        const pAx = A.x + cA * c.lAx - sA * c.lAy, pAy = A.y + sA * c.lAx + cA * c.lAy;
        const pBx = B.x + cB * c.lBx - sB * c.lBy, pBy = B.y + sB * c.lBx + cB * c.lBy;
        const sep = (pBx - pAx) * nx + (pBy - pAy) * ny + c.sep;
        let bias = 0, ms = 1, is = 0;
        if (sep > 0) bias = sep * inv_h;
        else if (useBias) {
          bias = Math.max(soft.biasRate * sep, -P.maxPushout);
          ms = soft.massScale; is = soft.impulseScale;
        }
        const dvx = B.vx - B.w * c.rBy - A.vx + A.w * c.rAy;
        const dvy = B.vy + B.w * c.rBx - A.vy - A.w * c.rAx;
        const vn = dvx * nx + dvy * ny;
        let imp = -c.nMass * ms * (vn + bias) - is * c.Pn;
        const newP = Math.max(c.Pn + imp, 0);
        imp = newP - c.Pn; c.Pn = newP;
        applyImpulse(A, B, c, imp * nx, imp * ny);
      }
    }
  }

  // Pick the topmost body under a point.
  pick(x, y) {
    for (let i = this.bodies.length - 1; i >= 0; i--) {
      const b = this.bodies[i];
      const dx = x - b.x, dy = y - b.y, c = Math.cos(b.a), s = Math.sin(b.a);
      const lx = c * dx + s * dy, ly = -s * dx + c * dy;
      if (Math.abs(lx) <= b.half && Math.abs(ly) <= b.half) return { body: b, lx, ly };
    }
    return null;
  }

  // Grab a body at local point (lx, ly); the hand target is the world point (tx, ty).
  startDrag(body, lx, ly, tx, ty) {
    const g = this.p.gravity[1] ? Math.abs(this.p.gravity[1]) : 9.81;
    body.drag = {
      lx, ly, a0: body.a, cx: body.x, cy: body.y,
      maxForce: this.p.dragStrength * body.mass * g,
      maxTorque: this.p.dragStrength * body.mass * g * body.half,
    };
    this.moveDrag(body, tx, ty);
  }
  moveDrag(body, tx, ty) {
    const d = body.drag;
    if (!d) return;
    // grab point follows the pointer; the angle is held, so the centre target is fixed relative to it
    const c = Math.cos(d.a0), s = Math.sin(d.a0);
    const ox = c * d.lx - s * d.ly, oy = s * d.lx + c * d.ly;
    const h = body.half * 1.42, W = this.p.width;
    d.cx = Math.min(W - h, Math.max(h, tx - ox));
    d.cy = Math.max(h, ty - oy);
  }
  // release: the piece drops straight down from rest
  endDrag(body) {
    if (!body.drag) return;
    body.drag = null;
    body.vx = 0; body.vy = 0; body.w = 0;
  }

  // Diagnostics
  maxPenetration() {
    let m = 0;
    for (const man of this.manifolds.values()) for (const c of man.points) m = Math.max(m, -c.sep);
    return m;
  }
}

function applyImpulse(A, B, c, Px, Py) {
  A.vx -= A.invMass * Px; A.vy -= A.invMass * Py;
  A.w -= A.invI * (c.rAx * Py - c.rAy * Px);
  B.vx += B.invMass * Px; B.vy += B.invMass * Py;
  B.w += B.invI * (c.rBx * Py - c.rBy * Px);
}
