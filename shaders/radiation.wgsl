// Radiation onto wood faces and floor probes, by ray casting in the slice.
//
// The geometry is uniform along the unseen axis, so the polar integral can be
// done analytically. For a direction phi in the plane, with cos(theta_n) the
// in-plane cosine to the face normal:
//   opaque surface at Tw seen along phi:   dq = (sigma Tw^4 / 2) cos dphi
//   optically thin gas emission:           dq = 2 J cos dphi,  J = int k sigma T^4/pi ds
// (these integrate to sigma T^4 for a black enclosure). Rays hit other pieces
// (their face temperature), reflect once off the adiabatic floor, or leave the
// domain to ambient surroundings. A visible-band irradiance (soot emission at
// 580 nm) is also returned for lighting in the renderer.
//#include common

struct RadOut { q: f32, vis: f32, p0: f32, p1: f32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> T: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<vec4f>;
@group(0) @binding(3) var<storage, read> pieces: array<Piece>;
@group(0) @binding(4) var<storage, read_write> rad: array<RadOut>;

const NRAY: u32 = 64u;
const NFLOOR: u32 = 64u;
const PI: f32 = 3.14159265;
var<workgroup> sq: array<f32, 64>;
var<workgroup> sv: array<f32, 64>;

fn planck580(T: f32) -> f32 {
  let l = 580e-9;
  let x = 1.438777e-2 / (l * T);
  if (x > 80.0) { return 0.0; }
  return 1.191042e-16 / (l * l * l * l * l) / (exp(x) - 1.0) * 1e-9;
}

// ray vs rotated box: returns (t_enter, face) or t < 0
fn hitBox(o: vec2f, d: vec2f, pc: Piece) -> vec2f {
  let c = cos(pc.angle); let s = sin(pc.angle);
  let q = o - pc.pos;
  let lo = vec2f(c * q.x + s * q.y, -s * q.x + c * q.y);
  let ld = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
  let h = pc.half;
  let inv = 1.0 / select(ld, vec2f(1e-9), abs(ld) < vec2f(1e-9));
  let t1 = (vec2f(-h) - lo) * inv;
  let t2 = (vec2f(h) - lo) * inv;
  let tmin = min(t1, t2); let tmax = max(t1, t2);
  let tn = max(tmin.x, tmin.y); let tf = min(tmax.x, tmax.y);
  if (tf < 0.0 || tn > tf || tn < 1e-5) { return vec2f(-1.0, 0.0); }
  var face: f32;
  if (tmin.x > tmin.y) { face = select(2.0, 0.0, ld.x < 0.0); } else { face = select(3.0, 1.0, ld.y < 0.0); }
  return vec2f(tn, face);
}

fn gasAt(p: vec2f) -> vec2f {   // (k sigma T^4 / pi, visible emission coefficient * B580)
  let i = i32(p.x / P.dx); let j = i32(p.y / P.dx);
  if (i < 0 || j < 0 || i >= P.nx || j >= P.ny) { return vec2f(0.0); }
  let k = idxC(i, j);
  let Tg = T[k]; let Yg = Y[k];
  let fv = rhoOf(Tg) * Yg.w / P.sootDensity;
  let kap = P.kappaSootC * fv * Tg + P.kappaGas * Yg.z;
  let T2 = Tg * Tg;
  return vec2f(kap * P.sigma * (T2 * T2 - pow(P.Tamb, 4.0)) / PI, 6.3 * fv / 580e-9 * planck580(Tg));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let tgt = wg.x;
  let nFaces = u32(P.nPieces) * 4u;
  var origin = vec2f(0.0); var nrm = vec2f(0.0, 1.0); var valid = false;
  if (tgt < nFaces) {
    let pc = pieces[tgt / 4u];
    let f = tgt % 4u;
    if (pc.alive > 0.0) {
      let c = cos(pc.angle); let s = sin(pc.angle);
      var ln = vec2f(1.0, 0.0);
      if (f == 1u) { ln = vec2f(0.0, 1.0); } else if (f == 2u) { ln = vec2f(-1.0, 0.0); } else if (f == 3u) { ln = vec2f(0.0, -1.0); }
      nrm = vec2f(c * ln.x - s * ln.y, s * ln.x + c * ln.y);
      // sample along the face (stratified, jittered per frame)
      let tang = vec2f(-nrm.y, nrm.x);
      let u = (fract(f32(li) * 0.618034 + f32(P.frame) * 0.1234) - 0.5) * 1.6 * pc.half;
      origin = pc.pos + nrm * (pc.half + 1e-4) + tang * u;
      valid = true;
    }
  } else {
    // floor probes along the bottom of the domain
    let fi = tgt - nFaces;
    if (fi < NFLOOR) {
      origin = vec2f((f32(fi) + 0.5) / f32(NFLOOR) * f32(P.nx) * P.dx, 1e-4);
      nrm = vec2f(0.0, 1.0);
      valid = true;
    }
  }
  var dq = 0.0; var dv = 0.0;
  if (valid) {
    // direction within the hemisphere: stratified in angle with per-frame jitter
    let jit = fract(f32(P.frame) * 0.7548777 + f32(tgt) * 0.5698403);
    let a = ((f32(li) + jit) / f32(NRAY) - 0.5) * PI;
    let cosn = cos(a);
    let base = atan2(nrm.y, nrm.x);
    var d = vec2f(cos(base + a), sin(base + a));
    var o = origin;
    var J = 0.0; var Jv = 0.0;
    var Twall = P.Tamb;
    var bounced = false;
    let ds = P.dx;
    for (var step = 0; step < 4096; step++) {
      // nearest piece along the ray
      var tHit = 1e9; var hitT = 0.0;
      for (var n = 0; n < P.nPieces; n++) {
        let pc = pieces[n];
        if (pc.alive <= 0.0) { continue; }
        let h = hitBox(o, d, pc);
        if (h.x > 0.0 && h.x < tHit) { tHit = h.x; hitT = pc.Ts[u32(h.y)]; }
      }
      // floor
      var tFloor = 1e9;
      if (d.y < -1e-6) { tFloor = -o.y / d.y; }
      let tEnd = min(tHit, tFloor);
      // march the gas to the first obstacle or the domain edge
      let W = f32(P.nx) * P.dx; let H = f32(P.ny) * P.dx;
      var tx = 1e9; var ty = 1e9;
      if (d.x > 1e-6) { tx = (W - o.x) / d.x; } else if (d.x < -1e-6) { tx = -o.x / d.x; }
      if (d.y > 1e-6) { ty = (H - o.y) / d.y; }
      let tOut = min(min(tx, ty), tEnd);
      let ns = i32(ceil(tOut / ds));
      for (var m = 0; m < ns; m++) {
        let t = (f32(m) + 0.5) * tOut / f32(ns);
        let g = gasAt(o + d * t);
        J += g.x * tOut / f32(ns);
        Jv += g.y * tOut / f32(ns);
      }
      if (tHit < tFloor && tHit < 1e8) { Twall = hitT; break; }
      if (tFloor < 1e8 && tFloor <= tHit && !bounced) {
        o = o + d * tFloor + vec2f(0.0, 1e-5);
        d = vec2f(d.x, -d.y);
        bounced = true;
        continue;
      }
      Twall = P.Tamb;
      break;
    }
    let w = PI / f32(NRAY) * cosn;
    let T2 = Twall * Twall;
    dq = w * (0.5 * P.sigma * T2 * T2 + 2.0 * J);
    dv = w * (2.0 * Jv);
  }
  sq[li] = dq; sv[li] = dv;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s = s >> 1u) {
    if (li < s) { sq[li] += sq[li + s]; sv[li] += sv[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u && valid) {
    // temporal smoothing of the stochastic estimate
    let old = rad[tgt];
    let a = select(0.35, 1.0, old.q <= 0.0);
    rad[tgt] = RadOut(mix(old.q, sq[0], a), mix(old.vis, sv[0], a), 0.0, 0.0);
  }
}
