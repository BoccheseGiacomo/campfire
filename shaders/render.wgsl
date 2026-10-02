// Rendering. The viewer looks along the unseen axis.
// Gas: the slab of thickness `depth` is uniform along the line of sight, so the
//   spectral radiance is exactly L = (1 - exp(-k_l depth)) B_l(T) for soot
//   (Rayleigh absorption k_l = 6.3 fv / lambda), plus thin chemiluminescence
//   proportional to the local heat release (CH* 431 nm, C2* 516 nm).
// Wood: we see the log ends, i.e. the cross-section: a char ring of the
//   simulated depth with the simulated surface -> front -> core temperature
//   profile, emitting as a grey body, plus firelight from the computed
//   visible-band irradiance.
// Spectra are integrated with CIE 1931 matching functions to XYZ -> sRGB.
//#include common

struct RP {
  view: i32, nPieces: i32, exposure: f32, width: f32,
  height: f32, ground: f32, time: f32, tmax: f32,
  chemi: f32, camX: f32, camY: f32, zoom: f32,   // camera: world point at the view's bottom-left, zoom
};
struct WoodR { Tp: vec4f, d: vec4f, misc: vec4f };   // misc: charFrac, Tcore, visAvg, hasVirgin

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> R: RP;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read> Y: array<vec4f>;
@group(0) @binding(4) var<storage, read> SQ: array<vec2f>;
@group(0) @binding(5) var<storage, read> pieces: array<Piece>;
@group(0) @binding(6) var<storage, read> U: array<f32>;
@group(0) @binding(7) var<storage, read> V: array<f32>;
@group(0) @binding(8) var<storage, read> wood: array<WoodR>;
@group(0) @binding(9) var<storage, read> rad: array<vec4f>;

//#sampler C T P.Tamb
//#sampler C4 Y ambY()

const NFLOOR: i32 = 64;

fn fQ(i: i32, j: i32) -> f32 {
  if (i < 0 || i >= P.nx || j < 0 || j >= P.ny) { return 0.0; }
  return SQ[idxC(i, j)].y;
}
fn sQ(g: vec2f) -> f32 {
  let f = g - vec2f(0.5); let b = floor(f); let t = f - b; let i = i32(b.x); let j = i32(b.y);
  return mix(mix(fQ(i, j), fQ(i + 1, j), t.x), mix(fQ(i, j + 1), fQ(i + 1, j + 1), t.x), t.y);
}

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(0.5 * (p[vi].x + 1.0), 0.5 * (p[vi].y + 1.0));
  return o;
}

// ---------------- colour science ----------------
fn g3(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
  let t = (x - mu) / select(s2, s1, x < mu);
  return exp(-0.5 * t * t);
}
// CIE 1931 2-degree matching functions (Wyman, Sloan & Shirley 2013 fit)
fn cie(l: f32) -> vec3f {
  let x = 1.056 * g3(l, 599.8, 37.9, 31.0) + 0.362 * g3(l, 442.0, 16.0, 26.7) - 0.065 * g3(l, 501.1, 20.4, 26.2);
  let y = 0.821 * g3(l, 568.8, 46.9, 40.5) + 0.286 * g3(l, 530.9, 16.3, 31.1);
  let z = 1.217 * g3(l, 437.0, 11.8, 36.0) + 0.681 * g3(l, 459.0, 26.0, 13.8);
  return vec3f(x, y, z);
}
fn xyz2rgb(c: vec3f) -> vec3f {
  return vec3f(
     3.2406 * c.x - 1.5372 * c.y - 0.4986 * c.z,
    -0.9689 * c.x + 1.8758 * c.y + 0.0415 * c.z,
     0.0557 * c.x - 0.2040 * c.y + 1.0570 * c.z);
}
// Planck spectral radiance, W/(m^2 sr nm)
fn planck(lnm: f32, T: f32) -> f32 {
  let l = lnm * 1e-9;
  let x = 1.438777e-2 / (l * T);
  if (x > 80.0) { return 0.0; }
  return 1.191042e-16 / (l * l * l * l * l) / (exp(x) - 1.0) * 1e-9;
}
fn line(l: f32, l0: f32, w: f32) -> f32 { let t = (l - l0) / w; return exp(-0.5 * t * t) / (2.5066 * w); }

const NL: i32 = 12;
fn lam(n: i32) -> f32 { return 400.0 + (f32(n) + 0.5) * 300.0 / f32(NL); }

fn gasXYZ(Tg: f32, fv: f32, q: f32) -> vec3f {
  var xyz = vec3f(0.0);
  for (var n = 0; n < NL; n++) {
    let l = lam(n);
    let k = 6.3 * fv / (l * 1e-9);
    var L = (1.0 - exp(-k * P.depth)) * planck(l, Tg);
    L += R.chemi * q * P.depth / 12.566 * (0.6 * line(l, 431.0, 6.0) + 0.4 * line(l, 516.0, 8.0));
    xyz += cie(l) * L;
  }
  return xyz * (300.0 / f32(NL));
}
fn greyXYZ(Ts: f32, eps: f32) -> vec3f {
  if (Ts < 650.0) { return vec3f(0.0); }
  var xyz = vec3f(0.0);
  for (var n = 0; n < NL; n++) { let l = lam(n); xyz += cie(l) * eps * planck(l, Ts); }
  return xyz * (300.0 / f32(NL));
}
// XYZ of light whose 580 nm spectral value is 1, with a ~1400 K soot spectrum
fn fireLightXYZ() -> vec3f {
  var xyz = vec3f(0.0);
  let b0 = planck(580.0, 1400.0) / 580.0;
  for (var n = 0; n < NL; n++) { let l = lam(n); xyz += cie(l) * planck(l, 1400.0) / l / b0; }
  return xyz * (300.0 / f32(NL));
}
fn srgbEncode(c: vec3f) -> vec3f {
  let s = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(s, vec3f(1.0 / 2.4)) - 0.055, 12.92 * s, s <= vec3f(0.0031308));
}
// Luminance-logarithmic tone mapping (the eye/film compresses the ~10^4 range
// between embers and flame); chromaticity is kept, and channels that would
// exceed the display range bleed toward white at constant luminance.
fn tonemapXYZ(xyz: vec3f, base: vec3f) -> vec3f {
  let L = max(xyz.y * R.exposure, 0.0);
  let L0 = 0.001; let Lw = 20.0;
  let Ld = log(1.0 + L / L0) / log(1.0 + Lw / L0);   // perceptual display luminance
  var rgb = vec3f(0.0);
  if (xyz.y > 1e-12) {
    let c = max(xyz2rgb(xyz / xyz.y), vec3f(0.0));    // linear rgb with unit luminance
    let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    // per-channel film shoulder: overexposed red saturates first, so bright
    // flame shifts red -> orange -> yellow -> white as on film or to the eye
    let pre = c / max(lum, 1e-6) * pow(Ld, 2.2) * 1.6;
    rgb = vec3f(1.0) - exp(-pre);
  }
  return srgbEncode(rgb + base);
}
fn inferno(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3f(0.0002, 0.0016, -0.0194); let c1 = vec3f(0.1065, 0.5640, 3.9327);
  let c2 = vec3f(11.6024, -3.9728, -15.9423); let c3 = vec3f(-41.7040, 17.4364, 44.3541);
  let c4 = vec3f(77.1629, -33.4023, -81.8073); let c5 = vec3f(-71.3194, 32.6261, 73.2095);
  let c6 = vec3f(25.1311, -12.2426, -23.0703);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1.0, 0.0)), u.x), mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x), u.y);
}

// ---------------- wood cross-section ----------------
struct WoodHit { idx: i32, l: vec2f };
fn woodAt(p: vec2f) -> WoodHit {
  for (var n = R.nPieces - 1; n >= 0; n--) {
    let pc = pieces[n];
    if (pc.alive <= 0.0) { continue; }
    let c = cos(pc.angle); let s = sin(pc.angle);
    let q = p - pc.pos;
    let l = vec2f(c * q.x + s * q.y, -s * q.x + c * q.y);
    if (abs(l.x) <= pc.half && abs(l.y) <= pc.half) { return WoodHit(n, l); }
  }
  return WoodHit(-1, vec2f(0.0));
}
// depth below each face (local frame): +x, +y, -x, -y
fn faceDepths(l: vec2f, h: f32) -> vec4f { return vec4f(h - l.x, h - l.y, h + l.x, h + l.y); }

struct WoodPix { T: f32, char: f32 };
fn woodProfile(pc: Piece, w: WoodR, l: vec2f) -> WoodPix {
  let h = pc.half;
  let dep = faceDepths(l, h);
  // soft weights toward the nearest faces
  let wgt = exp(-(dep - vec4f(min(min(dep.x, dep.y), min(dep.z, dep.w)))) / (0.15 * h));
  var T = 0.0; var ch = 0.0; var ws = 0.0;
  for (var f = 0; f < 4; f++) {
    let dd = dep[f];
    let cd = max(w.d[f], 1e-5);
    var Tf: f32;
    if (dd < cd) { Tf = mix(pc.Ts[f], w.Tp[f], dd / cd); }
    else { Tf = mix(w.Tp[f], w.misc.y, smoothstep(cd, h, dd)); }
    T += wgt[f] * Tf;
    ch += wgt[f] * select(0.0, 1.0, dd < cd || w.misc.w < 0.5);
    ws += wgt[f];
  }
  return WoodPix(T / ws, ch / ws);
}

fn floorVis(x: f32) -> f32 {
  let base = R.nPieces * 4;
  let fx = clamp(x / R.width * f32(NFLOOR) - 0.5, 0.0, f32(NFLOOR - 1));
  let i0 = i32(floor(fx)); let i1 = min(i0 + 1, NFLOOR - 1);
  return mix(rad[base + i0].y, rad[base + i1].y, fx - floor(fx));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let wpos = vec2f(in.uv.x * R.width, in.uv.y * (R.height + R.ground)) / R.zoom + vec2f(R.camX, R.camY);
  let g = wpos / P.dx;
  let hit = woodAt(wpos);
  let ground = wpos.y < 0.0;

  if (R.view == 0) {
    // ---------------- photoreal ----------------
    var xyz = vec3f(0.0);
    var rgbAdd = vec3f(0.0);
    let fl = fireLightXYZ();
    if (hit.idx >= 0) {
      let pc = pieces[hit.idx]; let w = wood[hit.idx];
      let wp = woodProfile(pc, w, hit.l);
      let r = length(hit.l) / pc.half;
      // end grain: growth rings and rays; char: dark, cracked
      let rings = 0.5 + 0.5 * sin(r * 38.0 + vnoise(hit.l * 300.0) * 2.0 + f32(hit.idx) * 3.0);
      let grain = vec3f(0.42, 0.28, 0.15) * (0.8 + 0.25 * rings) * (0.85 + 0.3 * vnoise(hit.l * 900.0));
      let crackN = vnoise(hit.l * 420.0 + vec2f(f32(hit.idx) * 13.0));
      let crack = smoothstep(0.72, 0.8, crackN);
      let charCol = vec3f(0.035, 0.032, 0.03) * (0.7 + 0.6 * vnoise(hit.l * 1500.0));
      let albedo = mix(grain, charCol, wp.char);
      // glow: cracks expose hotter material slightly deeper in the char
      let Tglow = wp.T * (1.0 + 0.05 * crack * wp.char);
      xyz += greyXYZ(Tglow, 0.9);
      // firelight: a fraction of the faces' visible irradiance reaches the end face
      xyz += fl * dot(albedo, vec3f(0.2126, 0.7152, 0.0722)) * w.misc.z * 0.035 / 3.14159;
      rgbAdd = albedo * 0.006;
    } else if (ground) {
      let n = vnoise(wpos * 180.0) * 0.6 + vnoise(wpos * 900.0) * 0.4;
      let alb = vec3f(0.16, 0.13, 0.10) * (0.6 + 0.6 * n);
      let fade = exp(wpos.y / R.ground * 2.5);
      xyz += fl * floorVis(wpos.x) / 3.14159 * dot(alb, vec3f(0.333)) * fade * 0.06;
      rgbAdd = alb * 0.01;
    } else {
      let Tg = s_T(g);
      let Yg = s_Y(g);
      let rho = rhoOf(Tg);
      let fv = rho * Yg.w / P.sootDensity;
      xyz += gasXYZ(Tg, fv, sQ(g));
      // night background
      rgbAdd = mix(vec3f(0.006, 0.006, 0.010), vec3f(0.0015, 0.0015, 0.003), in.uv.y);
    }
    return vec4f(tonemapXYZ(xyz, rgbAdd), 1.0);
  }

  var col = vec3f(0.0);
  if (R.view == 1) {
    // ---------------- temperature ----------------
    var Tv = P.Tamb;
    if (hit.idx >= 0) { Tv = woodProfile(pieces[hit.idx], wood[hit.idx], hit.l).T; }
    else if (!ground) { Tv = s_T(g); }
    col = inferno((Tv - P.Tamb) / (R.tmax - P.Tamb));
    if (ground) { col = vec3f(0.08); }
  } else if (!ground && hit.idx < 0) {
    let Yg = s_Y(g);
    if (R.view == 2) {
      col = inferno(pow(sQ(g) / 2.0e7, 0.5));
    } else if (R.view == 3) {
      col = mix(vec3f(0.02, 0.02, 0.08), vec3f(0.35, 0.85, 1.0), Yg.y / max(P.YOamb, 1e-3));
    } else if (R.view == 4) {
      col = inferno(sqrt(Yg.x / 0.4));
    } else if (R.view == 5) {
      let gi = vec2i(g);
      let ii = clamp(gi.x, 0, P.nx - 1); let jj = clamp(gi.y, 0, P.ny - 1);
      let u = 0.5 * (U[idxU(ii, jj)] + U[idxU(ii + 1, jj)]);
      let v = 0.5 * (V[idxV(ii, jj)] + V[idxV(ii, jj + 1)]);
      col = inferno(length(vec2f(u, v)) / 3.0);
    } else if (R.view == 6) {
      col = inferno(sqrt(rhoOf(s_T(g)) * Yg.w / P.sootDensity / 1e-6));
    }
  } else if (ground) {
    col = vec3f(0.08);
  } else {
    col = vec3f(0.35, 0.3, 0.25);
  }
  // piece outlines
  if (hit.idx >= 0) {
    let h = pieces[hit.idx].half;
    let e = h - max(abs(hit.l.x), abs(hit.l.y));
    col = mix(col, vec3f(0.85), (1.0 - smoothstep(0.0, 0.0012, e)) * 0.6);
  }
  return vec4f(col, 1.0);
}
