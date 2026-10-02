// Semi-Lagrangian advection (RK2 back-trace) with a limited MacCormack
// correction (second order, clamped to the departure stencil).
//#include common

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Uh: array<f32>;
@group(0) @binding(4) var<storage, read_write> Vh: array<f32>;
@group(0) @binding(5) var<storage, read_write> U2: array<f32>;
@group(0) @binding(6) var<storage, read_write> V2: array<f32>;
@group(0) @binding(7) var<storage, read> T: array<f32>;
@group(0) @binding(8) var<storage, read> Y: array<vec4f>;
@group(0) @binding(9) var<storage, read_write> Th: array<f32>;
@group(0) @binding(10) var<storage, read_write> Yh: array<vec4f>;
@group(0) @binding(11) var<storage, read_write> T2: array<f32>;
@group(0) @binding(12) var<storage, read_write> Y2: array<vec4f>;

//#sampler U U
//#sampler V V
//#sampler U Uh
//#sampler V Vh
//#sampler C T P.Tamb
//#sampler C Th P.Tamb
//#sampler C4 Y ambY()
//#sampler C4 Yh ambY()

fn vel(g: vec2f) -> vec2f { return vec2f(s_U(g), s_V(g)) / P.dx; }

// departure point of grid position g over time dt (dt < 0 traces forward)
fn trace(g: vec2f, dt: f32) -> vec2f {
  let gm = g - 0.5 * dt * vel(g);
  return g - dt * vel(gm);
}

@compute @workgroup_size(8, 8)
fn velForward(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i <= P.nx && j < P.ny) {
    let g = vec2f(f32(i), f32(j) + 0.5);
    Uh[idxU(i, j)] = s_U(trace(g, P.dt));
  }
  if (i < P.nx && j <= P.ny) {
    let g = vec2f(f32(i) + 0.5, f32(j));
    Vh[idxV(i, j)] = s_V(trace(g, P.dt));
  }
}

@compute @workgroup_size(8, 8)
fn velCorrect(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i <= P.nx && j < P.ny) {
    let k = idxU(i, j);
    let g = vec2f(f32(i), f32(j) + 0.5);
    let back = s_Uh(trace(g, -P.dt));
    let mm = mm_U(trace(g, P.dt));
    U2[k] = clamp(Uh[k] + 0.5 * (U[k] - back), mm.x, mm.y);
  }
  if (i < P.nx && j <= P.ny) {
    let k = idxV(i, j);
    let g = vec2f(f32(i) + 0.5, f32(j));
    let back = s_Vh(trace(g, -P.dt));
    let mm = mm_V(trace(g, P.dt));
    V2[k] = clamp(Vh[k] + 0.5 * (V[k] - back), mm.x, mm.y);
  }
}

@compute @workgroup_size(8, 8)
fn scalForward(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= P.nx || j >= P.ny) { return; }
  let k = idxC(i, j);
  let d = trace(vec2f(f32(i) + 0.5, f32(j) + 0.5), P.dt);
  Th[k] = s_T(d);
  Yh[k] = s_Y(d);
}

@compute @workgroup_size(8, 8)
fn scalCorrect(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= P.nx || j >= P.ny) { return; }
  let k = idxC(i, j);
  let g = vec2f(f32(i) + 0.5, f32(j) + 0.5);
  let d = trace(g, P.dt);
  let f = trace(g, -P.dt);
  let mT = mm_T(d);
  let mY = mm_Y(d);
  let Tc = Th[k] + 0.5 * (T[k] - s_Th(f));
  let Yc = Yh[k] + 0.5 * (Y[k] - s_Yh(f));
  // The first-order value is a convex mix of neighbouring states (physically
  // realisable: it can never burn above the adiabatic temperature). Keep the
  // second-order correction only if it stays within the stencil bounds and does
  // not create co-existing fuel and oxidiser beyond the first-order mixture.
  let inT = Tc >= mT.x && Tc <= mT.y;
  let inY = all(Yc >= mY.lo) && all(Yc <= mY.hi);
  let mixC = min(Yc.x, Yc.y / P.sO2);
  let mixL = min(Yh[k].x, Yh[k].y / P.sO2);
  if (inT && inY && mixC <= mixL + 1e-5) {
    T2[k] = Tc; Y2[k] = Yc;
  } else {
    T2[k] = Th[k]; Y2[k] = Yh[k];
  }
}
