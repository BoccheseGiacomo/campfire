// Explicit molecular diffusion of heat and species (conservative face fluxes,
// temperature-dependent rho*D = k/cp, unity Lewis number). Soot does not diffuse.
// Wood surfaces and domain edges are zero-flux here: wall heat exchange is
// handled in the sources pass.
//#include common

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> Tin: array<f32>;
@group(0) @binding(2) var<storage, read> Yin: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> Tout: array<f32>;
@group(0) @binding(4) var<storage, read_write> Yout: array<vec4f>;
@group(0) @binding(5) var<storage, read> fx: array<f32>;
@group(0) @binding(6) var<storage, read> fy: array<f32>;
@group(0) @binding(7) var<uniform> D: DiffParams;

struct DiffParams { dt: f32, p0: f32, p1: f32, p2: f32 };

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= P.nx || j >= P.ny) { return; }
  let k = idxC(i, j);
  let T0 = Tin[k]; let Y0 = Yin[k];
  var sT = 0.0; var sY = vec4f(0.0);
  let mask = vec4f(1.0, 1.0, 1.0, 0.0);
  // west, east, south, north
  if (i > 0) {
    let f = fx[idxU(i, j)];
    if (f > 0.0) { let n = idxC(i - 1, j); let a = f * rhoDOf(0.5 * (T0 + Tin[n]));
      sT += a * (Tin[n] - T0); sY += a * (Yin[n] - Y0); }
  }
  if (i < P.nx - 1) {
    let f = fx[idxU(i + 1, j)];
    if (f > 0.0) { let n = idxC(i + 1, j); let a = f * rhoDOf(0.5 * (T0 + Tin[n]));
      sT += a * (Tin[n] - T0); sY += a * (Yin[n] - Y0); }
  }
  if (j > 0) {
    let f = fy[idxV(i, j)];
    if (f > 0.0) { let n = idxC(i, j - 1); let a = f * rhoDOf(0.5 * (T0 + Tin[n]));
      sT += a * (Tin[n] - T0); sY += a * (Yin[n] - Y0); }
  }
  if (j < P.ny - 1) {
    let f = fy[idxV(i, j + 1)];
    if (f > 0.0) { let n = idxC(i, j + 1); let a = f * rhoDOf(0.5 * (T0 + Tin[n]));
      sT += a * (Tin[n] - T0); sY += a * (Yin[n] - Y0); }
  }
  let c = D.dt / (rhoOf(T0) * P.dx * P.dx);
  Tout[k] = T0 + c * sT;
  Yout[k] = max(Y0 + c * sY * mask, vec4f(0.0));
}
