// Variable-density pressure projection:  div( f (1/rho) grad p ) = (div(f u*) - S) / dt
// solved by a cell-centred multigrid V-cycle (red-black Gauss-Seidel).
// Face coefficient a = f/rho (f: open fraction). Open sides/top: p = 0 (Dirichlet
// at the face); floor and wood: zero normal velocity (a = 0).
//#include common

struct Level { nx: i32, ny: i32, h2: f32, color: i32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> L: Level;
@group(0) @binding(2) var<storage, read_write> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read> T: array<f32>;
@group(0) @binding(5) var<storage, read> fx: array<f32>;
@group(0) @binding(6) var<storage, read> fy: array<f32>;
@group(0) @binding(7) var<storage, read> SQ: array<vec2f>;
@group(0) @binding(8) var<storage, read_write> p: array<f32>;
@group(0) @binding(9) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(10) var<storage, read_write> ax: array<f32>;
@group(0) @binding(11) var<storage, read_write> ay: array<f32>;
// coarse level for transfers
@group(0) @binding(12) var<storage, read_write> pc: array<f32>;
@group(0) @binding(13) var<storage, read_write> rhsc: array<f32>;
@group(0) @binding(14) var<storage, read_write> axc: array<f32>;
@group(0) @binding(15) var<storage, read_write> ayc: array<f32>;

fn invRho(T: f32) -> f32 { return T / (P.rhoAmb * P.Tamb); }

// ---- level-0 face coefficients from temperature and geometry ----
@compute @workgroup_size(8, 8)
fn coef(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i <= P.nx && j < P.ny) {
    let k = idxU(i, j);
    let Tf = 0.5 * (T[idxC(clamp(i - 1, 0, P.nx - 1), j)] + T[idxC(min(i, P.nx - 1), j)]);
    ax[k] = fx[k] * invRho(Tf);
  }
  if (i < P.nx && j <= P.ny) {
    let k = idxV(i, j);
    let Tf = 0.5 * (T[idxC(i, clamp(j - 1, 0, P.ny - 1))] + T[idxC(i, min(j, P.ny - 1))]);
    ay[k] = fy[k] * invRho(Tf);
  }
}

// ---- level-0 right-hand side ----
@compute @workgroup_size(8, 8)
fn rhs0(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= P.nx || j >= P.ny) { return; }
  let div = (fx[idxU(i + 1, j)] * U[idxU(i + 1, j)] - fx[idxU(i, j)] * U[idxU(i, j)]
           + fy[idxV(i, j + 1)] * V[idxV(i, j + 1)] - fy[idxV(i, j)] * V[idxV(i, j)]) / P.dx;
  let open = fx[idxU(i + 1, j)] + fx[idxU(i, j)] + fy[idxV(i, j + 1)] + fy[idxV(i, j)];
  rhs[idxC(i, j)] = select(0.0, (div - SQ[idxC(i, j)].x) / P.dt, open > 0.0);
}

// ---- coefficient restriction to the next coarser level (L = fine dims) ----
@compute @workgroup_size(8, 8)
fn coefRestrict(@builtin(global_invocation_id) id: vec3u) {
  let I = i32(id.x); let J = i32(id.y);
  let nxc = L.nx / 2; let nyc = L.ny / 2;
  if (I <= nxc && J < nyc) {
    axc[J * (nxc + 1) + I] = 0.5 * (ax[(2 * J) * (L.nx + 1) + 2 * I] + ax[(2 * J + 1) * (L.nx + 1) + 2 * I]);
  }
  if (I < nxc && J <= nyc) {
    ayc[J * nxc + I] = 0.5 * (ay[(2 * J) * L.nx + 2 * I] + ay[(2 * J) * L.nx + 2 * I + 1]);
  }
}

// stencil helpers on level L
fn aW(i: i32, j: i32) -> f32 { return ax[j * (L.nx + 1) + i]; }
fn aE(i: i32, j: i32) -> f32 { return ax[j * (L.nx + 1) + i + 1]; }
fn aS(i: i32, j: i32) -> f32 { return ay[j * L.nx + i]; }
fn aN(i: i32, j: i32) -> f32 { return ay[(j + 1) * L.nx + i]; }
fn pAt(i: i32, j: i32) -> f32 { return p[j * L.nx + i]; }

// sum a*p_nb and diagonal (Dirichlet p=0 at open domain faces -> ghost = -p)
fn stencil(i: i32, j: i32) -> vec2f {
  var off = 0.0; var diag = 0.0;
  let w = aW(i, j); let e = aE(i, j); let s = aS(i, j); let n = aN(i, j);
  if (i > 0) { off += w * pAt(i - 1, j); diag += w; } else { diag += 2.0 * w; }
  if (i < L.nx - 1) { off += e * pAt(i + 1, j); diag += e; } else { diag += 2.0 * e; }
  if (j > 0) { off += s * pAt(i, j - 1); diag += s; }
  if (j < L.ny - 1) { off += n * pAt(i, j + 1); diag += n; } else { diag += 2.0 * n; }
  return vec2f(off, diag);
}

@compute @workgroup_size(8, 8)
fn relax(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= L.nx || j >= L.ny || ((i + j) & 1) != L.color) { return; }
  let st = stencil(i, j);
  let k = j * L.nx + i;
  p[k] = select(0.0, (st.x - rhs[k] * L.h2) / st.y, st.y > 0.0);
}

fn residual(i: i32, j: i32) -> f32 {
  let st = stencil(i, j);
  let k = j * L.nx + i;
  if (st.y <= 0.0) { return 0.0; }
  return rhs[k] - (st.x - st.y * p[k]) / L.h2;
}

// residual on level L restricted (cell average) to the coarse rhs; coarse p = 0
@compute @workgroup_size(8, 8)
fn restrictRes(@builtin(global_invocation_id) id: vec3u) {
  let I = i32(id.x); let J = i32(id.y);
  let nxc = L.nx / 2; let nyc = L.ny / 2;
  if (I >= nxc || J >= nyc) { return; }
  let r = residual(2 * I, 2 * J) + residual(2 * I + 1, 2 * J)
        + residual(2 * I, 2 * J + 1) + residual(2 * I + 1, 2 * J + 1);
  rhsc[J * nxc + I] = 0.25 * r;
  pc[J * nxc + I] = 0.0;
}

// coarse correction added to level L (bilinear from coarse cell centres)
@compute @workgroup_size(8, 8)
fn prolong(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i >= L.nx || j >= L.ny) { return; }
  let nxc = L.nx / 2; let nyc = L.ny / 2;
  let I = i / 2; let J = j / 2;
  let I2 = clamp(I + select(-1, 1, (i & 1) == 1), 0, nxc - 1);
  let J2 = clamp(J + select(-1, 1, (j & 1) == 1), 0, nyc - 1);
  let c = 0.5625 * pc[J * nxc + I] + 0.1875 * (pc[J * nxc + I2] + pc[J2 * nxc + I]) + 0.0625 * pc[J2 * nxc + I2];
  let k = j * L.nx + i;
  // no correction inside closed (solid) cells
  let st = stencil(i, j);
  p[k] += select(0.0, c, st.y > 0.0);
}

// ---- coarsest level: many Gauss-Seidel sweeps in one workgroup ----
const MAXC: i32 = 256;
var<workgroup> wp: array<f32, 256>;
@compute @workgroup_size(256)
fn coarsest(@builtin(local_invocation_index) li: u32) {
  let n = L.nx * L.ny;
  let k = i32(li);
  if (k < n) { wp[k] = 0.0; }
  workgroupBarrier();
  let i = k % L.nx; let j = k / L.nx;
  for (var it = 0; it < 120; it++) {
    for (var c = 0; c < 2; c++) {
      if (k < n && ((i + j) & 1) == c) {
        var off = 0.0; var diag = 0.0;
        let w = aW(i, j); let e = aE(i, j); let s = aS(i, j); let nn = aN(i, j);
        if (i > 0) { off += w * wp[k - 1]; diag += w; } else { diag += 2.0 * w; }
        if (i < L.nx - 1) { off += e * wp[k + 1]; diag += e; } else { diag += 2.0 * e; }
        if (j > 0) { off += s * wp[k - L.nx]; diag += s; }
        if (j < L.ny - 1) { off += nn * wp[k + L.nx]; diag += nn; } else { diag += 2.0 * nn; }
        wp[k] = select(0.0, (off - rhs[k] * L.h2) / diag, diag > 0.0);
      }
      workgroupBarrier();
    }
  }
  if (k < n) { p[k] = wp[k]; }
}

// ---- velocity update u = u* - dt (1/rho) grad p ----
@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  if (i <= P.nx && j < P.ny) {
    let k = idxU(i, j);
    if (fx[k] > 0.0) {
      let pl = select(-p[idxC(0, j)], p[idxC(max(i - 1, 0), j)], i > 0);
      let pr = select(-p[idxC(P.nx - 1, j)], p[idxC(min(i, P.nx - 1), j)], i < P.nx);
      let Tf = 0.5 * (T[idxC(clamp(i - 1, 0, P.nx - 1), j)] + T[idxC(min(i, P.nx - 1), j)]);
      U[k] -= P.dt * invRho(Tf) * (pr - pl) / P.dx;
    } else { U[k] = 0.0; }
  }
  if (i < P.nx && j <= P.ny) {
    let k = idxV(i, j);
    if (j > 0 && fy[k] > 0.0) {
      let pb = p[idxC(i, j - 1)];
      let pt = select(-p[idxC(i, P.ny - 1)], p[idxC(i, min(j, P.ny - 1))], j < P.ny);
      let Tf = 0.5 * (T[idxC(i, j - 1)] + T[idxC(i, min(j, P.ny - 1))]);
      V[k] -= P.dt * invRho(Tf) * (pt - pb) / P.dx;
    } else { V[k] = 0.0; }
  }
}
