// Momentum: viscous diffusion (Sutherland mu(T)), non-Boussinesq buoyancy
// a_y = g (T/Tamb - 1) = -g (rho - rho_amb)/rho, momentum dilution by entrained air.
// Floor is no-slip; wood faces are no-slip (zero velocity). Domain sides and top
// are open to still ambient air: outflow leaves with its momentum (zero
// gradient); inflow arrives from rest (no momentum), drawn in by the pressure.
//#include common

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> U2: array<f32>;
@group(0) @binding(2) var<storage, read> V2: array<f32>;
@group(0) @binding(3) var<storage, read> T: array<f32>;
@group(0) @binding(4) var<storage, read> fx: array<f32>;
@group(0) @binding(5) var<storage, read> fy: array<f32>;
@group(0) @binding(6) var<storage, read_write> U: array<f32>;
@group(0) @binding(7) var<storage, read_write> V: array<f32>;

fn Tc(i: i32, j: i32) -> f32 {
  return T[idxC(clamp(i, 0, P.nx - 1), clamp(j, 0, P.ny - 1))];
}
fn u2(i: i32, j: i32) -> f32 {
  let k = idxU(i, j);
  return select(0.0, U2[k], fx[k] > 0.0);
}
fn v2(i: i32, j: i32) -> f32 {
  let k = idxV(i, j);
  return select(0.0, V2[k], fy[k] > 0.0);
}
// entrained ambient air arrives with no momentum: rho du/dt = -m_e u,
// m_e/rho = 2 alpha |u| (rho_amb/rho) / depth = 2 alpha |u| (T/Tamb) / depth
fn exchange(spd: f32, Tf: f32) -> f32 {
  return exp(-2.0 * P.exchAlpha * spd * (Tf / P.Tamb) / P.depth * P.dt);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  let idx2 = 1.0 / (P.dx * P.dx);
  // ---- U faces ----
  if (i <= P.nx && j < P.ny) {
    let k = idxU(i, j);
    if (fx[k] <= 0.0) {
      U[k] = 0.0;
    } else if (i == 0) {
      let ui = U2[idxU(1, j)];
      U[k] = min(ui, 0.0);
    } else if (i == P.nx) {
      let ui = U2[idxU(P.nx - 1, j)];
      U[k] = max(ui, 0.0);
    } else {
      let u = U2[k];
      let TL = Tc(i - 1, j); let TR = Tc(i, j);
      let rhoF = rhoOf(0.5 * (TL + TR));
      let muE = muOf(TR); let muW = muOf(TL);
      var uN = u; var muN = muOf(0.5 * (TL + TR));
      if (j < P.ny - 1) {
        uN = u2(i, j + 1);
        muN = muOf(0.25 * (TL + TR + Tc(i - 1, j + 1) + Tc(i, j + 1)));
      }
      var uS = -u; var muS = muOf(0.5 * (TL + TR));
      if (j > 0) {
        uS = u2(i, j - 1);
        muS = muOf(0.25 * (TL + TR + Tc(i - 1, j - 1) + Tc(i, j - 1)));
      }
      let visc = (muE * (u2(i + 1, j) - u) + muW * (u2(i - 1, j) - u)
                + muN * (uN - u) + muS * (uS - u)) * idx2 / rhoF;
      let vAvg = 0.25 * (V2[idxV(i - 1, j)] + V2[idxV(i, j)] + V2[idxV(i - 1, j + 1)] + V2[idxV(i, j + 1)]);
      U[k] = (u + P.dt * visc) * exchange(length(vec2f(u, vAvg)), 0.5 * (TL + TR));
    }
  }
  // ---- V faces ----
  if (i < P.nx && j <= P.ny) {
    let k = idxV(i, j);
    if (j == 0 || fy[k] <= 0.0) {
      V[k] = 0.0;
    } else if (j == P.ny) {
      let vi = V2[idxV(i, P.ny - 1)];
      V[k] = max(vi, 0.0);
    } else {
      let v = V2[k];
      let TB = Tc(i, j - 1); let TT = Tc(i, j);
      let Tf = 0.5 * (TB + TT);
      let rhoF = rhoOf(Tf);
      let muN = muOf(TT); let muS = muOf(TB);
      var vE = v; var muE = muOf(Tf);
      if (i < P.nx - 1) {
        vE = v2(i + 1, j);
        muE = muOf(0.25 * (TB + TT + Tc(i + 1, j - 1) + Tc(i + 1, j)));
      }
      var vW = v; var muW = muOf(Tf);
      if (i > 0) {
        vW = v2(i - 1, j);
        muW = muOf(0.25 * (TB + TT + Tc(i - 1, j - 1) + Tc(i - 1, j)));
      }
      let visc = (muN * (v2(i, j + 1) - v) + muS * (v2(i, j - 1) - v)
                + muE * (vE - v) + muW * (vW - v)) * idx2 / rhoF;
      let buoy = P.grav * (Tf / P.Tamb - 1.0);
      let uAvg = 0.25 * (U2[idxU(i, j - 1)] + U2[idxU(i + 1, j - 1)] + U2[idxU(i, j)] + U2[idxU(i + 1, j)]);
      V[k] = (v + P.dt * (visc + buoy)) * exchange(length(vec2f(uAvg, v)), Tf);
    }
  }
}
