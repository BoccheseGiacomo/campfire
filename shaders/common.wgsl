// Shared parameters and helpers. Grid coordinates are in cell units:
// cell (i,j) centre at (i+0.5, j+0.5); U(i,j) at (i, j+0.5); V(i,j) at (i+0.5, j).

struct Params {
  nx: i32, ny: i32, dx: f32, dt: f32,
  Tamb: f32, rhoAmb: f32, YOamb: f32, grav: f32,
  cp: f32, Pr: f32, viscMult: f32, diffMult: f32,
  reactB: f32, reactTa: f32, dHc: f32, sO2: f32,
  sootFormA: f32, sootFormTa: f32, sootOxA: f32, sootOxTa: f32,
  kappaSootC: f32, kappaGas: f32, exchAlpha: f32, depth: f32,
  charA: f32, charTa: f32, nPieces: i32, frame: u32,
  heaterX0: f32, heaterX1: f32, heaterY1: f32, heaterT: f32,
  burnerX0: f32, burnerX1: f32, burnerFlux: f32, burnerT: f32,
  sootDensity: f32, sigma: f32, charO2: f32, time: f32,
};

struct Piece {
  pos: vec2f, angle: f32, half: f32,
  vel: vec2f, omega: f32, alive: f32,
  Ts: vec4f,       // surface temperature per local face (+x, +y, -x, -y)
  mflux: vec4f,    // pyrolysis gas mass flux per face, kg/m^2/s
  charCov: vec4f,  // char coverage of each face (0..1)
};

struct MM4 { lo: vec4f, hi: vec4f };

const SOLID_NONE: u32 = 0xffffffffu;

fn idxC(i: i32, j: i32) -> i32 { return j * P.nx + i; }
fn idxU(i: i32, j: i32) -> i32 { return j * (P.nx + 1) + i; }
fn idxV(i: i32, j: i32) -> i32 { return j * P.nx + i; }

fn rhoOf(T: f32) -> f32 { return P.rhoAmb * P.Tamb / T; }

// Sutherland viscosity of air, Pa s
fn muOf(T: f32) -> f32 {
  return 1.716e-5 * pow(T / 273.15, 1.5) * (273.15 + 110.4) / (T + 110.4) * P.viscMult;
}
// rho*D and k/cp (unity Lewis number), kg/m/s
fn rhoDOf(T: f32) -> f32 {
  return 1.716e-5 * pow(T / 273.15, 1.5) * (273.15 + 110.4) / (T + 110.4) / P.Pr * P.diffMult;
}

fn ambY() -> vec4f { return vec4f(0.0, P.YOamb, 0.0, 0.0); }

// signed distance to a rotated square (negative inside)
fn sdBox(p: vec2f, pc: Piece) -> f32 {
  let c = cos(pc.angle); let s = sin(pc.angle);
  let d = p - pc.pos;
  let l = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
  let q = abs(l) - vec2f(pc.half);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

// local face index nearest to point p: 0:+x 1:+y 2:-x 3:-y
fn faceOf(p: vec2f, pc: Piece) -> u32 {
  let c = cos(pc.angle); let s = sin(pc.angle);
  let d = p - pc.pos;
  let l = vec2f(c * d.x + s * d.y, -s * d.x + c * d.y);
  if (abs(l.x) > abs(l.y)) { return select(2u, 0u, l.x > 0.0); }
  return select(3u, 1u, l.y > 0.0);
}
