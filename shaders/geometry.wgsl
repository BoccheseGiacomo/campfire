// Rasterise wood pieces into the grid: per cell the signed distance to the
// nearest piece and which piece/face it is; per face the open fraction used by
// the projection (fractional face areas give smooth motion as pieces shrink).
//#include common

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pieces: array<Piece>;
@group(0) @binding(2) var<storage, read_write> cellG: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> fx: array<f32>;
@group(0) @binding(4) var<storage, read_write> fy: array<f32>;
@group(0) @binding(5) var<storage, read_write> faceW: array<atomic<i32>>;

// Wood/gas exchange is smeared over a two-cell band outside each face with a
// hat weight w = max(0, 1 - |d - dx|/dx)/dx, which integrates to the face length
// for any grid offset (partition of unity). The band's total weight per face is accumulated
// here so the sources pass can normalise it to the face's true length
// (exactly conservative exchange), and so the CPU knows how exposed each face is.
const W_SCALE: f32 = 1.0e8;

fn nearest(p: vec2f) -> vec2f {
  var best = 1e9; var bid = SOLID_NONE;
  for (var n = 0; n < P.nPieces; n++) {
    let pc = pieces[n];
    if (pc.alive <= 0.0) { continue; }
    let d = sdBox(p, pc);
    if (d < best) { best = d; bid = u32(n) * 4u + faceOf(p, pc); }
  }
  return vec2f(best, bitcast<f32>(bid));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x); let j = i32(id.y);
  let dx = P.dx;
  if (i < P.nx && j < P.ny) {
    let g = nearest(vec2f(f32(i) + 0.5, f32(j) + 0.5) * dx);
    cellG[idxC(i, j)] = g;
    let id = bitcast<u32>(g.y);
    if (id != SOLID_NONE && g.x >= 0.0 && g.x < 2.0 * dx) {
      let w = max(0.0, 1.0 - abs(g.x - dx) / dx) / dx;
      atomicAdd(&faceW[id], i32(round(w * dx * dx * W_SCALE)));
    }
  }
  if (i <= P.nx && j < P.ny) {
    let d = nearest(vec2f(f32(i), f32(j) + 0.5) * dx).x;
    fx[idxU(i, j)] = clamp(0.5 + d / dx, 0.0, 1.0);
  }
  if (i < P.nx && j <= P.ny) {
    let d = nearest(vec2f(f32(i) + 0.5, f32(j)) * dx).x;
    fy[idxV(i, j)] = select(clamp(0.5 + d / dx, 0.0, 1.0), 0.0, j == 0);
  }
}
