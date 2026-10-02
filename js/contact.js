// Heat conduction between touching wood pieces.
//
// Real wood and char surfaces touch at asperities with a thin gas gap between
// them; heat crosses by conduction through the gap gas and by radiation across
// it. Per unit contact area:  h = k_gas(T)/gap + 4 sigma T^3 / (2/eps - 1).
// The rigid-body contacts give, for every touching pair, the contact normal and
// the length of the shared contact line; the two face surface nodes then relax
// toward each other exactly over the frame (energy conserving, stable for any
// capacities). The floor is adiabatic and takes part in no exchange.
import { WOOD } from './wood.js';

export const CONTACT = {
  gap: 0.0005,        // m, effective gas gap between rough surfaces
  pointLength: 0.001, // m, contact length of a corner touching a face
  maxSep: 0.0005,     // m, contact points farther apart than this do not conduct
};

// face index of a body most aligned with world direction (nx, ny): 0:+x 1:+y 2:-x 3:-y (local)
function faceToward(a, nx, ny) {
  const c = Math.cos(a), s = Math.sin(a);
  const lx = c * nx + s * ny, ly = -s * nx + c * ny;
  if (Math.abs(lx) > Math.abs(ly)) return lx > 0 ? 0 : 2;
  return ly > 0 ? 1 : 3;
}

export function contactConductance(T) {
  const kGas = 0.0241 * Math.pow(T / 273.15, 0.8);      // air
  const eps = WOOD.eps;
  return kGas / CONTACT.gap + 4 * WOOD.sigma * T * T * T / (2 / eps - 1);
}

// Exchange heat over dt across all wood-wood contacts. Returns the energy moved (J/m).
export function exchangeContacts(world, dt) {
  let moved = 0;
  if (dt <= 0) return moved;
  for (const m of world.manifolds.values()) {
    const pa = m.A.userData, pb = m.B.userData;
    if (!pa || !pb) continue;                       // floor / walls: adiabatic
    const pts = m.points.filter(c => c.sep < CONTACT.maxSep);
    if (!pts.length) continue;
    const L = pts.length >= 2 ? Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) : CONTACT.pointLength;
    if (L <= 0) continue;
    const wa = pa.wood, wb = pb.wood;
    const fa = faceToward(m.A.a, m.nx, m.ny);       // normal points from A to B
    const fb = faceToward(m.B.a, -m.nx, -m.ny);
    const Ta = wa.Ts[fa], Tb = wb.Ts[fb];
    const G = contactConductance(0.5 * (Ta + Tb)) * L;   // W/K per m
    const Ca = wa.surfaceCap(fa), Cb = wb.surfaceCap(fb);
    const inv = 1 / Ca + 1 / Cb;
    const E = (Ta - Tb) * (1 - Math.exp(-G * inv * dt)) / inv;
    wa.Ts[fa] -= E / Ca;
    wb.Ts[fb] += E / Cb;
    moved += Math.abs(E);
  }
  return moved;
}
