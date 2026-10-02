# Campfire

A campfire simulated in the browser with WebGPU. The scene is a vertical 2D slice through a fire whose logs run along the unseen axis. The flame is not drawn or animated: what you see is the simulated gas, rendered from its temperature and soot.

**Live:** https://bocchesegiacomo.github.io/campfire/ (Chrome or Edge; needs WebGPU)

## What it does

- Wood pieces (2.5 cm sticks and 5 cm logs) are rigid squares. Drag them in from the panel, or pick up pieces in the scene. They stack, slide, tip and come to rest under gravity and friction.
- Hot wood releases combustible gas (pyrolysis) and turns to char. The gas burns where it meets air; the heat comes back to the wood by convection, radiation and contact, so the fire sustains, spreads or dies depending on how the wood is arranged.
- Char smoulders and glows, and pieces shrink as they lose mass.
- With the default physics the fire needs more wood every minute or two.
- Views: photoreal, temperature, and diagnostics (heat release, O₂, fuel, soot, velocity). The panel has physical controls (viscosity, diffusivity, reaction rate, soot, pyrolysis and char rates, char yield, wood density, friction, ambient O₂, fire depth) and three grid resolutions.

## Run locally

Any static file server works:

```bash
python -m http.server 8123
```

Run it in this folder, then open http://localhost:8123. Opening `index.html` directly from disk does not work, because browsers block module and shader loading from `file://`.

## Model

Notation: T temperature, ρ density, **u** velocity, Y_k mass fractions (fuel F, O₂ O, products P, soot S), subscript ∞ ambient.

### Gas (low-Mach number, variable density)

Pressure is thermodynamically constant (p₀), so sound waves are filtered out, but density varies strongly with temperature:

$$\rho = \frac{p_0 W}{R T}$$

$$\rho\frac{D\mathbf u}{Dt} = -\nabla p' + \nabla\cdot(\mu\nabla\mathbf u) + (\rho-\rho_\infty)\,\mathbf g$$

$$\rho c_p\frac{DT}{Dt} = \nabla\cdot(k\nabla T) + \dot q_{comb} - \dot q_{rad}$$

$$\rho\frac{DY_k}{Dt} = \nabla\cdot(\rho D\nabla Y_k) + \dot\omega_k$$

Mass conservation becomes a constraint on the divergence: thermal expansion and injected gas make room for themselves.

$$\nabla\cdot\mathbf u = \frac{\dot m'''}{\rho} + \frac{1}{T}\frac{DT}{Dt}$$

- **Transport:** μ(T) from Sutherland's law; k = μc_p/Pr and ρD = μ/Sc with Pr = Sc = 0.7. Soot does not diffuse.
- **Grid:** staggered MAC.
- **Advection:** semi-Lagrangian with an RK2 back-trace and a MacCormack correction. It falls back to first order where the correction would leave the local bounds or create unphysical fuel–oxygen mixtures.
- **Pressure:** a variable-coefficient Poisson equation solved by multigrid V-cycles: ∇·(f/ρ ∇p') = (∇·(f**u**\*) − S)/Δt, where f is each face's open fraction (wood pieces are cut out with fractional faces).
- **Boundaries:** the floor is a no-slip wall. The sides and top are open to still air: outflow keeps its momentum, inflow arrives from rest.

### Third dimension

The slice is uniform along the logs. Air enters through the front and back faces of a fire of depth L by plume entrainment (Morton–Taylor–Turner, α = 0.05):

$$\dot m'''_e = \frac{2\alpha\,|\mathbf u|\,\rho_\infty}{L}$$

### Combustion

One-step global reaction of the wood volatiles:

$$\mathrm{F} + s\,\mathrm{O_2} \rightarrow (1+s)\,\mathrm{P}$$

$$\dot\omega_F = B\,\rho^2\,Y_F\,Y_O\,e^{-T_a/T}$$

- **Heat release:** ΔH = 15 MJ/kg of volatiles; s = ΔH / 13.1 MJ/kg O₂ (oxygen-consumption principle).
- **Kinetics:** T_a = 15 000 K and B = 2·10¹⁰ m³/(kg·s). B is chosen so a 1D premixed flame with this gas model propagates at 0.38 m/s (`tests/flame1d.mjs`).
- **Integration:** the reaction is sub-stepped within each cell. There is no ignition temperature: the rate is continuous in T and composition.

### Soot

Soot forms from hot fuel-rich gas and burns as condensed fuel:

$$\dot Y_S^{form} = A_f\,Y_F\,e^{-T_f/T}\qquad(T_f = 25\,000\ \mathrm K)$$

$$\dot Y_S^{ox} = A_o\,Y_S\,Y_O\,e^{-T_o/T}$$

### Radiation

- **Gas loss:** optically thin, with κ = 1862 f_v T + κ_g Y_P.

$$\dot q_{rad} = 4\kappa\sigma\,(T^4 - T_\infty^4)$$

- **Flux onto wood:** rays are cast from every wood face (and floor probes) through the gas. They hit other pieces, reflect once off the floor, or escape to the surroundings. Because the scene is uniform along the unseen axis, the out-of-plane integral is analytic. For in-plane direction φ, with cosine c to the face normal:

$$dq = \Big(\tfrac{1}{2}\sigma T_{hit}^4 + 2\!\int\!\tfrac{\kappa\sigma T^4}{\pi}\,ds\Big)\,c\,d\varphi$$

### Wood

Each piece has, per face, a surface node T_s and a pyrolysis-front node T_p beneath a char layer of thickness d, plus one shared core node T_c.

- **Pyrolysis (single-step Arrhenius):** A = 2·10⁷ s⁻¹, T_a = 12 000 K. The front releases gas and leaves char:

$$\dot m = A\,e^{-T_a/T_p}\,m_{front},\qquad \text{gas} = (1-Y_c)\,\dot m,\quad \text{char} = Y_c\,\dot m$$

- **Surface energy balance (exposed part of the face):**

$$C_s\dot T_s = q_{conv} + \varepsilon\,(q_{inc} - \sigma T_s^4) + q_{char} - G_{sp}(T_s - T_p) - \dot m_g c_{p,g}(T_s - T_p)$$

- **Char conductivity** includes radiation across pores: k_c = k₀ + 4εσT³·d_pore.
- **Char oxidation** uses the apparent surface rate of porous char, about 48 kJ/mol. That is about half the intrinsic value, as expected in the pore-diffusion regime. It smoulders at about 0.1 g/m²s near 700 K and is limited by the oxygen in the gas at high temperature:

$$\dot m''_{O_2} = A_c\,e^{-T_c/T_s}\,\rho\,Y_O$$

- **Contact conduction** between touching pieces, through the air gap plus radiation across it, over the contact length reported by the rigid-body solver. Each contact is solved as an exact two-node relaxation. The floor is adiabatic.

$$h = \frac{k_{air}(T)}{\delta} + \frac{4\sigma T^3}{2/\varepsilon - 1},\qquad \delta = 0.5\ \mathrm{mm}$$

- **Coupling to the gas:** wood–gas exchange is spread over a two-cell band normalised to each face's length, so heat and mass are conserved exactly. Gas from covered faces exits through the open ones. Each piece's size follows from its remaining virgin and char volume.

### Rigid bodies

- **Solver:** soft-step (sub-steps, soft contacts, relax pass, warm starting) with two-point box manifolds from SAT and clipping, plus speculative contacts.
- **Friction:** Coulomb, with separate static and kinetic coefficients, solved once per contact face. A persistent material anchor holds static contacts, so resting piles do not creep.
- **Dragging:** pieces are held by a critically damped spring that carries their weight and keeps their angle. Its extra force is limited to 2.5× the piece's weight, so it cannot bulldoze a stack.

### Rendering

Looking along the logs, the gas is a uniform slab of depth L, so the spectral radiance is exact. Soot emits through the depth with Rayleigh absorption, and CH*/C₂* chemiluminescence is proportional to the heat release:

$$L_\lambda = \big(1 - e^{-k_\lambda L}\big)\,B_\lambda(T),\qquad k_\lambda = \frac{6.3\,f_v}{\lambda}$$

- **Colour:** spectra are integrated with CIE 1931 colour-matching functions, converted to sRGB, compressed logarithmically in luminance, and given a film-like per-channel shoulder.
- **Log ends:** drawn as cross-sections, showing the simulated char ring and temperature profile glowing as a grey body.

## Code

| Path | Contents |
|---|---|
| `js/main.js` | app loop, UI, input, default scene |
| `js/fluid.js`, `js/gpu.js` | GPU buffers, kernels, WGSL preprocessing |
| `js/wood.js`, `js/contact.js` | wood thermal and pyrolysis model, contact conduction |
| `js/rigid.js` | rigid-body solver |
| `js/config.js` | physical constants and defaults |
| `shaders/*.wgsl` | advection, diffusion, sources (chemistry, soot, radiation loss, wood coupling), forces, pressure multigrid, geometry, radiation rays, rendering |

## Tests

Run with Node 18 or later.

```bash
node tests/rigid_test.mjs
```

Stacks, piles, friction on inclines, overhangs, shrinking piles, impacts, dragging, performance.

```bash
node tests/wood_test.mjs
```

Ignition time and pyrolysis mass flux under a known heat flux.

```bash
node tests/char_test.mjs
```

Smouldering rate, a lone ember dying, an ember inside a hot pile sustaining, contact conduction and its energy conservation.

```bash
node tests/flame1d.mjs B=2e10
```

Laminar flame speed and its grid dependence.

`dev/gas_test.html` is the gas-only test page (burner, buoyant blob, per-stage GPU timing). Add `?zoom=2&cx=0.35&cy=0.15` to the app URL for a closer camera.

## Limitations

- The flame front is thinner than a grid cell, so on coarse grids it propagates faster than a real flame. A thickened-flame model would fix this.
- In a pure 2D slice, a cavity closed by wood and the floor gets air only from above and through the front/back entrainment.
- Gaps smaller than about two grid cells are not resolved, so tightly packed coals get little oxygen.
- No ash, no moisture, no smoke absorption in the rendering.
