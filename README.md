# Campfire

**A physically based fire simulation, not an animation.** Nothing in the flame is drawn, scripted or shaped by hand. Wood heats up and releases combustible gas. The gas mixes with air, burns with Arrhenius chemistry, rises by buoyancy and radiates. What you see is that gas, rendered from its temperature and soot. The fluid model is a low-Mach approximation of the Navier–Stokes equations: it keeps viscosity, diffusion, buoyancy and large density changes, and leaves out only sound waves.

It runs in the browser on WebGPU. The scene is a vertical 2D slice through a campfire, with the logs running along the unseen axis.

**Live:** https://bocchesegiacomo.github.io/campfire/ (Chrome or Edge; needs WebGPU)

## What you can do

- Drop in sticks (2.5 cm) and logs (5 cm), or drag pieces that are already in the scene. They stack, slide and tip as rigid bodies with friction.
- Watch the fire sustain itself, spread or die, depending on how the wood is arranged. It needs new wood every minute or two.
- Switch between three views: photoreal, temperature, and diagnostics (heat release, O₂, fuel, soot, velocity).
- Change the physics: viscosity, diffusivity, reaction rate, soot, pyrolysis and char rates, wood density, friction, ambient O₂ and fire depth. There are three grid resolutions.

## Run locally

Any static file server works. Run this in this folder, then open http://localhost:8123:

```bash
python -m http.server 8123
```

Opening `index.html` directly from disk does not work, because browsers block module and shader loading from `file://`.

## Model

Notation:

| Symbol | Meaning |
|---|---|
| T | temperature |
| ρ | density |
| **u** | velocity |
| Y | mass fraction; subscripts F fuel, O oxygen, P products, S soot |
| ∞ | ambient value |

### Gas flow

The flow uses the low-Mach equations with variable density. Pressure is split into a constant thermodynamic part p₀ and a small dynamic part p′, so density follows temperature through the ideal-gas law:

```math
\rho = \frac{p_0 W}{R T}
```

```math
\rho \frac{D\mathbf{u}}{Dt} = -\nabla p' + \nabla\cdot(\mu \nabla \mathbf{u}) + (\rho - \rho_\infty)\,\mathbf{g}
```

```math
\rho c_p \frac{DT}{Dt} = \nabla\cdot(k \nabla T) + \dot q_\mathrm{comb} - \dot q_\mathrm{rad}
```

```math
\rho \frac{DY_k}{Dt} = \nabla\cdot(\rho D \nabla Y_k) + \dot\omega_k
```

Mass conservation becomes a condition on the velocity divergence. Gas that heats up expands, and gas released by the wood needs room. In this equation, ṁ is the gas mass added per unit volume per second:

```math
\nabla\cdot\mathbf{u} = \frac{\dot m}{\rho} + \frac{1}{T}\frac{DT}{Dt}
```

- **Transport:** viscosity μ(T) follows Sutherland's law. Conductivity is k = μc_p/Pr and diffusivity is ρD = μ/Sc, with Pr = Sc = 0.7.
- **Numerics:** the grid is a staggered MAC grid. Advection is semi-Lagrangian with a MacCormack correction. The pressure equation is solved with multigrid, and wood pieces are cut out of the grid with fractional cell faces.
- **Boundaries:** the floor is a solid wall. The sides and top are open to still air.

### Air from the front and back

A real fire also draws air in along the logs. This is modelled as plume entrainment through the front and back of a fire of depth L, with entrainment coefficient α = 0.05:

```math
\dot m_e = \frac{2 \alpha \, |\mathbf{u}| \, \rho_\infty}{L}
```

### Combustion

The gas released by the wood burns in a one-step global reaction:

```math
\mathrm{F} + s\,\mathrm{O_2} \rightarrow (1+s)\,\mathrm{P}
```

```math
\dot\omega_F = B \rho^2 Y_F Y_O \, e^{-T_a/T}
```

- The heat of combustion is ΔH = 15 MJ per kg of fuel. The oxygen ratio is s = ΔH / (13.1 MJ per kg O₂).
- The activation temperature is T_a = 15 000 K and the pre-exponential factor is B = 2·10¹⁰ m³/(kg·s). B is calibrated so that a 1D flame burns at 0.38 m/s, which `tests/flame1d.mjs` checks.
- There is no ignition temperature or threshold. The flame lights wherever the rate becomes large enough.

### Soot

Soot forms in hot, fuel-rich gas and burns where it meets oxygen:

```math
\dot Y_{S,\mathrm{form}} = A_f Y_F \, e^{-T_f/T}
```

```math
\dot Y_{S,\mathrm{ox}} = A_o Y_S Y_O \, e^{-T_o/T}
```

### Radiation

The gas loses heat by optically thin radiation. The absorption coefficient is κ = 1862·f_v·T + κ_g·Y_P, where f_v is the soot volume fraction:

```math
\dot q_\mathrm{rad} = 4 \kappa \sigma (T^4 - T_\infty^4)
```

Radiation reaching each wood face is computed by casting rays through the gas. A ray can hit another piece, reflect off the floor or escape. Because the scene is uniform along the logs, the out-of-plane part of the integral is exact. For in-plane direction φ, at angle θ to the face normal:

```math
dq = \left( \frac{1}{2} \sigma T_\mathrm{hit}^4 + 2 \int \frac{\kappa \sigma T^4}{\pi} \, ds \right) \cos\theta \, d\varphi
```

### Wood

Each face of a piece has a surface temperature T_s and a pyrolysis-front temperature T_p, separated by a char layer. The piece also has one core temperature.

Pyrolysis turns wood into gas and char at a rate that rises smoothly with temperature (A = 2·10⁷ s⁻¹, T_a = 12 000 K). The fraction Y_c becomes char:

```math
\dot m = A \, e^{-T_a/T_p} \, m_\mathrm{front}, \qquad \dot m_\mathrm{gas} = (1 - Y_c)\,\dot m, \qquad \dot m_\mathrm{char} = Y_c \, \dot m
```

The surface energy balance is:

```math
C_s \frac{dT_s}{dt} = q_\mathrm{conv} + \varepsilon (q_\mathrm{inc} - \sigma T_s^4) + q_\mathrm{char} - G (T_s - T_p) - \dot m_\mathrm{gas} c_{p,g} (T_s - T_p)
```

The terms are convection from the gas, absorbed minus emitted radiation, heat from burning char, conduction into the wood, and cooling by the escaping gas.

- **Char conductivity** includes radiation across its pores: k_c = k₀ + 4εσT³·d_pore.
- **Char burns at its surface** with oxygen from the gas. It smoulders slowly at about 700 K, and at high temperature its rate is limited by the available oxygen:

```math
\dot m_{O_2} = A_c \, e^{-T_c/T_s} \, \rho Y_O
```

- **Touching pieces exchange heat** through the thin air gap between them and by radiation across it. The floor does not conduct heat.

```math
h = \frac{k_\mathrm{air}}{\delta} + \frac{4 \sigma T^3}{2/\varepsilon - 1}, \qquad \delta = 0.5\ \mathrm{mm}
```

- **Coupling to the gas:** heat and mass pass between the wood and the gas cells next to it, and both are conserved. A piece's size follows from its remaining wood and char, so it shrinks as it burns.

### Rigid bodies

- **Solver:** contacts are solved with sub-steps and soft constraints, and warm-started from the previous step.
- **Friction:** Coulomb friction with separate static and kinetic coefficients, so resting piles do not creep.
- **Dragging:** a held piece follows the pointer on a spring that keeps its angle. The spring's force is limited, so it cannot shove a whole stack aside.

### Rendering

Looking along the logs, the gas is a uniform layer of depth L. Each pixel's spectral radiance comes from soot emission (Planck's law B_λ, with soot absorption k_λ):

```math
L_\lambda = \left(1 - e^{-k_\lambda L}\right) B_\lambda(T), \qquad k_\lambda = \frac{6.3 \, f_v}{\lambda}
```

The blue glow of the flame front is added in proportion to the heat release. The spectrum is converted to screen colour with the CIE 1931 colour-matching functions, then a camera-like tone curve. The visible log ends show the simulated char ring and temperature.

## Code

| Path | Contents |
|---|---|
| `js/main.js` | app loop, UI, input, default scene |
| `js/fluid.js`, `js/gpu.js` | GPU buffers, kernels, shader loading |
| `js/wood.js`, `js/contact.js` | wood heating and pyrolysis, contact conduction |
| `js/rigid.js` | rigid-body solver |
| `js/config.js` | physical constants and defaults |
| `shaders/*.wgsl` | GPU code: advection, diffusion, chemistry, forces, pressure, radiation, rendering |

## Tests

These run with Node 18 or later:

```bash
node tests/rigid_test.mjs
```

```bash
node tests/wood_test.mjs
```

```bash
node tests/char_test.mjs
```

```bash
node tests/flame1d.mjs B=2e10
```

They check, in order:

1. Stacking, friction, tipping and dragging.
2. Ignition time under a known heat flux.
3. Smouldering, embers dying or surviving, and contact conduction.
4. Flame speed.

## Limitations

- The flame front is thinner than a grid cell, so on coarse grids it spreads faster than a real flame.
- A pure 2D slice gets air only from the open sides and top, plus the front/back entrainment.
- Gaps narrower than about two grid cells are not resolved, so tightly packed coals get little oxygen.
- There is no ash or moisture.
