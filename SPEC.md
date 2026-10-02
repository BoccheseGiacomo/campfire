# Campfire Simulator — Specification

## Decisions (2026-10-02)
- GPU compute (WebGPU/WebGL); multiple files allowed, all static (no server logic).
- Fresh implementation, no prototype code. Real governing equations and real physical constants; a reduced model is fine (no full compressible Navier–Stokes or detailed chemistry), but combustion must be modelled faithfully.
- Gas has physical diffusion and viscosity, and turbulence is allowed to develop.
- Scene about 1 m × 1.5 m or a bit smaller; wood about 4 cm and 8 cm across, or a bit smaller.

## Decisions (round 2)
- **WebGPU only.**
- Gas: low-Mach variable-density model. Add a subgrid turbulence model or a low-diffusion advection scheme only where it is justified; don't over-engineer. Keep it realistic.
- Genuinely 2D and uniform along the depth axis. Front/back air exchange is a **control with a small default**, so it never dominates.
- Real time with **smaller wood** (sticks and logs).
- **Char and glowing coals** included. **Minimal soot scalar** for luminosity and radiation.
- Wood thermal model: **surface + core**.
- Starts with a **pre-lit default arrangement**; the user adds wood. The user can add burning pieces. **No wind control** for now.
- **Flat floor.** **Generic wood.**
- Rigid bodies: very stable, no glitches, and nothing moves unless a real physical instability causes it. **High friction.**
- Views: **photoreal + temperature** (diagnostic views allowed).
- Controls: a good number of meaningful ones, including reaction rate and diffusivity, with no two governing the same process. Don't over-engineer.

## Decisions (round 3)
- Target GPU: **GTX 1650**. Resolution modes: **Fine / Balanced / Coarse**.
- Interaction: **drop fresh wood** and **drag existing pieces** (burning or not), nothing else.
- Scene 0.7 × 1.0 m; sticks 2.5 cm, logs 5 cm.
- Default scene: cross-section of a log cabin fire (two floor logs with a gap, sticks across the top, a coal bed in the gap), lit at start.
- Static files served by a local server (`python -m http.server`), with separate clean `.wgsl` shader files.

## Original brief

(Carried over from a previous chat summary.)

Build a continuously running HTML fire simulator. It represents a small campfire or fire bowl as a **vertical 2D slice**. Gas can leave through the top and sides; the only solid boundary is the floor. Wood must remain within the scene horizontally. The unseen third dimension can be treated as approximately uniform, with a little front-to-back air exchange if needed.

**The flame must emerge from the simulation.** Use a practical reduced physical model for gas motion, temperature, an oxidizer such as oxygen, combustible gas released by wood, and inert products. Wood heats up and emits combustible gas at a temperature-dependent rate. Emission gradually consumes its mass and changes its size. The gas reacts where it meets oxidizer, producing heat that affects both the gas and nearby wood. Reaction should depend smoothly on conditions rather than use a binary ignition-temperature rule. A fire with too little wood, poor placement, insufficient oxygen, or inadequate heat should weaken and go out through those dynamics.

The visible flame should correspond to the simulated reaction and hot gas. Its height, width, movement, and response to the wood arrangement must arise from transport, mixing, combustion, and heat transfer. **Do not code a flame silhouette, prescribe high- and low-mixing zones by position, or use visual workarounds to conceal a flawed physical result.** Full high-cost fluid equations are not required, but approximations should be physically defensible and internally consistent. There is no need to simulate water, ash, soot, or smoke.

Wood pieces are small square cross-sections of logs whose long axis extends into the unseen dimension. Provide two sizes. The user drags a piece into the scene and releases it; gravity and collisions determine where it lands. Each piece may have one evolving temperature for now. Fuel-gas release rate and the amount of wood consumed per unit of gas emitted are distinct physical quantities.

**Solid-body stability is essential.** Pieces must collide without passing through each other or the floor, stack securely when supported, and have convincing static and sliding friction against wood and floor. Resting piles should not creep, chatter, bounce repeatedly, or suddenly jump as pieces shrink. Unsupported pieces should still tip and fall naturally. Numerical techniques for stable resting contacts are welcome; arbitrary glue is not.

At sensible defaults, the fire should need attention and refuelling roughly every **1–2 minutes**. Provide a small set of meaningful tuning controls and a reset to good defaults, without duplicating controls that govern the same process.

Our first prototype established the basic gas–wood interaction, but its plume stayed unusually tall and narrow. Increasing diffusion or flow mixing, changing viscosity or buoyancy, and increasing reaction rate did not reliably solve that. Its block stacks also showed occasional sliding and penetration. Treat that prototype as a reference for the desired interactions, **not as an architecture to copy**. Prioritize a believable flame and stable blocks, and check those outcomes directly while developing the new model.
