// Physical constants, model parameters and resolution modes. SI units.

export const SCENE_WIDTH = 0.7;   // m (fixed; height follows from the grid)

// Grid dims are multiples of 32 so the multigrid hierarchy has 5 clean levels.
export const MODES = {
  coarse:   { nx: 160, ny: 224, dt: 0.004 },
  balanced: { nx: 224, ny: 320, dt: 0.003 },
  fine:     { nx: 352, ny: 512, dt: 0.002 },
};

export const PHYS = {
  // ambient
  Tamb: 293.15,           // K
  p0: 101325,             // Pa
  Wmix: 0.0289,           // kg/mol, one molecular weight for all gas species
  R: 8.314462,
  g: 9.81,
  YO2air: 0.232,          // O2 mass fraction of air
  // gas transport
  cp: 1300,               // J/kg/K, mean for air/combustion products over 300-2000 K
  Pr: 0.7,                // Prandtl (= Schmidt: unity Lewis number)
  // gas-phase combustion: Fuel + s O2 -> (1+s) Products
  dHc: 15.0e6,            // J/kg of volatiles (wood pyrolysate, incl. its inert part)
  dHO2: 13.1e6,           // J per kg O2 consumed (Huggett) -> s = dHc/dHO2
  reactB: 2.0e10,         // m^3/(kg s): w = B rho^2 YF YO exp(-Ta/T); laminar flame speed 0.38 m/s (tests/flame1d.mjs)
  reactTa: 15000,         // K activation temperature (Ea ~ 125 kJ/mol)
  // soot (condensed fuel; burns with the same heat and O2 demand as fuel)
  sootFormA: 6.0e6,       // 1/s  (~1/s at 1600 K; inception needs ~1300 K+ in fuel-rich gas)
  sootFormTa: 25000,      // K   (~210 kJ/mol)
  sootOxA: 5.4e8,         // 1/s
  sootOxTa: 19800,        // K
  sootDensity: 1800,      // kg/m^3
  kappaSootC: 1862,       // 1/(m K): kappa_soot = C fv T
  kappaGas: 0.8,          // 1/m per unit product mass fraction (CO2+H2O Planck mean)
  // front/back exchange (entrainment through the finite depth)
  exchAlpha: 0.05,
  // char surface oxidation  C + O2 -> CO2  (m_O2'' = A exp(-Ta/Ts) rho YO2)
  // apparent surface rate of porous wood char (pore-diffusion regime: about half the
  // intrinsic activation energy): smoulders ~0.1 g/m^2/s at 700 K, k = 0.02 m/s at 1000 K
  charA: 6.6,             // m/s
  charTa: 5800,           // K (~48 kJ/mol)
  charO2: 2.667,          // kg O2 per kg C
  sigma: 5.670374e-8,
};
PHYS.rhoAmb = PHYS.p0 * PHYS.Wmix / (PHYS.R * PHYS.Tamb);
PHYS.sO2 = PHYS.dHc / PHYS.dHO2;

// User-tunable controls (multipliers default to 1 = real values)
export const CONTROL_DEFAULTS = {
  viscMult: 1,
  diffMult: 1,
  reactMult: 1,
  sootMult: 1,
  pyroMult: 1,
  charMult: 1,
  charYield: 0.25,
  woodDensity: 500,
  friction: 0.7,
  ambientO2: 20.9,        // % by volume
  depth: 0.5,             // m, fire depth along the unseen axis (log length)
  speed: 1,
  quality: 'balanced',
  view: 'photo',
  exposure: 1,
};
