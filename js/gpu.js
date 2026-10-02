// WebGPU helpers: WGSL preprocessing, kernels with bindings derived from the
// shader source, buffer creation.

const shaderCache = new Map();
const ROOT = new URL('../', import.meta.url);   // paths are relative to the app root
async function fetchText(path) {
  if (!shaderCache.has(path)) {
    const r = await fetch(new URL(path, ROOT), { cache: 'no-store' });
    if (!r.ok) throw new Error(`failed to load ${path}`);
    shaderCache.set(path, await r.text());
  }
  return shaderCache.get(path);
}

// Bilinear samplers in grid coordinates (cell units, see common.wgsl).
function genSampler(kind, name, amb) {
  if (kind === 'C' || kind === 'C4') {
    const ty = kind === 'C' ? 'f32' : 'vec4f';
    const mm = kind === 'C'
      ? `fn mm_${name}(g: vec2f) -> vec2f {
  let b = floor(g - vec2f(0.5)); let i = i32(b.x); let j = i32(b.y);
  let a = f_${name}(i, j); let c = f_${name}(i + 1, j); let d = f_${name}(i, j + 1); let e = f_${name}(i + 1, j + 1);
  return vec2f(min(min(a, c), min(d, e)), max(max(a, c), max(d, e)));
}`
      : `fn mm_${name}(g: vec2f) -> MM4 {
  let b = floor(g - vec2f(0.5)); let i = i32(b.x); let j = i32(b.y);
  let a = f_${name}(i, j); let c = f_${name}(i + 1, j); let d = f_${name}(i, j + 1); let e = f_${name}(i + 1, j + 1);
  return MM4(min(min(a, c), min(d, e)), max(max(a, c), max(d, e)));
}`;
    return `
// open sides/top sample the ambient; the floor is zero-gradient
fn f_${name}(i: i32, j: i32) -> ${ty} {
  if (i < 0 || i >= P.nx || j >= P.ny) { return ${amb}; }
  return ${name}[idxC(i, max(j, 0))];
}
fn s_${name}(g: vec2f) -> ${ty} {
  let f = g - vec2f(0.5); let b = floor(f); let t = f - b; let i = i32(b.x); let j = i32(b.y);
  return mix(mix(f_${name}(i, j), f_${name}(i + 1, j), t.x), mix(f_${name}(i, j + 1), f_${name}(i + 1, j + 1), t.x), t.y);
}
${mm}`;
  }
  if (kind === 'U' || kind === 'V') {
    const fetch = kind === 'U'
      ? `if (i < 0 || i > P.nx || j >= P.ny) { return 0.0; }   // still ambient air outside
  if (j < 0) { return -${name}[idxU(i, 0)]; }             // no-slip floor ghost
  return ${name}[idxU(i, j)];`
      : `if (i < 0 || i >= P.nx || j > P.ny) { return 0.0; }   // still ambient air outside
  return ${name}[idxV(i, max(j, 0))];`;
    const off = kind === 'U' ? 'vec2f(0.0, 0.5)' : 'vec2f(0.5, 0.0)';
    return `
fn f_${name}(i: i32, j: i32) -> f32 {
  ${fetch}
}
fn s_${name}(g: vec2f) -> f32 {
  let f = g - ${off}; let b = floor(f); let t = f - b; let i = i32(b.x); let j = i32(b.y);
  return mix(mix(f_${name}(i, j), f_${name}(i + 1, j), t.x), mix(f_${name}(i, j + 1), f_${name}(i + 1, j + 1), t.x), t.y);
}
fn mm_${name}(g: vec2f) -> vec2f {
  let b = floor(g - ${off}); let i = i32(b.x); let j = i32(b.y);
  let a = f_${name}(i, j); let c = f_${name}(i + 1, j); let d = f_${name}(i, j + 1); let e = f_${name}(i + 1, j + 1);
  return vec2f(min(min(a, c), min(d, e)), max(max(a, c), max(d, e)));
}`;
  }
  throw new Error('unknown sampler kind ' + kind);
}

export async function loadWGSL(path) {
  const src = await fetchText(path);
  const out = [];
  for (const line of src.split(/\r?\n/)) {
    let m;
    if ((m = line.match(/^\s*\/\/#include\s+(\w+)/))) out.push(await fetchText(`shaders/${m[1]}.wgsl`));
    else if ((m = line.match(/^\s*\/\/#sampler\s+(\w+)\s+(\w+)\s*(.*)$/))) out.push(genSampler(m[1], m[2], m[3].trim()));
    else out.push(line);
  }
  return out.join('\n');
}

// --- static analysis: which module-scope bindings an entry point uses ---
function analyze(code) {
  const globals = [];
  const reG = /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<([^>]+)>)?\s+(\w+)\s*:\s*([^;]+);/g;
  let m;
  while ((m = reG.exec(code))) {
    const space = (m[2] || '').replace(/\s/g, '');
    let type;
    if (space === 'uniform') type = 'uniform';
    else if (space === 'storage,read' || space === 'storage') type = 'read-only-storage';
    else type = 'storage';
    globals.push({ binding: +m[1], name: m[3], type });
  }
  const fns = {};
  const reF = /fn\s+(\w+)\s*\(/g;
  while ((m = reF.exec(code))) {
    let k = code.indexOf('{', m.index);
    let depth = 0, start = k;
    for (; k < code.length; k++) {
      if (code[k] === '{') depth++;
      else if (code[k] === '}') { depth--; if (depth === 0) break; }
    }
    // include the signature (parameter names shadow nothing we care about)
    fns[m[1]] = code.slice(m.index, k + 1);
  }
  return { globals, fns };
}

function usedBindings(info, entry) {
  const seen = new Set();
  const stack = [entry];
  let text = '';
  while (stack.length) {
    const f = stack.pop();
    if (seen.has(f) || !info.fns[f]) continue;
    seen.add(f);
    const body = info.fns[f];
    text += body;
    for (const g of Object.keys(info.fns)) if (!seen.has(g) && new RegExp(`\\b${g}\\s*\\(`).test(body)) stack.push(g);
  }
  return info.globals.filter(g => new RegExp(`\\b${g.name}\\b`).test(text));
}

export class Kernel {
  constructor(device, code, entry, label = entry) {
    this.device = device;
    this.label = label;
    const info = analyze(code);
    this.bindings = usedBindings(info, entry);
    const layout = device.createBindGroupLayout({
      label,
      entries: this.bindings.map(b => ({
        binding: b.binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: b.type },
      })),
    });
    this.layout = layout;
    const module = device.createShaderModule({ label, code });
    this.module = module;
    this.pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: entry },
    });
  }
  bind(map) {
    return this.device.createBindGroup({
      label: this.label,
      layout: this.layout,
      entries: this.bindings.map(b => {
        const buf = map[b.name];
        if (!buf) throw new Error(`${this.label}: no buffer for '${b.name}'`);
        return { binding: b.binding, resource: { buffer: buf } };
      }),
    });
  }
}

export function makeBuffer(device, label, size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
  return device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 4) * 4), usage });
}
