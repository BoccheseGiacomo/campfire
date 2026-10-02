// Full-screen fragment renderer of the simulated fields.
import { loadWGSL, makeBuffer } from './gpu.js';
import { MAX_PIECES } from './fluid.js';

export const VIEWS = { photo: 0, temperature: 1, reaction: 2, oxygen: 3, fuel: 4, velocity: 5, soot: 6 };
export const GROUND = 0.03;   // m of ground shown below the floor line

export class Renderer {
  constructor(device, context, format) {
    this.device = device; this.context = context; this.format = format;
    this.rp = makeBuffer(device, 'renderParams', 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.wood = makeBuffer(device, 'woodRender', MAX_PIECES * 48);
  }

  async init(sim) {
    const d = this.device;
    const code = await loadWGSL('shaders/render.wgsl');
    const module = d.createShaderModule({ label: 'render', code });
    const ro = { type: 'read-only-storage' };
    const vis = GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX;
    const layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: vis, buffer: { type: 'uniform' } },
        { binding: 1, visibility: vis, buffer: { type: 'uniform' } },
        ...[2, 3, 4, 5, 6, 7, 8, 9].map(b => ({ binding: b, visibility: GPUShaderStage.FRAGMENT, buffer: ro })),
      ],
    });
    this.pipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    const B = sim.buf;
    this.bg = d.createBindGroup({
      layout,
      entries: [B.P, this.rp, B.T, B.Y, B.SQ, B.pieces, B.U, B.V, this.wood, B.rad]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.sim = sim;
  }

  writeParams(o) {
    const dv = new DataView(new ArrayBuffer(48));
    dv.setInt32(0, o.view, true); dv.setInt32(4, o.nPieces, true);
    dv.setFloat32(8, o.exposure, true); dv.setFloat32(12, this.sim.nx * this.sim.dx, true);
    dv.setFloat32(16, this.sim.ny * this.sim.dx, true); dv.setFloat32(20, GROUND, true);
    dv.setFloat32(24, o.time, true); dv.setFloat32(28, o.tmax ?? 1800, true);
    dv.setFloat32(32, o.chemi ?? 3e-7, true);   // fraction of heat release emitted as CH*/C2* light
    const cam = o.cam || { x: 0, y: -GROUND, zoom: 1 };
    dv.setFloat32(36, cam.x, true); dv.setFloat32(40, cam.y, true); dv.setFloat32(44, cam.zoom, true);
    this.device.queue.writeBuffer(this.rp, 0, dv.buffer);
  }

  writeWood(data) { this.device.queue.writeBuffer(this.wood, 0, data); }

  encode(encoder) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bg);
    pass.draw(3);
    pass.end();
  }
}
