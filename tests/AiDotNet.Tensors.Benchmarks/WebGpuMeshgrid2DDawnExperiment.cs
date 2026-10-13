#if NET7_0_OR_GREATER
using System;
using System.Diagnostics;
using System.IO;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #924: WebGPU's fused Meshgrid2D (bx_meshgrid2d, one dispatch for both grids) against the composed route it replaces
/// (one tile_axis dispatch per grid), on Dawn. Writes the WGSL WebGPU ships, then times both routes as WebGpuBackend
/// dispatches them: a uniform buffer and bind group per dispatch, submit and wait, outputs allocated per call. Both
/// routes are checked against torch.meshgrid first. Needs AIDOTNET_WEBGPU_NODE_DIR (node_modules/webgpu).
/// </summary>
internal static class WebGpuMeshgrid2DDawnExperiment
{
    public static int Run()
    {
        string? dir = Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_NODE_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(Path.Combine(dir, "node_modules", "webgpu")))
        {
            Console.WriteLine("Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
            return 1;
        }
        File.WriteAllText(Path.Combine(dir, "mesh.wgsl"), WebGpuShaderPrelude.Prepare(WebGpuBatchExecutionKernels.ByKernelName["bx_meshgrid2d"]));
        File.WriteAllText(Path.Combine(dir, "tile.wgsl"), WebGpuShaderPrelude.Prepare(WebGpuKernels.TileSource));
        string script = Path.Combine(dir, "aidotnet-meshgrid2d-bench.mjs");
        File.WriteAllText(script, Script);
        var start = new ProcessStartInfo("node", $"\"{script}\"") { WorkingDirectory = dir, UseShellExecute = false };
        using var node = Process.Start(start) ?? throw new InvalidOperationException("node did not start");
        node.WaitForExit();
        return node.ExitCode;
    }

    private const string Script = @"
// #924: WebGPU's fused Meshgrid2D against its composed route (one tile_axis dispatch per grid), on Dawn.
// Mirrors WebGpuBackend: each dispatch creates its uniform buffer and bind group, submits and waits for completion;
// outputs are allocated per call as the engine allocates them. Interleaved samples, median and p95 per route.
import { create, globals } from 'webgpu';
import fs from 'fs';
Object.assign(globalThis, globals);
const gpu = create([]);
globalThis.__keepGpu = gpu;
process.on('unhandledRejection', e => { console.log('ERROR ' + (e && e.stack || e)); process.exit(1); });
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
const info = adapter.info || {};
console.log(`adapter ${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`);
const meshMod = device.createShaderModule({ code: fs.readFileSync('mesh.wgsl', 'utf8') });
const tileMod = device.createShaderModule({ code: fs.readFileSync('tile.wgsl', 'utf8') });
const mesh = device.createComputePipeline({ layout: 'auto', compute: { module: meshMod, entryPoint: 'main' } });
const tile = device.createComputePipeline({ layout: 'auto', compute: { module: tileMod, entryPoint: 'tile_axis' } });
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
function storage(data) { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: S }); device.queue.writeBuffer(b, 0, data); return b; }
function uniform(words) { const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(u, 0, new Uint32Array([...words, 0, 0, 0, 0].slice(0, 4))); return u; }
async function dispatch(pipe, buffers, words, threads) {
  const u = uniform(words);
  const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
  entries.push({ binding: buffers.length, resource: { buffer: u } });
  const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }));
  pass.dispatchWorkgroups(Math.ceil(threads / 256)); pass.end();
  device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); u.destroy();
}
async function verify(o, expected) {
  const r = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(o, 0, r, 0, expected.byteLength); device.queue.submit([enc.finish()]);
  await r.mapAsync(GPUMapMode.READ); const got = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy();
  for (let i = 0; i < expected.length; i++) if (got[i] !== expected[i]) throw new Error(`mismatch at ${i}: ${got[i]} vs ${expected[i]}`);
}
for (const [n0, n1] of [[256, 256], [1024, 256]]) for (const xy of [false, true]) {
  const s0 = storage(Float32Array.from({ length: n0 }, (_, i) => i)), s1 = storage(Float32Array.from({ length: n1 }, (_, i) => 1000 + i));
  const total = n0 * n1, bytes = total * 4;
  const rows = xy ? n1 : n0, cols = xy ? n0 : n1;
  // Composed: grid 0 is vector 0 laid on its axis and tiled along the other; likewise grid 1 (engine TileAxis args).
  const g0 = xy ? [1, 1, n0, n1] : [n0, 1, 1, n1];   // outer, axis, inner, repeats
  const g1 = xy ? [n1, 1, 1, n0] : [1, 1, n1, n0];
  async function fused() {
    const o0 = device.createBuffer({ size: bytes, usage: S }), o1 = device.createBuffer({ size: bytes, usage: S });
    await dispatch(mesh, [s0, s1, o0, o1], [n0, n1, xy ? 1 : 0], total); return [o0, o1];
  }
  async function composed() {
    const o0 = device.createBuffer({ size: bytes, usage: S }), o1 = device.createBuffer({ size: bytes, usage: S });
    // TileParams order: inner_size, repeats, outer_size, axis_size
    await dispatch(tile, [s0, o0], [g0[2], g0[3], g0[0], g0[1]], total);
    await dispatch(tile, [s1, o1], [g1[2], g1[3], g1[0], g1[1]], total);
    return [o0, o1];
  }
  // Both routes must produce torch.meshgrid's grids before they are timed.
  const e0 = new Float32Array(total), e1 = new Float32Array(total);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { e0[r * cols + c] = xy ? c : r; e1[r * cols + c] = 1000 + (xy ? r : c); }
  for (const route of [fused, composed]) { const [a, b] = await route(); await verify(a, e0); await verify(b, e1); a.destroy(); b.destroy(); }
  async function sample(route, calls) { const t = performance.now(); for (let k = 0; k < calls; k++) { const [a, b] = await route(); a.destroy(); b.destroy(); } return (performance.now() - t) * 1000 / calls; }
  for (let w = 0; w < 30; w++) { await sample(fused, 1); await sample(composed, 1); }
  const f = [], c = [];
  for (let s = 0; s < 21; s++) { f.push(await sample(fused, 40)); c.push(await sample(composed, 40)); }
  f.sort((a, b) => a - b); c.sort((a, b) => a - b);
  const p95 = a => a[Math.floor(a.length * 0.95)];
  console.log(`(${n0},${n1}) ${xy ? 'xy' : 'ij'}: fused median ${f[10].toFixed(1)}us p95 ${p95(f).toFixed(1)} | composed median ${c[10].toFixed(1)}us p95 ${p95(c).toFixed(1)} | speedup ${(c[10] / f[10]).toFixed(2)}x p95 ${(p95(c) / p95(f)).toFixed(2)}x`);
  s0.destroy(); s1.destroy();
}
console.log('done ' + typeof globalThis.__keepGpu);
process.exit(0);
";
}
#endif