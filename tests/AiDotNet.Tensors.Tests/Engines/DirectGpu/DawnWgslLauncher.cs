#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Runs the WebGPU backend's shared-kernel WGSL ports (<see cref="WebGpuBatchExecutionKernels.ByKernelName"/>) on
/// Dawn (dawn.node), so their numerics are checked on a real WebGPU implementation: each launch downloads its
/// argument buffers from a host backend, binds them exactly as WebGpuBackend.DispatchBatchAsync does (buffers in
/// argument order, every scalar one 4-byte uniform field, ceil(threads / 256) workgroups), and writes the read_write
/// bindings back in place. Needs AIDOTNET_WEBGPU_NODE_DIR (a directory whose node_modules holds "webgpu").
/// </summary>
internal sealed class DawnWgslLauncher : IDisposable
{
    private static readonly Regex Binding = new(@"@binding\((\d+)\) var<storage, (read|read_write)>", RegexOptions.None, TimeSpan.FromSeconds(1));

    private const string Server = @"
import { create, globals } from 'webgpu';
import readline from 'readline';
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
const pipelines = new Map();
const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  const job = JSON.parse(line);
  try {
    let pipe = pipelines.get(job.code);
    if (!pipe) {
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code: job.code });
      pipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const err = await device.popErrorScope();
      if (err) throw new Error(err.message);
      pipelines.set(job.code, pipe);
    }
    const bufs = job.buffers.map(b => {
      const bytes = Buffer.from(b, 'base64');
      const buf = device.createBuffer({ size: Math.max(16, bytes.length), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, bytes);
      return { buf, size: bytes.length };
    });
    const ubytes = Buffer.from(job.uniform, 'base64');
    const ubuf = device.createBuffer({ size: Math.max(16, ubytes.length), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ubuf, 0, ubytes);
    const entries = bufs.map((b, i) => ({ binding: i, resource: { buffer: b.buf } }));
    entries.push({ binding: bufs.length, resource: { buffer: ubuf } });
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(job.workgroups); pass.end();
    const reads = bufs.map(b => {
      const r = device.createBuffer({ size: Math.max(16, b.size), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(b.buf, 0, r, 0, Math.max(16, b.size));
      return r;
    });
    device.queue.submit([enc.finish()]);
    const out = [];
    for (let i = 0; i < reads.length; i++) {
      await reads[i].mapAsync(GPUMapMode.READ);
      out.push(Buffer.from(reads[i].getMappedRange(0, Math.max(16, bufs[i].size)).slice(0, bufs[i].size)).toString('base64'));
      reads[i].unmap(); reads[i].destroy(); bufs[i].buf.destroy();
    }
    ubuf.destroy();
    console.log(JSON.stringify({ ok: true, out }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: String(e && e.message || e) }));
  }
}
";

    private readonly Process _node;
    private readonly IReadOnlyDictionary<string, string>? _modules;

    private DawnWgslLauncher(Process node, IReadOnlyDictionary<string, string>? modules) { _node = node; _modules = modules; }

    /// <summary>A launcher, or null when AIDOTNET_WEBGPU_NODE_DIR is not set up.</summary>
    /// <param name="modules">Kernel name to WGSL module; null runs the batch-execution ports.</param>
    public static DawnWgslLauncher? TryCreate(IReadOnlyDictionary<string, string>? modules = null)
    {
        string? dir = Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_NODE_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(Path.Combine(dir, "node_modules", "webgpu"))) return null;
        string script = Path.Combine(dir, "aidotnet-wgsl-server.mjs");
        File.WriteAllText(script, Server);
        var start = new ProcessStartInfo("node", $"\"{script}\"")
        {
            WorkingDirectory = dir,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        var node = Process.Start(start);
        return node is null ? null : new DawnWgslLauncher(node, modules);
    }

    /// <summary>A <see cref="PortableGenericFft.Launcher"/> that runs each launch's WGSL port on Dawn.</summary>
    public PortableGenericFft.Launcher For(IDirectGpuBackend host) => (name, threads, args) => Launch(host, name, threads, args);

    private void Launch(IDirectGpuBackend host, string name, int threads, object[] args)
    {
        if (threads <= 0) return;
        string? source = null;
        if (!(_modules?.TryGetValue(name, out source) ?? false) && !WebGpuBatchExecutionKernels.ByKernelName.TryGetValue(name, out source))
            throw new InvalidOperationException($"No WGSL port for {name}.");
        string code = WebGpuShaderPrelude.Prepare(source ?? "");
        AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.OnLaunch();   // device work, as a backend's own launch counts it
        var writable = new HashSet<int>();
        foreach (Match m in Binding.Matches(code))
            if (m.Groups[2].Value == "read_write") writable.Add(int.Parse(m.Groups[1].Value));

        var buffers = new List<IGpuBuffer>();
        var encoded = new List<string>();
        var fields = new List<byte>();
        foreach (var arg in args)
        {
            switch (arg)
            {
                case IGpuBuffer buffer:
                    float[] data = host.DownloadBuffer(buffer);
                    var bytes = new byte[data.Length * 4];
                    Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                    buffers.Add(buffer);
                    encoded.Add(Convert.ToBase64String(bytes));
                    break;
                case int value: fields.AddRange(BitConverter.GetBytes(value)); break;
                case float value: fields.AddRange(BitConverter.GetBytes(value)); break;
                case ulong value: fields.AddRange(BitConverter.GetBytes(value)); break;
                default: throw new ArgumentException($"Unsupported argument {arg?.GetType().Name ?? "null"} for {name}.");
            }
        }
        while (fields.Count % 16 != 0 || fields.Count == 0) fields.Add(0);
        var job = new { code, buffers = encoded, uniform = Convert.ToBase64String(fields.ToArray()), workgroups = (threads + 255) / 256 };
        _node.StandardInput.WriteLine(JsonSerializer.Serialize(job));
        _node.StandardInput.Flush();
        string? reply = _node.StandardOutput.ReadLine();
        if (reply is null) throw new InvalidOperationException("Dawn server exited: " + _node.StandardError.ReadToEnd());
        using var doc = JsonDocument.Parse(reply);
        if (!doc.RootElement.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException($"{name} on Dawn: {doc.RootElement.GetProperty("error").GetString()}");
        var outputs = doc.RootElement.GetProperty("out");
        for (int b = 0; b < buffers.Count; b++)
        {
            if (!writable.Contains(b)) continue;
            byte[] bytes = Convert.FromBase64String(outputs[b].GetString() ?? "");
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            host.UploadBufferInPlace(data, buffers[b]);
        }
    }

    public void Dispose()
    {
        try { _node.StandardInput.Close(); _node.WaitForExit(10_000); }
        catch (InvalidOperationException) { }
        _node.Dispose();
    }
}
#endif
