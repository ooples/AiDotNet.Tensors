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
    if (job.compileOnly) { console.log(JSON.stringify({ ok: true, out: [] })); continue; }
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

    // A launch longer than this is a hung shader or a stuck mapAsync, not a slow one.
    private static readonly TimeSpan ReplyTimeout = TimeSpan.FromMinutes(5);

    private readonly Process _node;
    private readonly IReadOnlyDictionary<string, string>? _modules;
    private readonly System.Text.StringBuilder _stderr;

    private DawnWgslLauncher(Process node, IReadOnlyDictionary<string, string>? modules, System.Text.StringBuilder stderr)
    {
        _node = node;
        _modules = modules;
        _stderr = stderr;
    }

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
        if (node is null) return null;
        // Drain stderr as it arrives: a full stderr pipe would block node, and then the reply read below.
        var stderr = new System.Text.StringBuilder();
        node.ErrorDataReceived += (_, e) => { if (e.Data is not null) lock (stderr) stderr.AppendLine(e.Data); };
        node.BeginErrorReadLine();
        return new DawnWgslLauncher(node, modules, stderr);
    }

    /// <summary>A <see cref="PortableGenericFft.Launcher"/> that runs each launch's WGSL port on Dawn.</summary>
    public PortableGenericFft.Launcher For(IDirectGpuBackend host) => (name, threads, args) => Launch(host, name, threads, args);

    /// <summary>
    /// Builds <paramref name="name"/>'s compute pipeline on Dawn without dispatching it: null when it compiles, else
    /// Dawn's error (the way a backend's readiness check asks its device compiler).
    /// </summary>
    public string? Compile(string name)
    {
        string? source = null;
        if (!(_modules?.TryGetValue(name, out source) ?? false) && !WebGpuBatchExecutionKernels.ByKernelName.TryGetValue(name, out source))
            return $"No WGSL port for {name}.";
        var job = new { code = WebGpuShaderPrelude.Prepare(source ?? ""), compileOnly = true };
        _node.StandardInput.WriteLine(JsonSerializer.Serialize(job));
        _node.StandardInput.Flush();
        var read = _node.StandardOutput.ReadLineAsync();
        if (!read.Wait(ReplyTimeout)) return $"{name}: Dawn did not reply within {ReplyTimeout.TotalMinutes} minutes.";
        string? reply = read.Result;
        if (reply is null) return "Dawn server exited: " + StderrText();
        using var doc = JsonDocument.Parse(reply);
        return doc.RootElement.GetProperty("ok").GetBoolean() ? null : doc.RootElement.GetProperty("error").GetString() ?? "compile failed";
    }

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
                    byte[] bytes;
                    if (IsByteBuffer(buffer))
                    {
                        // A byte buffer (FP16 or int8 operands): its Size is a byte count, and the host reads it
                        // through its own call. The shader binds it as 32-bit words, so the copy is padded to a
                        // whole word; the kernels mask every element out of its word and never use the padding.
                        byte[] raw = host.DownloadByteBuffer(buffer, checked((int)buffer.SizeInBytes));
                        bytes = new byte[(raw.Length + 3) / 4 * 4];
                        Buffer.BlockCopy(raw, 0, bytes, 0, raw.Length);
                    }
                    else
                    {
                        float[] data = host.DownloadBuffer(buffer);
                        bytes = new byte[data.Length * 4];
                        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                    }
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
        var read = _node.StandardOutput.ReadLineAsync();
        if (!read.Wait(ReplyTimeout))
        {
            try { _node.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            throw new TimeoutException($"{name} on Dawn did not reply within {ReplyTimeout.TotalMinutes} minutes: {StderrText()}");
        }
        string? reply = read.Result;
        if (reply is null) throw new InvalidOperationException("Dawn server exited: " + StderrText());
        using var doc = JsonDocument.Parse(reply);
        if (!doc.RootElement.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException($"{name} on Dawn: {doc.RootElement.GetProperty("error").GetString()}");
        var outputs = doc.RootElement.GetProperty("out");
        for (int b = 0; b < buffers.Count; b++)
        {
            if (!writable.Contains(b)) continue;
            byte[] bytes = Convert.FromBase64String(outputs[b].GetString() ?? "");
            if (IsByteBuffer(buffers[b]))
            {
                // The shader wrote whole 32-bit words; the buffer holds only SizeInBytes of them.
                var exact = new byte[checked((int)buffers[b].SizeInBytes)];
                Buffer.BlockCopy(bytes, 0, exact, 0, Math.Min(exact.Length, bytes.Length));
                host.UploadByteBuffer(buffers[b], exact);
                continue;
            }
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            host.UploadBufferInPlace(data, buffers[b]);
        }
    }

    private string StderrText()
    {
        lock (_stderr) return _stderr.ToString();
    }

    // A byte buffer (AllocateByteBuffer) counts its Size in bytes; a float buffer's SizeInBytes is four times its Size.
    private static bool IsByteBuffer(IGpuBuffer buffer) => buffer.SizeInBytes != (long)buffer.Size * sizeof(float);

    public void Dispose()
    {
        try { _node.StandardInput.Close(); _node.WaitForExit(10_000); }
        catch (InvalidOperationException) { }
        _node.Dispose();
    }
}
#endif
