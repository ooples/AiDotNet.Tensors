#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Runs the library's own <c>webgpu-tensors.js</c> in Node, with Dawn (the <c>webgpu</c> npm package) as
/// <c>navigator.gpu</c>, and serves <see cref="WebGpuNativeBindings"/>'s calls from it. With this installed as
/// <see cref="WebGpuNativeBindings.Host"/> the real <see cref="WebGpuBackend"/> executes on this machine: its
/// dispatch code, its buffer management and the shipped JS glue, on a real WebGPU implementation. Only the
/// C#-to-JS marshalling differs from a browser (a pipe carrying JSON here, the WebAssembly interop there).
/// <para>
/// Needs AIDOTNET_WEBGPU_NODE_DIR: a directory whose node_modules holds "webgpu" (npm install webgpu).
/// </para>
/// <para>
/// The module's <c>console.error</c> calls are how it reports a missing buffer or an uncaptured device error. They
/// are collected, and the next call throws with them, so a validation error fails the test that caused it (or the
/// one after: uncaptured errors arrive asynchronously).
/// </para>
/// </summary>
internal sealed class DawnWebGpuHost : IWebGpuJsHost, IDisposable
{
    private const string Server = @"
import { create, globals } from 'webgpu';
import readline from 'readline';
Object.assign(globalThis, globals);
Object.defineProperty(globalThis, 'navigator', { value: { gpu: create([]) }, configurable: true, writable: true });
const errors = [];
console.log = (...a) => process.stderr.write(a.map(String).join(' ') + '\n');
console.error = (...a) => { errors.push(a.map(x => (x && x.message) || String(x)).join(' ')); };
const mod = await import(process.argv[2]);
const reply = o => process.stdout.write(JSON.stringify(o) + '\n');
reply({ ready: true });
const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
  let job;
  try {
    job = JSON.parse(line);
    const fn = mod[job.f];
    if (typeof fn !== 'function') throw new Error(`webgpu-tensors.js has no export named '${job.f}'`);
    const r = await fn(...job.a);
    reply({ ok: true, r: r === undefined ? null : r, errors: errors.splice(0) });
  } catch (e) {
    reply({ ok: false, error: String((e && e.stack) || e), errors: errors.splice(0) });
  }
}
";

    // A call longer than this is a hung device or a stuck mapAsync, not a slow one. Once one call has timed out or the
    // device has been lost, the host is dead: every later call fails at once with the original reason, so one broken
    // test costs one timeout and the rest of the run reports what actually happened.
    private static readonly TimeSpan ReplyTimeout = TimeSpan.FromSeconds(90);

    private readonly Process _node;
    private readonly StringBuilder _stderr = new();
    private readonly string[] _temporaryFiles;
    private readonly object _gate = new();
    private bool _disposed;
    private string? _dead;

    /// <summary>Why the host can no longer serve calls (a timeout, a lost device, an exited process), or null.</summary>
    public string? Dead { get { lock (_gate) return _dead; } }

    /// <summary>Calls served, by export name. For tests that assert a path really went through the host.</summary>
    public Dictionary<string, int> Calls { get; } = new(StringComparer.Ordinal);

    private DawnWebGpuHost(Process node, string[] temporaryFiles)
    {
        _node = node;
        _temporaryFiles = temporaryFiles;
        node.ErrorDataReceived += (_, e) => { if (e.Data is not null) lock (_stderr) _stderr.AppendLine(e.Data); };
        node.BeginErrorReadLine();
    }

    /// <summary>The module the library ships, found by walking up from the test binaries to the repository.</summary>
    internal static string? FindShippedModule()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "WebGpu", "wwwroot", "js", "webgpu-tensors.js");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>A started host, or null when AIDOTNET_WEBGPU_NODE_DIR is not set up or the module cannot be found.</summary>
    public static DawnWebGpuHost? TryCreate()
    {
        string? dir = Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_NODE_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(Path.Combine(dir, "node_modules", "webgpu"))) return null;
        string? shipped = FindShippedModule();
        if (shipped is null) return null;

        // The module is an ES module in a .js file; Node only treats it as one under an .mjs name. It is copied
        // unchanged next to the server script, which is where 'webgpu' resolves from.
        string stamp = Guid.NewGuid().ToString("N");
        string script = Path.Combine(dir, $"aidotnet-webgpu-host-{stamp}.mjs");
        string module = Path.Combine(dir, $"aidotnet-webgpu-tensors-{stamp}.mjs");
        File.WriteAllText(script, Server);
        File.Copy(shipped, module, overwrite: true);

        var start = new ProcessStartInfo("node", $"\"{script}\" \"{new Uri(module).AbsoluteUri}\"")
        {
            WorkingDirectory = dir,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        var node = Process.Start(start);
        if (node is null)
        {
            File.Delete(script);
            File.Delete(module);
            return null;
        }
        var host = new DawnWebGpuHost(node, new[] { script, module });
        try
        {
            using var ready = JsonDocument.Parse(host.ReadReply("startup"));
            if (!ready.RootElement.TryGetProperty("ready", out _))
                throw new InvalidOperationException("The WebGPU host did not report ready: " + ready.RootElement.GetRawText());
        }
        catch
        {
            host.Dispose();
            throw;
        }
        return host;
    }

    private string ReadReply(string what)
    {
        Task<string?> read = _node.StandardOutput.ReadLineAsync();
        if (!read.Wait(ReplyTimeout))
        {
            try { _node.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            _dead = $"The WebGPU host did not answer '{what}' within {ReplyTimeout.TotalSeconds} seconds and was stopped.";
            throw new TimeoutException(_dead);
        }
        string? line = read.Result;
        if (line is null)
        {
            string stderr;
            lock (_stderr) stderr = _stderr.ToString();
            _dead = $"The WebGPU host exited during '{what}'. stderr: {stderr}";
            throw new InvalidOperationException(_dead);
        }
        return line;
    }

    public string Invoke(string function, string argumentsJson)
    {
        lock (_gate)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(DawnWebGpuHost));
            if (_dead is not null) throw new InvalidOperationException($"The WebGPU host is no longer usable ({function} not sent): {_dead}");
            Calls[function] = Calls.TryGetValue(function, out int n) ? n + 1 : 1;
            _node.StandardInput.Write("{\"f\":");
            _node.StandardInput.Write(JsonSerializer.Serialize(function));
            _node.StandardInput.Write(",\"a\":");
            _node.StandardInput.Write(argumentsJson);
            _node.StandardInput.WriteLine("}");
            _node.StandardInput.Flush();

            using var reply = JsonDocument.Parse(ReadReply(function));
            var root = reply.RootElement;
            string reported = "";
            if (root.TryGetProperty("errors", out var errors) && errors.GetArrayLength() > 0)
            {
                var text = new List<string>();
                foreach (var e in errors.EnumerateArray()) text.Add(e.GetString() ?? "");
                reported = string.Join(Environment.NewLine, text);
                // The module nulls its device when it is lost; nothing after that can work.
                if (reported.IndexOf("device lost", StringComparison.OrdinalIgnoreCase) >= 0)
                    _dead = $"The WebGPU device was lost (seen at {function}): {reported}";
            }
            if (!root.GetProperty("ok").GetBoolean())
                throw new InvalidOperationException($"webgpu-tensors.js {function} failed: {root.GetProperty("error").GetString()}"
                    + (reported.Length > 0 ? Environment.NewLine + reported : ""));
            if (reported.Length > 0)
                throw new InvalidOperationException($"webgpu-tensors.js reported an error (seen at {function}): {reported}");
            return root.GetProperty("r").GetRawText();
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
        }
        try
        {
            _node.StandardInput.Close();
            if (!_node.WaitForExit(5000)) _node.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException) { }
        catch (IOException) { }
        _node.Dispose();
        foreach (string file in _temporaryFiles)
        {
            try { File.Delete(file); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
#endif
