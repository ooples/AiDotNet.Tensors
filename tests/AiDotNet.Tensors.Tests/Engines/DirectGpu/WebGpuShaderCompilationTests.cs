#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Every WGSL source in the WebGPU backend, compiled by Dawn (the WebGPU implementation in Chromium) at WebGPU's
/// default limits, and a compute pipeline built for each entry point. Before this, no WGSL here had been compiled
/// anywhere: the first run found 37 shaders Dawn rejects (reserved words, inf / NaN constant expressions, barriers in
/// non-uniform control flow, more than 8 storage buffers). Needs Node and the dawn.node "webgpu" package: point
/// AIDOTNET_WEBGPU_NODE_DIR at a directory whose node_modules holds it (npm install webgpu); skipped otherwise.
/// </summary>
public sealed class WebGpuShaderCompilationTests
{
    // Sources that are fragments, only ever compiled concatenated with the module that defines what they call.
    private static readonly HashSet<string> Fragments = new(StringComparer.Ordinal)
    {
        "WebGpuBatchExecutionKernels.Main",   // the shared main() prologue every batch-execution shader appends to
        "WebGpuKernels.RnnCellSource",        // concatenated after the common header that defines fast_tanh
        "WebGpuTunedRowKernels.RowPrologue",  // the row kernels' shared main() prologue, closed by each family's body
        "WebGpuBatchNormRegisterKernels.Forward",    // templates: SourceFor fills in the name and the values per thread
        "WebGpuBatchNormRegisterKernels.Backward",
        "WebGpuConvDgradS2Kernels.Entry",     // the stride-2 dgrad kernels' shared entry-point header, closed by each body
    };

    private const string Validator = @"
import { create, globals } from 'webgpu';
import fs from 'fs';
import path from 'path';
Object.assign(globalThis, globals);
const dir = process.argv[2];
// Keep the GPU object referenced: dawn.node tears the instance down once it is collected.
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (!adapter) { console.log('NOADAPTER'); process.exit(0); }
const device = await adapter.requestDevice();
for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.wgsl')).sort()) {
  const code = fs.readFileSync(path.join(dir, f), 'utf8');
  device.pushErrorScope('validation');
  const m = device.createShaderModule({ code });
  const info = await m.getCompilationInfo();
  const errs = info.messages.filter(x => x.type === 'error').map(x => `${x.lineNum}:${x.linePos} ${x.message}`);
  // A folded copy (name.fold-entry.wgsl) builds only the entry point it folds.
  const fold = f.match(/\.fold-(\w+)\.wgsl$/);
  const eps = fold ? [fold[1]] : [...code.matchAll(/@compute(?:\s*@\w+(?:\([^)]*\))?)*\s*fn\s+(\w+)/g)].map(x => x[1]);
  for (const ep of (eps.length ? eps : ['main'])) {
    try { device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: ep } }); }
    catch (e) { errs.push(ep + ': ' + e.message.split('\n')[0]); }
  }
  const scope = await device.popErrorScope();
  if (scope) errs.push(scope.message.split('\n')[0]);
  if (errs.length) console.log('FAIL ' + f + ' | ' + errs[0]);
}
console.log('DONE');
process.exit(0);
";

    [SkippableFact]
    public void EveryWgslSource_CompilesUnderDawn_AtDefaultLimits()
    {
        string? nodeDir = Environment.GetEnvironmentVariable("AIDOTNET_WEBGPU_NODE_DIR");
        Skip.If(string.IsNullOrEmpty(nodeDir) || !Directory.Exists(Path.Combine(nodeDir, "node_modules", "webgpu")),
            "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).");
        string node = nodeDir ?? string.Empty;

        string work = Path.Combine(Path.GetTempPath(), "aidotnet-wgsl-" + Guid.NewGuid().ToString("N"));
        string shaders = Path.Combine(work, "shaders");
        Directory.CreateDirectory(shaders);
        try
        {
            int count = 0, folded = 0;
            var unfoldable = new List<string>();
            // Each source as the backend compiles it, and the folded copy of each of its entry points.
            void Emit(string name, string source)
            {
                string prepared = WebGpuShaderPrelude.Prepare(source);
                File.WriteAllText(Path.Combine(shaders, name + ".wgsl"), prepared);
                count++;
                foreach (System.Text.RegularExpressions.Match m in System.Text.RegularExpressions.Regex.Matches(prepared,
                    @"@compute(?:\s*@\w+(?:\([^)]*\))?)*\s*fn\s+(\w+)", System.Text.RegularExpressions.RegexOptions.None, TimeSpan.FromSeconds(5)))
                {
                    string entry = m.Groups[1].Value;
                    if (WebGpuDispatchFold.FoldSource(prepared, entry) is { } fold)
                    {
                        File.WriteAllText(Path.Combine(shaders, name + ".fold-" + entry + ".wgsl"), fold);
                        folded++;
                    }
                    else unfoldable.Add(name + ":" + entry);
                }
            }
            const BindingFlags F = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
            foreach (var type in typeof(WebGpuBatchExecutionKernels).Assembly.GetTypes()
                .Where(t => t.Namespace == typeof(WebGpuBatchExecutionKernels).Namespace))
            {
                foreach (var member in type.GetFields(F).Cast<MemberInfo>().Concat(type.GetProperties(F)))
                {
                    if (Fragments.Contains(type.Name + "." + member.Name)) continue;
                    object? value;
                    try
                    {
                        value = member switch
                        {
                            FieldInfo field => field.GetValue(null),
                            PropertyInfo property when property.GetIndexParameters().Length == 0 => property.GetValue(null),
                            _ => null,
                        };
                    }
                    catch (TargetInvocationException)
                    {
                        continue;   // a runtime getter (JS interop, device state), not a shader source
                    }
                    if (value is not string source || !source.Contains("@compute")) continue;
                    Emit(type.Name + "." + member.Name, source);
                }
            }
            // The register-resident BatchNorm, one module per kernel and per-thread variant (generated, not fields).
            foreach (var name in AiDotNet.Tensors.Engines.DirectGpu.BatchNormRegisterKernelTemplate.GetKernelNames())
                Emit("BatchNormRegister." + name, WebGpuBatchNormRegisterKernels.SourceFor(name));
            // The FP32 PyTorch-parity kernels, one translated module per kernel (#1125, #1128).
            foreach (var module in CudaDialectToWgsl.Translate(AiDotNet.Tensors.Engines.DirectGpu.TorchParityF32Kernels.GetSource()))
            {
                Emit("TorchParityF32." + module.Key, module.Value);
            }
            // The tuned-kernel variants are generated per lane count (methods, not fields), so the reflection above never
            // sees them. Every row variant runs 256 invocations; a column variant runs 32 * R, so only the ones within the
            // default 256 are compiled here (WebGpuBackend offers the wider ones only on devices whose limits admit them).
            foreach (int lanes in WebGpuTunedRowKernels.RowLanes)
            {
                foreach (var (name, source) in new[]
                {
                    ("Softmax", WebGpuTunedRowKernels.Softmax(lanes)), ("SoftmaxBackward", WebGpuTunedRowKernels.SoftmaxBackward(lanes)),
                    ("LayerNorm", WebGpuTunedRowKernels.LayerNorm(lanes)), ("LayerNormBackward", WebGpuTunedRowKernels.LayerNormBackward(lanes)),
                })
                {
                    Emit($"Tuned.{name}.lanes{lanes}", source);
                }
            }
            foreach (int rows in WebGpuTunedRowKernels.ColumnRowLanes)
            {
                if (32 * rows > 256) continue;
                Emit($"Tuned.LayerNormGradParams.rows{rows}", WebGpuTunedRowKernels.LayerNormGradParams(rows));
                Emit($"Tuned.RmsNormGradGamma.rows{rows}", WebGpuTunedRowKernels.RmsNormGradGamma(rows));
            }
            Assert.True(count > 100, $"only {count} WGSL sources found");
            // Every entry point must fold: a dispatch past 65,535 workgroups runs the folded copy (WebGpuDispatchFold).
            Assert.True(unfoldable.Count == 0, $"{unfoldable.Count} entry points cannot be folded: " + string.Join(", ", unfoldable));
            Assert.True(folded > 600, $"only {folded} entry points folded");

            // The script must sit beside node_modules for the 'webgpu' import to resolve.
            string script = Path.Combine(node, "aidotnet-wgsl-validate.mjs");
            File.WriteAllText(script, Validator);
            var start = new ProcessStartInfo("node", $"\"{script}\" \"{shaders}\"")
            {
                WorkingDirectory = node,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using var process = Process.Start(start) ?? throw new InvalidOperationException("node did not start");
            // Both pipes drain concurrently (a full stderr pipe would block node), and the wait is bounded.
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(600_000))
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                Assert.Fail("Dawn shader validation did not finish within 10 minutes.");
            }
            string output = stdout.GetAwaiter().GetResult();
            string errors = stderr.GetAwaiter().GetResult();
            Skip.If(output.Contains("NOADAPTER"), "Dawn found no WebGPU adapter on this machine.");
            Assert.True(output.Contains("DONE"), $"node exited {process.ExitCode} before finishing: {errors}");
            var failures = output.Split('\n').Where(line => line.StartsWith("FAIL ", StringComparison.Ordinal)).ToArray();
            Assert.True(failures.Length == 0, $"{failures.Length} of {count} WGSL sources fail under Dawn:\n" + string.Join("\n", failures));
        }
        finally
        {
            try { Directory.Delete(work, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
#endif
