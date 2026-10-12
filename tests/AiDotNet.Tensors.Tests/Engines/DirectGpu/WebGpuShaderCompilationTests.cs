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
  const eps = [...code.matchAll(/@compute(?:\s*@\w+(?:\([^)]*\))?)*\s*fn\s+(\w+)/g)].map(x => x[1]);
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

        string work = Path.Combine(Path.GetTempPath(), "aidotnet-wgsl-" + Guid.NewGuid().ToString("N"));
        string shaders = Path.Combine(work, "shaders");
        Directory.CreateDirectory(shaders);
        try
        {
            int count = 0;
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
                    File.WriteAllText(Path.Combine(shaders, type.Name + "." + member.Name + ".wgsl"), WebGpuShaderPrelude.Prepare(source));
                    count++;
                }
            }
            Assert.True(count > 100, $"only {count} WGSL sources found");

            // The script must sit beside node_modules for the 'webgpu' import to resolve.
            string script = Path.Combine(nodeDir!, "aidotnet-wgsl-validate.mjs");
            File.WriteAllText(script, Validator);
            var start = new ProcessStartInfo("node", $"\"{script}\" \"{shaders}\"")
            {
                WorkingDirectory = nodeDir!,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            using var process = Process.Start(start) ?? throw new InvalidOperationException("node did not start");
            string output = process.StandardOutput.ReadToEnd();
            string errors = process.StandardError.ReadToEnd();
            process.WaitForExit(600_000);
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
