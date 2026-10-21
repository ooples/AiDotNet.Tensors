// Copyright (c) AiDotNet. All rights reserved.
// --sass-probe: the own SASS backend's passes on the device (#1442). For every catalog kernel the affine emitter
// lowers (data-dependent indexing excepted), the PTX path's module and the module rewritten by the backend run on
// the same inputs; their outputs must be bit-identical (the passes keep every instruction's arithmetic), and their
// times are compared (median of 15 timed launches each, alternating).
//
// Options: --pass control (SassScheduler.RecomputeControl, the default).

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Sass;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class SassProbeTool
{
    internal static void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        string pass = Array.IndexOf(args, "--pass") is int at && at >= 0 && at + 1 < args.Length ? args[at + 1] : "control";
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;
        Console.WriteLine("SASS PROBE sm_" + major + minor + " - PTX path vs rewritten cubin (control words recomputed)");
        Console.WriteLine("kernel                                   size     identical   ptx us   sass us   ratio");
        var failures = new List<string>();
        int dumped = 0;
        int upto = Array.IndexOf(args, "--upto") is int u && u >= 0 ? int.Parse(args[u + 1], CultureInfo.InvariantCulture) : int.MaxValue;
        string? only = Array.IndexOf(args, "--only") is int on && on >= 0 ? args[on + 1] : null;
        string onlyTag = Array.IndexOf(args, "--tag") is int tg && tg >= 0 ? args[tg + 1] : "verify";
        foreach (var entry in CodegenKernelCatalog.All)
        {
            if (only is not null && entry.Name != only) continue;
            foreach (var (tag, spec) in new[] { ("verify", entry.Verify), ("bench", entry.Bench) })
            {
                if (only is not null && tag != onlyTag) continue;
                if (spec.Output.HasIndirection || spec.Output.ElementType != CodegenElementType.Float32 ||
                    System.Linq.Enumerable.Any(spec.Inputs, b => b.HasIndirection || b.ElementType != CodegenElementType.Float32)) continue;
                var emitter = new PtxAffineEmitter();
                string ptx;
                try
                {
                    ptx = emitter.Emit(spec, major, minor);
                }
                catch (NotSupportedException)
                {
                    continue;
                }
                string name = (entry.Name + " " + tag).PadRight(40);
                string? decline = null;
                byte[] Rewrite(byte[] image)
                {
                    var cubin = new SassCubin(image);
                    if (Array.IndexOf(args, "--save") is int sv && sv >= 0) System.IO.File.WriteAllBytes(args[sv + 1], image);
                    foreach (string kernel in cubin.Kernels)
                    {
                        var code = pass == "none" ? cubin.Instructions(kernel) : SassScheduler.RecomputeControl(cubin.Instructions(kernel));
                        if (code is null) { decline = "declined: " + SassScheduler.LastDecline; return image; }
                        var ptxasCode = cubin.Instructions(kernel);
                        for (int i = 0; i < code.Length; i++)
                        {
                            var o = ptxasCode[i].Control; var n = code[i].Control;
                            code[i].Control = pass switch
                            {
                                "stall" => o with { Stall = n.Stall },                    // our stalls, ptxas scoreboards
                                "sb" => n with { Stall = o.Stall },                       // our scoreboards, ptxas stalls
                                "max" => n with { Stall = Math.Max(n.Stall, o.Stall) },   // ours, never shorter stalls
                                "plus" => o with { Stall = Math.Min(15, o.Stall + 4), Reuse = 0 },
                                "plusalu" => SassOpcodeTable.Decode(ptxasCode[i])?.Latency == SassLatency.Fixed ? o with { Stall = Math.Min(15, o.Stall + 4) } : o,
                                "plusvar" => i < upto && SassOpcodeTable.Decode(ptxasCode[i])?.Latency == SassLatency.Variable ? o with { Stall = Math.Min(15, o.Stall + 4) } : o,
                                "pluscontrol" => SassOpcodeTable.Decode(ptxasCode[i])?.Latency == SassLatency.Control ? o with { Stall = Math.Min(15, o.Stall + 4) } : o,
                                "plus1" => o with { Stall = Math.Min(15, o.Stall + 1) },
                                _ when pass.StartsWith("var", StringComparison.Ordinal) => SassOpcodeTable.Decode(ptxasCode[i]) is { Latency: SassLatency.Variable } ve && (pass.Length < 5 || ve.Mnemonic == pass.Substring(5)) ? o with { Stall = int.Parse(pass.Substring(3, 2), CultureInfo.InvariantCulture) } : o,
                                _ when pass.StartsWith("keep:", StringComparison.Ordinal) => SassOpcodeTable.Decode(ptxasCode[i])?.Mnemonic == pass.Substring(5) ? n with { Stall = Math.Max(n.Stall, o.Stall) } : n,
                                "keepcontrol" => SassOpcodeTable.Decode(ptxasCode[i])?.Latency == SassLatency.Control ? n with { Stall = Math.Max(n.Stall, o.Stall) } : n,
                                "upto" => i < upto ? n : n with { Stall = Math.Max(n.Stall, o.Stall) },
                                _ => n,
                            };
                        }
                        if (Array.IndexOf(args, "--dump") >= 0 && dumped++ == 0)
                        {
                            var original = cubin.Instructions(kernel);
                            for (int i = 0; i < original.Length; i++)
                            {
                                var o = original[i].Control; var n = code[i].Control;
                                Console.WriteLine(i.ToString("D3") + " " + (SassOpcodeTable.Decode(original[i])?.Mnemonic ?? "?").PadRight(10) +
                                    " ptxas s" + o.Stall + " w" + o.WriteScoreboard + " r" + o.ReadScoreboard + " wait" + o.WaitMask.ToString("x2") + " y" + (o.Yield ? 1 : 0) +
                                    "   ours s" + n.Stall + " w" + n.WriteScoreboard + " r" + n.ReadScoreboard + " wait" + n.WaitMask.ToString("x2"));
                            }
                        }
                        cubin.SetInstructions(kernel, code);
                    }
                    return cubin.ToArray();
                }
                using var ptxModule = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                using var sassModule = runtime.LoadRewrittenModule(ptx, Rewrite, "sass-control");
                if (decline is not null)
                {
                    Console.WriteLine(name + " " + decline);
                    continue;
                }
                var result = Compare(runtime, spec, emitter, ptxModule, sassModule);
                if (!result.Identical) failures.Add(entry.Name + " " + tag);
                Console.WriteLine(name + result.Size.ToString(CultureInfo.InvariantCulture).PadLeft(9) +
                    (result.Identical ? "yes" : "NO").PadLeft(12) +
                    result.PtxUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                    result.SassUs.ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                    (result.PtxUs / result.SassUs).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x");
            }
        }
        if (failures.Count != 0) throw new InvalidOperationException("Not bit-identical: " + string.Join(", ", failures));
    }

    private static unsafe (bool Identical, long Size, double PtxUs, double SassUs) Compare(DirectPtxRuntime runtime,
        CodegenKernelSpec spec, PtxAffineEmitter emitter, DirectPtxModule ptxModule, DirectPtxModule sassModule)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            var pointers = new IntPtr[spec.ParameterCount];
            long outputBytes = spec.Output.ElementCount * spec.Output.ElementBytes;
            var random = new Random(1442);
            foreach (var binding in spec.Inputs)
            {
                var data = new float[binding.ElementCount];
                for (int i = 0; i < data.Length; i++) data[i] = (float)(random.NextDouble() * 2 - 1);
                var buffer = runtime.AllocateBytes((nuint)(data.Length * sizeof(float)));
                buffer.Upload<float>(data);
                buffers.Add(buffer);
                pointers[binding.ParameterIndex] = buffer.Pointer;
            }
            var outputs = new DirectPtxBuffer[2];
            for (int k = 0; k < 2; k++)
            {
                outputs[k] = runtime.AllocateBytes((nuint)outputBytes);
                buffers.Add(outputs[k]);
            }
            IntPtr ptxFunction = ptxModule.GetFunction(spec.Name), sassFunction = sassModule.GetFunction(spec.Name);
            void Launch(DirectPtxModule module, IntPtr function, IntPtr output)
            {
                var values = (IntPtr[])pointers.Clone();
                values[spec.Output.ParameterIndex] = output;
                IntPtr* storage = stackalloc IntPtr[values.Length];
                void** argv = stackalloc void*[values.Length];
                for (int a = 0; a < values.Length; a++) { storage[a] = values[a]; argv[a] = &storage[a]; }
                module.Launch(function, emitter.LaunchBlocks, 1, 1, (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY,
                    1, 0, argv);
            }
            Launch(ptxModule, ptxFunction, outputs[0].Pointer);
            Launch(sassModule, sassFunction, outputs[1].Pointer);
            runtime.Synchronize();
            var a0 = new float[outputBytes / sizeof(float)];
            var a1 = new float[outputBytes / sizeof(float)];
            outputs[0].Download<float>(a0);
            outputs[1].Download<float>(a1);
            bool identical = true;
            for (int i = 0; i < a0.Length && identical; i++)
                identical = BitConverter.SingleToInt32Bits(a0[i]) == BitConverter.SingleToInt32Bits(a1[i]);

            var ptxSamples = new List<double>();
            var sassSamples = new List<double>();
            for (int round = 0; round < 15; round++)
            {
                ptxSamples.Add(runtime.MeasureKernelMilliseconds(() => Launch(ptxModule, ptxFunction, outputs[0].Pointer), 1, 5) * 1000);
                sassSamples.Add(runtime.MeasureKernelMilliseconds(() => Launch(sassModule, sassFunction, outputs[1].Pointer), 1, 5) * 1000);
            }
            ptxSamples.Sort();
            sassSamples.Sort();
            return (identical, spec.Output.ElementCount, ptxSamples[7], sassSamples[7]);
        }
        finally
        {
            foreach (var buffer in buffers) buffer.Dispose();
        }
    }
}
