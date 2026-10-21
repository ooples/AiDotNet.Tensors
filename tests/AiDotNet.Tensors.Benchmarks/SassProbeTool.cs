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
        if (Array.IndexOf(args, "--gemm") >= 0)
        {
            RunGemm(args);
            return;
        }
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
                int registers = 0;
                byte[] Rewrite(byte[] image)
                {
                    if (Array.IndexOf(args, "--save") is int sv && sv >= 0) System.IO.File.WriteAllBytes(args[sv + 1], image);
                    return ApplyPass(image, pass, upto, Array.IndexOf(args, "--dump") >= 0 && dumped++ == 0, out registers, out decline);
                }
                using var ptxModule = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                // maxnreg:N -- the PTX path capped at N registers by ptxas itself: what occupancy alone buys.
                using var sassModule = pass.StartsWith("maxnreg:", StringComparison.Ordinal)
                    ? runtime.LoadModule(ReplaceFirst(ptx, ")\n{\n", ")\n.maxnreg " + pass.Substring(8) + "\n{\n"), allowExperimentalJitFallback: true)
                    : runtime.LoadRewrittenModule(ptx, Rewrite, "sass-control");
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
                    (result.PtxUs / result.SassUs).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" +
                    ("r" + registers + " b" + emitter.LaunchBlockX * emitter.LaunchBlockY).PadLeft(10));
            }
        }
        if (failures.Count != 0) throw new InvalidOperationException("Not bit-identical: " + string.Join(", ", failures));
    }

    // Rewrites every kernel of a cubin under one pass. The default is the scheduler's control words; the others mix
    // them with ptxas's per instruction, to bisect a fault or a mismatch to the rule that causes it.
    private static byte[] ApplyPass(byte[] image, string pass, int upto, bool dump, out int registers, out string? decline)
    {
        var cubin = new SassCubin(image);
        registers = 0;
        decline = null;
        foreach (string kernel in cubin.Kernels)
            registers = Math.Max(registers, cubin.RegisterCount(kernel));
        foreach (string kernel in cubin.Kernels)
        {
            var code = pass == "none" ? cubin.Instructions(kernel) : SassScheduler.RecomputeControl(cubin.Instructions(kernel));
            if (code is null)
            {
                decline = "declined: " + SassScheduler.LastDecline;
                return image;
            }
            var ptxasCode = cubin.Instructions(kernel);
            for (int i = 0; i < code.Length; i++)
            {
                var o = ptxasCode[i].Control; var n = code[i].Control;
                code[i].Control = pass switch
                {
                    "stall" => o with { Stall = n.Stall },                    // our stalls, ptxas scoreboards
                    "sb" => n with { Stall = o.Stall },                       // our scoreboards, ptxas stalls
                    "max" => n with { Stall = Math.Max(n.Stall, o.Stall) },   // ours, never shorter stalls
                    "noreuse" => n with { Reuse = 0 },                        // ours, no operand reuse
                    "ptxasreuse" => n with { Reuse = o.Reuse },               // ours, ptxas's reuse flags
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
            if (dump)
            {
                var original = cubin.Instructions(kernel);
                for (int i = 0; i < original.Length; i++)
                {
                    var o = original[i].Control; var n = code[i].Control;
                    Console.WriteLine(i.ToString("D3") + " " + (SassOpcodeTable.Decode(original[i])?.Mnemonic ?? "?").PadRight(10) +
                        " ptxas s" + o.Stall + " w" + o.WriteScoreboard + " r" + o.ReadScoreboard + " wait" + o.WaitMask.ToString("x2") + " y" + (o.Yield ? 1 : 0) + " u" + o.Reuse +
                        "   ours s" + n.Stall + " w" + n.WriteScoreboard + " r" + n.ReadScoreboard + " wait" + n.WaitMask.ToString("x2") + " u" + n.Reuse);
                }
            }
            cubin.SetInstructions(kernel, code);
        }
        return cubin.ToArray();
    }

    // --gemm: the register-blocked FP32 GEMM (PtxSgemmGenerator), the compute-bound kernel the backend's gate is measured
    // on. Each seed configuration runs a square problem through the PTX path's module and the rewritten one; outputs
    // must be bit-identical, and the times are paired (alternating launches, median of 15).
    private static unsafe void RunGemm(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;
        int size = Array.IndexOf(args, "--size") is int sz && sz >= 0 ? int.Parse(args[sz + 1], CultureInfo.InvariantCulture) : 2048;
        string pass = Array.IndexOf(args, "--pass") is int at && at >= 0 && at + 1 < args.Length ? args[at + 1] : "control";
        int upto = Array.IndexOf(args, "--upto") is int u && u >= 0 ? int.Parse(args[u + 1], CultureInfo.InvariantCulture) : int.MaxValue;
        string? only = Array.IndexOf(args, "--only") is int on && on >= 0 ? args[on + 1] : null;
        var layout = new PtxGemmLayout(false, false);
        Console.WriteLine("SASS GEMM sm_" + major + minor + " M=N=K=" + size + " (nn)");
        Console.WriteLine("configuration                              regs  identical   ptx us   sass us   ratio  ptx TFLOPS");
        var failures = new List<string>();
        var random = new Random(1442);
        var host = new float[(long)size * size];
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            IntPtr Upload()
            {
                for (int i = 0; i < host.Length; i++) host[i] = (float)(random.NextDouble() * 2 - 1);
                var buffer = runtime.AllocateBytes((nuint)(host.Length * sizeof(float)));
                buffer.Upload<float>(host);
                buffers.Add(buffer);
                return buffer.Pointer;
            }
            IntPtr a = Upload(), b = Upload();
            var c0 = runtime.AllocateBytes((nuint)(host.Length * sizeof(float)));
            buffers.Add(c0);
            var c1 = runtime.AllocateBytes((nuint)(host.Length * sizeof(float)));
            buffers.Add(c1);
            foreach (var config in PtxGemmEvolution.Seeds(layout, size, size, size, ccMajor: major))
            {
                string ptx = PtxSgemmGenerator.Emit(config, layout, major, minor);
                if (only is not null && config.Name != only) continue;
                string? decline = null;
                int registers = 0;
                byte[] Rewrite(byte[] image)
                {
                    if (Array.IndexOf(args, "--save") is int sv && sv >= 0) System.IO.File.WriteAllBytes(args[sv + 1], image);
                    return ApplyPass(image, pass, upto, Array.IndexOf(args, "--dump") >= 0, out registers, out decline);
                }
                using var ptxModule = runtime.LoadModule(ptx, allowExperimentalJitFallback: true);
                using var sassModule = runtime.LoadRewrittenModule(ptx, Rewrite, "sass-gemm");
                string name = config.Name.PadRight(42);
                if (decline is not null)
                {
                    Console.WriteLine(name + decline);
                    continue;
                }
                IntPtr ptxFunction = ptxModule.GetFunction(PtxSgemmGenerator.EntryPoint);
                IntPtr sassFunction = sassModule.GetFunction(PtxSgemmGenerator.EntryPoint);
                void Launch(DirectPtxModule module, IntPtr function, IntPtr c)
                {
                    ulong ap = (ulong)a, bp = (ulong)b, cp = (ulong)c;
                    uint dim = (uint)size;
                    float alpha = 1, beta = 0;
                    ulong stride = (ulong)host.Length;
                    void** argv = stackalloc void*[14];
                    argv[0] = &ap; argv[1] = &bp; argv[2] = &cp;
                    argv[3] = &dim; argv[4] = &dim; argv[5] = &dim; argv[6] = &dim; argv[7] = &dim; argv[8] = &dim;
                    argv[9] = &alpha; argv[10] = &beta; argv[11] = &stride; argv[12] = &stride; argv[13] = &stride;
                    module.Launch(function, (uint)((size + config.BlockN - 1) / config.BlockN),
                        (uint)((size + config.BlockM - 1) / config.BlockM), 1, (uint)config.Threads, 1, 1,
                        config.IsAsync ? (uint)config.SharedBytes : 0, argv);
                }
                Launch(ptxModule, ptxFunction, c0.Pointer);
                Launch(sassModule, sassFunction, c1.Pointer);
                runtime.Synchronize();
                var r0 = new float[host.Length];
                var r1 = new float[host.Length];
                c0.Download<float>(r0);
                c1.Download<float>(r1);
                bool identical = true;
                for (int i = 0; i < r0.Length && identical; i++)
                    identical = BitConverter.SingleToInt32Bits(r0[i]) == BitConverter.SingleToInt32Bits(r1[i]);
                if (!identical) failures.Add(config.Name);
                var ptxSamples = new List<double>();
                var sassSamples = new List<double>();
                for (int round = 0; round < 15; round++)
                {
                    ptxSamples.Add(runtime.MeasureKernelMilliseconds(() => Launch(ptxModule, ptxFunction, c0.Pointer), 1, 3) * 1000);
                    sassSamples.Add(runtime.MeasureKernelMilliseconds(() => Launch(sassModule, sassFunction, c1.Pointer), 1, 3) * 1000);
                }
                ptxSamples.Sort();
                sassSamples.Sort();
                double flops = 2.0 * size * size * size;
                Console.WriteLine(name + registers.ToString(CultureInfo.InvariantCulture).PadLeft(5) +
                    (identical ? "yes" : "NO").PadLeft(11) +
                    ptxSamples[7].ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) +
                    sassSamples[7].ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                    (ptxSamples[7] / sassSamples[7]).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" +
                    (flops / (ptxSamples[7] * 1e6)).ToString("F2", CultureInfo.InvariantCulture).PadLeft(12));
            }
        }
        finally
        {
            foreach (var buffer in buffers) buffer.Dispose();
        }
        if (failures.Count != 0) throw new InvalidOperationException("Not bit-identical: " + string.Join(", ", failures));
    }

    private static string ReplaceFirst(string text, string find, string replacement)
    {
        int at = text.IndexOf(find, StringComparison.Ordinal);
        if (at < 0) throw new InvalidOperationException("No '" + find.Trim() + "' in the PTX.");
        return text.Substring(0, at) + replacement + text.Substring(at + find.Length);
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
