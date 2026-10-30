// Copyright (c) CrowdTrainer LLC. All rights reserved.
// --sgemm-tile: the FP32 GEMM generator (PtxSgemmGenerator) at one square problem, any configurations by name, timed
// paired against a baseline configuration and checked against its output. The FP32 GEMM gap to cuBLAS on sm_86
// (cutlass_80_simt_sgemm_256x128_8x4: 800 us at 2048^3 against our 898) is a tile and pipeline question; this measures
// configurations the evolutionary search does not reach (thread tiles wider than 8).

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class SgemmTileTool
{
    // Name grammar (PtxGemmConfiguration.Name): b{M}x{N}x{K}-t{TM}x{TN}-v{V}-p{P}-{sb|db|as{S}[r]}[-split][-g{G}][-o{O}]
    internal static PtxGemmConfiguration Parse(string name)
    {
        // Strict: a near-miss name must not be timed as some other configuration under the requested name.
        if (!System.Text.RegularExpressions.Regex.IsMatch(name,
                @"^b\d+x\d+x\d+-t\d+x\d+-v\d+-p\d+-(sb|db|as\d+r?)(-split)?(-g\d+)?(-o\d+)?(-fp)?$"))
            throw new FormatException("not a configuration name (b{M}x{N}x{K}-t{TM}x{TN}-v{V}-p{P}-{sb|db|as{S}[r]}[-split][-g{G}][-o{O}][-fp]): " + name);
        var parts = name.Split('-');
        int[] Dims(string s, char prefix) => Array.ConvertAll(s.TrimStart(prefix).Split('x'), x => int.Parse(x, CultureInfo.InvariantCulture));
        var b = Dims(parts[0], 'b');
        var t = Dims(parts[1], 't');
        int v = int.Parse(parts[2].Substring(1), CultureInfo.InvariantCulture);
        int p = int.Parse(parts[3].Substring(1), CultureInfo.InvariantCulture);
        string mode = parts[4];
        bool db = mode == "db";
        int stages = mode.StartsWith("as", StringComparison.Ordinal) ? int.Parse(mode.Substring(2).TrimEnd('r'), CultureInfo.InvariantCulture) : 0;
        bool transpose = mode.EndsWith("r", StringComparison.Ordinal) && stages > 0;
        bool split = false;
        int swizzle = 1, occupancy = 1;
        bool fragments = false;
        for (int i = 5; i < parts.Length; i++)
        {
            if (parts[i] == "split") split = true;
            else if (parts[i][0] == 'g') swizzle = int.Parse(parts[i].Substring(1), CultureInfo.InvariantCulture);
            else if (parts[i] == "fp") fragments = true;
            else if (parts[i][0] == 'o') occupancy = int.Parse(parts[i].Substring(1), CultureInfo.InvariantCulture);
        }
        return new PtxGemmConfiguration(b[0], b[1], b[2], t[0], t[1], v, p, db, split, swizzle, occupancy, stages, transpose, fragments);
    }

    // cuBLAS's SGEMM on the same device buffers, in the runtime's context and stream (column-major: C^T = B^T A^T, so
    // row-major C = A B is cublasSgemm(N, N, n, m, k, B, A)), timed the same way as the generated kernels.
    private static void CuBlas(DirectPtxRuntime runtime, IntPtr a, IntPtr b, IntPtr c, int size)
    {
        using var scope = runtime.Enter();
        if (CuBlasNative.cublasCreate(out IntPtr handle) != CublasStatus.Success) { Console.WriteLine("cuBLAS unavailable"); return; }
        try
        {
            CuBlasNative.cublasSetStream(handle, runtime.Stream);
            void Launch()
            {
                float alpha = 1, beta = 0;
                var status = CuBlasNative.cublasSgemm(handle, CublasOperation.None, CublasOperation.None, size, size, size,
                    ref alpha, b, size, a, size, ref beta, c, size);
                if (status != CublasStatus.Success) throw new InvalidOperationException("cublasSgemm: " + status);
            }
            var samples = new List<double>();
            for (int round = 0; round < 15; round++) samples.Add(runtime.MeasureKernelMilliseconds(Launch, 1, 3) * 1000);
            samples.Sort();
            Console.WriteLine("cuBLAS cublasSgemm".PadRight(45) + "".PadLeft(30) + samples[7].ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                "".PadLeft(9) + (2.0 * size * size * size / (samples[7] * 1e6)).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8));
        }
        finally
        {
            CuBlasNative.cublasDestroy(handle);
        }
    }

    internal static unsafe void Run(string[] args)
    {
        using var runtime = new DirectPtxRuntime();
        int major = runtime.ComputeCapabilityMajor, minor = runtime.ComputeCapabilityMinor;
        string? Option(string flag) => Array.IndexOf(args, flag) is int at && at >= 0
            ? (at + 1 < args.Length ? args[at + 1] : throw new ArgumentException(flag + " needs a value"))
            : null;
        string? sizeText = Option("--size");
        int size = 2048;
        if (sizeText is not null && (!int.TryParse(sizeText, NumberStyles.Integer, CultureInfo.InvariantCulture, out size) || size <= 0))
            throw new ArgumentException("--size needs a positive integer, not " + sizeText);
        string baselineName = Option("--baseline") ?? "b128x128x16-t8x8-v4-p4-as2r-split-g8-o2";
        Parse(baselineName);
        var names = new List<string>();
        for (int i = 0; i < args.Length; i++)
            if (args[i] == "--config" && i + 1 < args.Length) names.AddRange(args[i + 1].Split(','));
        var layout = new PtxGemmLayout(false, false);
        Console.WriteLine("SGEMM TILE sm_" + major + minor + " M=N=K=" + size + " (nn), baseline " + baselineName);
        Console.WriteLine("configuration                                regs  shared   output   base us   this us   ratio  TFLOPS");
        var random = new Random(1442);
        long elements = (long)size * size;
        var host = new float[elements];
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            IntPtr Upload()
            {
                for (long i = 0; i < host.Length; i++) host[i] = (float)(random.NextDouble() * 2 - 1);
                var buffer = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
                buffer.Upload<float>(host);
                buffers.Add(buffer);
                return buffer.Pointer;
            }
            IntPtr a = Upload(), b = Upload();
            var c0 = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            var c1 = runtime.AllocateBytes((nuint)(elements * sizeof(float)));
            buffers.Add(c0);
            buffers.Add(c1);

            (DirectPtxModule Module, IntPtr Function, int Registers) Load(PtxGemmConfiguration config)
            {
                var module = runtime.LoadModule(PtxSgemmGenerator.Emit(config, layout, major, minor), allowExperimentalJitFallback: true);
                IntPtr function = module.GetFunction(PtxSgemmGenerator.EntryPoint, out var info);
                if (config.IsAsync) module.SetMaxDynamicSharedMemory(function, config.SharedBytes);
                return (module, function, info.RegistersPerThread);
            }
            void Launch(PtxGemmConfiguration config, DirectPtxModule module, IntPtr function, IntPtr c)
            {
                ulong ap = (ulong)a, bp = (ulong)b, cp = (ulong)c, stride = (ulong)elements;
                uint dim = (uint)size;
                float alpha = 1, beta = 0;
                void** argv = stackalloc void*[14];
                argv[0] = &ap; argv[1] = &bp; argv[2] = &cp;
                argv[3] = &dim; argv[4] = &dim; argv[5] = &dim; argv[6] = &dim; argv[7] = &dim; argv[8] = &dim;
                argv[9] = &alpha; argv[10] = &beta; argv[11] = &stride; argv[12] = &stride; argv[13] = &stride;
                module.Launch(function, (uint)((size + config.BlockN - 1) / config.BlockN), (uint)((size + config.BlockM - 1) / config.BlockM),
                    1, (uint)config.Threads, 1, 1, config.IsAsync ? (uint)config.SharedBytes : 0, argv);
            }

            if (Array.IndexOf(args, "--cublas") >= 0) CuBlas(runtime, a, b, c1.Pointer, size);
            var baseline = Parse(baselineName);
            var (baseModule, baseFunction, _) = Load(baseline);
            using (baseModule)
            {
                Launch(baseline, baseModule, baseFunction, c0.Pointer);
                runtime.Synchronize();
                var expected = new float[elements];
                c0.Download<float>(expected);
                foreach (string name in names)
                {
                    PtxGemmConfiguration config;
                    try { config = Parse(name); }
                    catch (Exception e) { Console.WriteLine(name.PadRight(45) + "unparsable: " + e.Message); continue; }
                    if (config.Invalidity() is string why) { Console.WriteLine(name.PadRight(45) + "invalid: " + why); continue; }
                    DirectPtxModule module;
                    IntPtr function;
                    int registers;
                    try { (module, function, registers) = Load(config); }
                    catch (Exception e) { Console.WriteLine(name.PadRight(45) + "rejected: " + e.Message.Split('\n')[0]); continue; }
                    using (module)
                    {
                        Launch(config, module, function, c1.Pointer);
                        runtime.Synchronize();
                        var actual = new float[elements];
                        c1.Download<float>(actual);
                        // Bit-identical, else within the generator tests' oracle bound relative to the K-term sum
                        // (a different reduction order rounds differently; an output near zero is still bounded by K).
                        long differ = 0;
                        double worst = 0;
                        for (long i = 0; i < elements; i++)
                        {
                            if (BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i])) continue;
                            differ++;
                            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / size);
                        }
                        string output = differ == 0 ? "same" : worst < 2e-5 ? "~" + worst.ToString("E0", CultureInfo.InvariantCulture) : "WRONG";
                        var baseSamples = new List<double>();
                        var samples = new List<double>();
                        for (int round = 0; round < 15; round++)
                        {
                            baseSamples.Add(runtime.MeasureKernelMilliseconds(() => Launch(baseline, baseModule, baseFunction, c0.Pointer), 1, 3) * 1000);
                            samples.Add(runtime.MeasureKernelMilliseconds(() => Launch(config, module, function, c1.Pointer), 1, 3) * 1000);
                        }
                        baseSamples.Sort();
                        samples.Sort();
                        Console.WriteLine(name.PadRight(45) + registers.ToString(CultureInfo.InvariantCulture).PadLeft(4) +
                            (config.SharedBytes / 1024.0).ToString("F0", CultureInfo.InvariantCulture).PadLeft(6) + "K" +
                            output.PadLeft(9) + baseSamples[7].ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                            samples[7].ToString("F1", CultureInfo.InvariantCulture).PadLeft(10) +
                            (baseSamples[7] / samples[7]).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8) + "x" +
                            (2.0 * size * size * size / (samples[7] * 1e6)).ToString("F2", CultureInfo.InvariantCulture).PadLeft(8));
                    }
                }
            }
        }
        finally
        {
            foreach (var buffer in buffers) buffer.Dispose();
        }
    }
}