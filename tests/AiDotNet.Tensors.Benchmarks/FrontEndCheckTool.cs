// Copyright (c) AiDotNet. All rights reserved.
// FE-1's gate: a graph built by the ORDINARY lowering path, emitted to PTX, executed on
// the device, and compared against the CPU emitter running the SAME graph.
//
// Emitting PTX text proves the translator parses. It does not prove the kernel computes
// the right thing, and it does not prove the pipe is connected end to end. This runs both
// emitters over one graph and requires agreement, which is the only claim worth making
// about a front end.

using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

internal static class FrontEndCheckTool
{
    internal static void Run()
    {
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        {
            Console.WriteLine("The front-end check requires the experimental SM86 device.");
            return;
        }

        Console.WriteLine();
        Console.WriteLine("FRONT-END CHECK - graph -> PTX -> device, against the CPU emitter");
        Console.WriteLine("protocol " + CodegenMeasurementProtocol.Tag);
        Console.WriteLine();
        Console.WriteLine("graph                                  elements   max abs dev   result");

        int passed = 0, failed = 0;
        foreach (var (label, graph) in Graphs())
        {
            try
            {
                bool ok = CheckOne(runtime, label, graph);
                if (ok) passed++; else failed++;
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine(label.PadRight(38) + "        -             -     ERROR");
                Console.WriteLine("    " + ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
            }
        }

        Console.WriteLine();
        Console.WriteLine("front end: " + passed.ToString(CultureInfo.InvariantCulture) + " passed, " +
                          failed.ToString(CultureInfo.InvariantCulture) + " failed");
        Console.WriteLine();
        Console.WriteLine("A pass means a graph the engine's own lowering produces was executed by");
        Console.WriteLine("our generated PTX and agreed with the CPU emitter on the same graph.");
    }

    private static IEnumerable<(string Label, CodegenGraph Graph)> Graphs()
    {
        yield return ("relu (LowerUnaryPointwise)",
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4, 4096 }));

        yield return ("mul (LowerBinaryPointwise)",
            CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Mul, new[] { 8, 2048 }));

        yield return ("mul+add+relu (hand-built chain)", MulAddRelu(16384));
    }

    private static CodegenGraph MulAddRelu(int n)
    {
        var g = new CodegenGraph();
        int a = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { n }));
        int b = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { n }));
        int c = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, new[] { n }));
        int mul = g.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { a, b },
            CodegenElementType.Float32, new[] { n }));
        int add = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { mul, c },
            CodegenElementType.Float32, new[] { n }));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { add },
            CodegenElementType.Float32, new[] { n }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu },
            CodegenElementType.Float32, new[] { n }));
        return g;
    }

    private static bool CheckOne(DirectPtxRuntime runtime, string label, CodegenGraph graph)
    {
        var gpu = new PtxGraphEmitter
        {
            ComputeMajor = runtime.ComputeCapabilityMajor,
            ComputeMinor = runtime.ComputeCapabilityMinor,
        };
        var emitted = gpu.Emit(graph, CodegenElementType.Float32);
        if (emitted.Source is null)
        {
            Console.WriteLine(label.PadRight(38) + "        -             -   DECLINED");
            Console.WriteLine("    " + emitted.DeclineReason);
            return false;
        }

        var spec = gpu.LastSpec!;
        long count = spec.Output.ElementCount;

        // Deterministic inputs, so a failure is reproducible.
        int operands = spec.Inputs.Count;
        var host = new float[operands][];
        for (int i = 0; i < operands; i++)
        {
            host[i] = new float[count];
            for (long e = 0; e < count; e++)
                host[i][e] = (float)((((e * 37 + i * 101) % 97) - 48) / 64.0);
        }

        // --- CPU emitter, the existing reference for this graph.
        var cpuKernel = CodegenDispatcher.TryEmitCpu(graph, CodegenElementType.Float32);
        if (cpuKernel is null)
        {
            Console.WriteLine(label.PadRight(38) + "        -             -   NO CPU REF");
            return false;
        }
        var cpuOut = new float[1][];
        cpuOut[0] = new float[count];
        cpuKernel.Execute<float>(host, cpuOut);

        // --- our PTX, on the device.
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            using var module = runtime.LoadModule(emitted.Source, allowExperimentalJitFallback: true);
            IntPtr fn = module.GetFunction(spec.Name, out _);

            var pointers = new IntPtr[spec.ParameterCount];
            for (int i = 0; i < operands; i++)
            {
                var buffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
                buffer.Upload<float>(host[i]);
                buffers.Add(buffer);
                pointers[i] = buffer.Pointer;
            }
            var outBuffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
            buffers.Add(outBuffer);
            pointers[operands] = outBuffer.Pointer;

            Launch(module, fn, pointers, gpu.LastLaunchBlocks, gpu.LastLaunchBlockX, gpu.LastLaunchBlockY);
            runtime.Synchronize();

            var gpuOut = new float[count];
            outBuffer.Download<float>(gpuOut);

            double worst = 0;
            for (long e = 0; e < count; e++)
                worst = Math.Max(worst, Math.Abs(gpuOut[e] - cpuOut[0][e]));

            bool ok = worst <= 1e-6;
            Console.WriteLine(label.PadRight(38) +
                count.ToString("N0", CultureInfo.InvariantCulture).PadLeft(10) +
                worst.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                (ok ? "PASS" : "FAIL").PadLeft(9));
            return ok;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    private static unsafe void Launch(
        DirectPtxModule module, IntPtr fn, IntPtr[] pointers, uint blocks, uint blockX, uint blockY)
    {
        fixed (IntPtr* pinned = pointers)
        {
            void** argv = stackalloc void*[pointers.Length];
            for (int i = 0; i < pointers.Length; i++) argv[i] = pinned + i;
            module.Launch(fn, blocks, 1, 1, blockX, blockY, 1, 0, argv);
        }
    }
}
