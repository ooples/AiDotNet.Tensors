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
        Console.WriteLine("graph                                  elements       rel dev   ref   result");

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
        Console.WriteLine("A pass means a graph the engine's own lowering produces was executed by our");
        Console.WriteLine("generated PTX and agreed with a reference. The 'ref' column says which:");
        Console.WriteLine();
        Console.WriteLine("  cpu   the CPU emitter on the SAME graph. Shares nothing with the PTX");
        Console.WriteLine("        translator, so it checks the translation and the emission together.");
        Console.WriteLine("  fp64  the translated spec's own fp64 interpretation. Checks the EMITTER");
        Console.WriteLine("        against the spec, and NOT the spec against the graph -- a translator");
        Console.WriteLine("        that swapped two axes would pass this. The reduction forms are");
        Console.WriteLine("        checked against independent hand-written contractions in");
        Console.WriteLine("        CodegenGraphReductionTests instead; that is what covers the gap.");
    }

    private static IEnumerable<(string Label, CodegenGraph Graph)> Graphs()
    {
        yield return ("relu (LowerUnaryPointwise)",
            CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4, 4096 }));

        yield return ("mul (LowerBinaryPointwise)",
            CodegenLowering.LowerBinaryPointwise<float>(CodegenOpKind.Mul, new[] { 8, 2048 }));

        yield return ("mul+add+relu (hand-built chain)", MulAddRelu(16384));

        // Reductions: these all DECLINED before the front end learned index maps, which
        // meant no matmul -- and so no linear layer -- could reach the PTX path at all.
        yield return ("matmul 128x96x64", MatMul(CodegenOpKind.MatMul, 128, 96, 64));
        yield return ("matmul A-transposed 128x96x64", MatMul(CodegenOpKind.MatMulTransposeA, 128, 96, 64));
        yield return ("matmul B-transposed 128x96x64", MatMul(CodegenOpKind.MatMulTransposeB, 128, 96, 64));
        yield return ("linear: matmul+bias+relu 256x128x64", Linear(256, 128, 64));
        yield return ("reduce-sum [512,256] over axis 1", Reduce(CodegenOpKind.ReduceSum, 512, 256));
        yield return ("reduce-max [512,256] over axis 1", Reduce(CodegenOpKind.ReduceMax, 512, 256));
    }

    private static int Load(CodegenGraph g, int[] shape) =>
        g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(),
            CodegenElementType.Float32, shape));

    private static CodegenGraph MatMul(CodegenOpKind op, int m, int k, int n)
    {
        var g = new CodegenGraph();
        int a = Load(g, op == CodegenOpKind.MatMulTransposeA ? new[] { k, m } : new[] { m, k });
        int b = Load(g, op == CodegenOpKind.MatMulTransposeB ? new[] { n, k } : new[] { k, n });
        int mm = g.AddNode(new CodegenNode(op, new[] { a, b },
            CodegenElementType.Float32, new[] { m, n }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mm },
            CodegenElementType.Float32, new[] { m, n }));
        return g;
    }

    private static CodegenGraph Linear(int m, int k, int n)
    {
        var g = new CodegenGraph();
        int a = Load(g, new[] { m, k });
        int w = Load(g, new[] { k, n });
        int bias = Load(g, new[] { n });
        int mm = g.AddNode(new CodegenNode(CodegenOpKind.MatMul, new[] { a, w },
            CodegenElementType.Float32, new[] { m, n }));
        int add = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { mm, bias },
            CodegenElementType.Float32, new[] { m, n }));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { add },
            CodegenElementType.Float32, new[] { m, n }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu },
            CodegenElementType.Float32, new[] { m, n }));
        return g;
    }

    private static CodegenGraph Reduce(CodegenOpKind op, int rows, int cols)
    {
        var g = new CodegenGraph();
        int x = Load(g, new[] { rows, cols });
        int r = g.AddNode(new CodegenNode(op, new[] { x },
            CodegenElementType.Float32, new[] { rows }, new[] { 1 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { r },
            CodegenElementType.Float32, new[] { rows }));
        return g;
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

        // Operands are sized from their OWN bindings, not from the output. They coincide
        // only for pointwise kernels; a matmul's A, B and bias are three different sizes.
        int operands = spec.Inputs.Count;
        var host = new float[operands][];
        var wide = new double[operands][];
        for (int i = 0; i < operands; i++)
        {
            long size = spec.Inputs[i].ElementCount;
            host[i] = new float[size];
            wide[i] = new double[size];
            for (long e = 0; e < size; e++)
            {
                double v = (((e * 37 + i * 101) % 97) - 48) / 64.0;
                host[i][e] = (float)v;
                wide[i][e] = v;
            }
        }

        // --- The reference. The CPU emitter running the SAME graph is the stronger one,
        // because it shares nothing with the translator; when it cannot take the graph,
        // fall back to the spec's own fp64 interpretation and SAY SO, since that only
        // checks the emitter against the spec, not the spec against the graph.
        string reference;
        double[] want;
        var cpuKernel = CodegenDispatcher.TryEmitCpu(
            graph, CodegenElementType.Float32, out var cpuDeclines);
        if (cpuKernel is not null)
        {
            var cpuOut = new float[1][];
            cpuOut[0] = new float[count];
            cpuKernel.Execute<float>(host, cpuOut);
            want = new double[count];
            for (long e = 0; e < count; e++) want[e] = cpuOut[0][e];
            reference = "cpu";
        }
        else
        {
            want = spec.Interpret(wide);
            reference = "fp64";
        }

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
                var buffer = runtime.AllocateBytes((nuint)(host[i].Length * sizeof(float)));
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

            double worst = 0, scale = 0;
            for (long e = 0; e < count; e++)
            {
                worst = Math.Max(worst, Math.Abs(gpuOut[e] - want[e]));
                scale = Math.Max(scale, Math.Abs(want[e]));
            }

            // A reduction accumulates, so the tolerance has to be relative to the result's
            // own magnitude; an absolute 1e-6 would be a fp32 epsilon test, not a
            // correctness test. Pointwise graphs still land on exact zero.
            double deviation = scale > 0 ? worst / scale : worst;
            bool ok = deviation <= 1e-6;
            Console.WriteLine(label.PadRight(38) +
                count.ToString("N0", CultureInfo.InvariantCulture).PadLeft(10) +
                deviation.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                "  " + reference.PadRight(5) +
                (ok ? "PASS" : "FAIL").PadLeft(7));

            // Say why the stronger reference was unavailable, rather than letting a
            // weaker check pass as though it were the same one.
            if (reference == "fp64")
                foreach (var (target, why) in cpuDeclines)
                    Console.WriteLine("    no CPU reference from " + target + ": " + why.Split('\n')[0]);

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
