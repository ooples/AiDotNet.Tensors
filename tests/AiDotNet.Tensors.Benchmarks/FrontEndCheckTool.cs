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
    /// <summary>
    /// Whether timings may be reported. Correctness does not care what else is on the
    /// GPU; a ratio does. Gating the whole check on an idle device would mean a busy box
    /// could not verify anything, and reporting microseconds taken against a foreign
    /// workload is worse than reporting none -- an earlier run of this tool produced a
    /// 64 us ReLU and a 466 us 512-element reduction while another process held 84% of
    /// the SMs, and those numbers meant nothing.
    /// </summary>
    private static bool _timingAllowed;

    /// <summary>SM clock when timing began, so drift across the run can be reported.</summary>
    private static int _clockAtStart;

    internal static void Run() => Run(Array.Empty<string>());

    internal static void Run(string[] args)
    {
        // --force-timing exists for ONE situation: a compute process that is holding a
        // context and its memory but has been externally SUSPENDED, so it occupies no
        // SMs. The guard keys on the process, not on whether it is running, and neither
        // a frozen process's context nor its allocation affects how long our kernels
        // take. It prints what it overrode, so the caveat travels with the numbers
        // instead of being lost between a terminal and a document.
        bool force = Array.IndexOf(args, "--force-timing") >= 0;

        try
        {
            GpuBenchmarkEnvironment.RequireIdleGpu("frontend-check");
            _timingAllowed = true;
        }
        catch (InvalidOperationException ex)
        {
            _timingAllowed = force;
            Console.WriteLine();
            Console.WriteLine((force ? "GUARD OVERRIDDEN - " : "TIMINGS SUPPRESSED - ") +
                              ex.Message.Split('\n')[0]);
            Console.WriteLine(force
                ? "Timings reported anyway on the caller's assertion that the above is idle."
                : "Correctness still runs; contention changes speed, not answers.");
        }

        if (_timingAllowed)
        {
            // Whatever the guard said, a moving clock means the two halves of a ratio did
            // not run on the same machine state.
            _clockAtStart = GpuBenchmarkEnvironment.SampleSmClockMhz();
            Console.WriteLine();
            Console.WriteLine("SM clock at start: " + _clockAtStart + " MHz");
        }

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

        foreach (var (label, program) in Programs())
        {
            try
            {
                bool ok = CheckProgram(runtime, label, program);
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
        if (_timingAllowed)
            Console.WriteLine("SM clock across the run: " + GpuBenchmarkEnvironment.DescribeClockDrift(
                _clockAtStart, GpuBenchmarkEnvironment.SampleSmClockMhz()));
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

        // Convolutions. Until CodegenOpKind gained these, the thirteen catalog kernels
        // that carry every measured win on this branch were reachable only as
        // hand-written specs -- no graph could ask for one.
        yield return ("depthwise 3x3 + bias + relu (the bake-off kernel)",
            CodegenLowering.LowerConv2D<float>(CodegenOpKind.DepthwiseConv2D,
                new[] { 4, 32, 28, 28 }, new[] { 32, 3, 3 },
                CodegenConvAttributes.Same3x3, withBias: true, withRelu: true));

        yield return ("dense 1x1 + bias + relu",
            CodegenLowering.LowerConv2D<float>(CodegenOpKind.Conv2D,
                new[] { 4, 32, 28, 28 }, new[] { 32, 32, 1, 1 },
                CodegenConvAttributes.Valid, withBias: true, withRelu: true));

        yield return ("dense 3x3 + bias + relu",
            CodegenLowering.LowerConv2D<float>(CodegenOpKind.Conv2D,
                new[] { 2, 16, 28, 28 }, new[] { 16, 16, 3, 3 },
                CodegenConvAttributes.Same3x3, withBias: true, withRelu: true));

        yield return ("conv-transpose 3x3 stride 2",
            CodegenLowering.LowerConv2D<float>(CodegenOpKind.ConvTranspose2D,
                new[] { 2, 16, 16, 16 }, new[] { 16, 16, 3, 3 },
                new CodegenConvAttributes(2, 2, 1, 1)));

        // Global average pooling: a mean over the spatial axes, which is a SUM with a
        // constant 1/(H*W) epilogue rather than its own reduce kind.
        yield return ("global average pool [8,64,28,28] -> [8,64]", GlobalAveragePool(8, 64, 28, 28));

        // Activations. These carry APPROXIMATE PTX instructions (ex2, rcp, tanh), so they
        // are the first kernels that cannot reach exact agreement with the fp64 oracle.
        // The deviation is the point of the row.
        foreach (var op in new[]
                 {
                     CodegenOpKind.Sigmoid, CodegenOpKind.Tanh,
                     CodegenOpKind.Swish, CodegenOpKind.GELU,
                 })
        {
            yield return ("conv 1x1 + bias + " + op.ToString().ToLowerInvariant(),
                ConvWithActivation(op));
        }
    }

    /// <summary>Longest reduction any pass performs, which bounds the accumulated error.</summary>
    private static long LongestReduction(CodegenProgram program)
    {
        long longest = 1;
        foreach (var pass in program.Passes)
        {
            long trips = 1;
            foreach (int axis in pass.Space.ReductionAxes) trips *= pass.Space.Axes[axis].Extent;
            longest = Math.Max(longest, trips);
        }
        return longest;
    }

    private static IEnumerable<(string Label, CodegenProgram Program)> Programs()
    {
        yield return ("softmax rows 512x256 (3 passes)", CodegenFusedStatistics.Softmax(512, 256));
        // Two lengths of the SAME operator. If the deviation tracks the reduction length
        // it is fp32 accumulation; if it does not, something is wrong with the maths.
        yield return ("mse per-sample 512x256 (1 kernel)",
            new CodegenProgram(new[] { CodegenFusedStatistics.MeanSquaredError(512, 256) },
                new long[] { 512 }, "mse"));
        yield return ("layernorm stats 512x64 (2 passes)",
            CodegenFusedStatistics.LayerNormStatistics(512, 64));
        yield return ("layernorm stats 512x256 (2 passes)",
            CodegenFusedStatistics.LayerNormStatistics(512, 256));
    }

    /// <summary>
    /// Runs a multi-pass program on the device and compares the final output against the
    /// fp64 interpretation of the same passes.
    /// </summary>
    /// <remarks>
    /// Parameter 0 of every pass is the source tensor; later parameters are the statistics
    /// earlier passes produced, in order. That convention is what lets the passes be
    /// chained without a scheduler.
    /// </remarks>
    private static bool CheckProgram(DirectPtxRuntime runtime, string label, CodegenProgram program)
    {
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        var buffers = new List<DirectPtxBuffer>();
        var modules = new List<DirectPtxModule>();
        try
        {
            long sourceCount = program.Passes[0].Inputs[0].ElementCount;
            var host = new float[sourceCount];
            var wide = new double[sourceCount];
            for (long e = 0; e < sourceCount; e++)
            {
                double v = ((((e * 37) % 97) - 48) / 16.0);
                host[e] = (float)v;
                wide[e] = v;
            }

            var source = runtime.AllocateBytes((nuint)(sourceCount * sizeof(float)));
            source.Upload<float>(host);
            buffers.Add(source);

            var producedGpu = new List<DirectPtxBuffer>();
            var producedCpu = new List<double[]>();

            foreach (var pass in program.Passes)
            {
                var emitter = new PtxAffineEmitter();
                string ptx = emitter.Emit(pass, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
                var module = runtime.LoadModule(ptx);
                modules.Add(module);
                IntPtr fn = module.GetFunction(pass.Name, out _);

                var args = new IntPtr[pass.ParameterCount];
                var cpuOperands = new double[pass.Inputs.Count][];
                for (int i = 0; i < pass.Inputs.Count; i++)
                {
                    // Parameter 0 is the source. A later parameter is an earlier pass's
                    // output when one exists, and otherwise a SECOND source -- which is
                    // what a two-operand kernel like MSE needs, since it consumes two
                    // tensors and produces no intermediate at all.
                    if (i == 0)
                    {
                        args[i] = source.Pointer;
                        cpuOperands[i] = wide;
                    }
                    else if (i - 1 < producedGpu.Count)
                    {
                        args[i] = producedGpu[i - 1].Pointer;
                        cpuOperands[i] = producedCpu[i - 1];
                    }
                    else
                    {
                        long extra = pass.Inputs[i].ElementCount;
                        var extraHost = new float[extra];
                        var extraWide = new double[extra];
                        for (long e = 0; e < extra; e++)
                        {
                            double v = ((((e * 53 + 17) % 89) - 44) / 16.0);
                            extraHost[e] = (float)v;
                            extraWide[e] = v;
                        }
                        var extraBuffer = runtime.AllocateBytes((nuint)(extra * sizeof(float)));
                        extraBuffer.Upload<float>(extraHost);
                        buffers.Add(extraBuffer);
                        args[i] = extraBuffer.Pointer;
                        cpuOperands[i] = extraWide;
                    }
                }

                long outCount = pass.Output.ElementCount;
                var outBuffer = runtime.AllocateBytes((nuint)(outCount * sizeof(float)));
                buffers.Add(outBuffer);
                args[pass.Inputs.Count] = outBuffer.Pointer;

                Launch(module, fn, args, emitter.LaunchBlocks,
                    (uint)emitter.LaunchBlockX, (uint)emitter.LaunchBlockY);

                producedGpu.Add(outBuffer);
                producedCpu.Add(pass.Interpret(cpuOperands));
            }
            runtime.Synchronize();

            var final = program.Passes[program.Passes.Count - 1];
            long count = final.Output.ElementCount;
            var got = new float[count];
            producedGpu[producedGpu.Count - 1].Download<float>(got);
            double[] want = producedCpu[producedCpu.Count - 1];

            double worst = 0, scale = 0;
            for (long e = 0; e < count; e++)
            {
                worst = Math.Max(worst, Math.Abs(got[e] - want[e]));
                scale = Math.Max(scale, Math.Abs(want[e]));
            }
            double deviation = scale > 0 ? worst / scale : worst;

            // THE TOLERANCE IS A FUNCTION OF THE REDUCTION LENGTH, not a constant.
            // Sequential fp32 accumulation drifts with the number of terms, and measuring
            // the same operator at two lengths showed exactly that: the LayerNorm variance
            // read 8.316E-007 over 64 terms and 3.335E-006 over 256 -- a 4.01x rise for a
            // 4x longer reduction, which is n*eps and not a defect. A fixed 1e-6 gate is
            // the wrong SHAPE for that, the same way an absolute tolerance was the wrong
            // shape for the autotuner's agreement check.
            long trips = LongestReduction(program);
            double bound = Math.Max(1e-6, trips * 1.2e-7);
            bool ok = deviation <= bound;

            Console.WriteLine(label.PadRight(38) +
                count.ToString("N0", CultureInfo.InvariantCulture).PadLeft(10) +
                deviation.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                "  fp64 " + (ok ? "PASS" : "FAIL").PadLeft(7) +
                ("  <= " + bound.ToString("E1", CultureInfo.InvariantCulture)).PadLeft(14));
            return ok;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
            foreach (var m in modules) m.Dispose();
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    /// <summary>Mean over the two spatial axes of an NCHW tensor.</summary>
    private static CodegenGraph GlobalAveragePool(int n, int c, int h, int w)
    {
        var g = new CodegenGraph();
        int x = Load(g, new[] { n, c, h, w });
        int mean = g.AddNode(new CodegenNode(CodegenOpKind.ReduceMean, new[] { x },
            CodegenElementType.Float32, new[] { n, c }, new[] { 2, 3 }));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mean },
            CodegenElementType.Float32, new[] { n, c }));
        return g;
    }

    /// <summary>A 1x1 convolution with a bias and one activation, to isolate the epilogue.</summary>
    private static CodegenGraph ConvWithActivation(CodegenOpKind activation)
    {
        int[] outShape = { 4, 32, 28, 28 };
        var g = new CodegenGraph();
        int input = Load(g, new[] { 4, 32, 28, 28 });
        int weights = Load(g, new[] { 32, 32, 1, 1 });
        int bias = Load(g, new[] { 1, 32, 1, 1 });

        int conv = g.AddNode(new CodegenNode(CodegenOpKind.Conv2D, new[] { input, weights },
            CodegenElementType.Float32, outShape, CodegenConvAttributes.Valid));
        int add = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { conv, bias },
            CodegenElementType.Float32, outShape));
        int act = g.AddNode(new CodegenNode(activation, new[] { add },
            CodegenElementType.Float32, outShape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { act },
            CodegenElementType.Float32, outShape));
        return g;
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

            void LaunchSingle() => Launch(module, fn, pointers,
                gpu.LastLaunchBlocks, gpu.LastLaunchBlockX, gpu.LastLaunchBlockY);

            LaunchSingle();
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
            double singleUs = Measure(runtime.Synchronize, LaunchSingle);
            Console.WriteLine(label.PadRight(38) +
                count.ToString("N0", CultureInfo.InvariantCulture).PadLeft(10) +
                deviation.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                "  " + reference.PadRight(5) +
                (ok ? "PASS" : "FAIL").PadLeft(7) +
                Us(singleUs) +
                ("  " + gpu.LastLaunchBlocks + "blk x" +
                 gpu.LastLaunchBlockX * gpu.LastLaunchBlockY).PadLeft(14));

            // Say why the stronger reference was unavailable, rather than letting a
            // weaker check pass as though it were the same one.
            if (reference == "fp64")
                foreach (var (target, why) in cpuDeclines)
                    Console.WriteLine("    no CPU reference from " + target + ": " + why.Split('\n')[0]);

            // The split route is an optimisation, so it is checked against the SAME
            // reference rather than trusted. A two-kernel path through a temporary is
            // exactly the shape that produces a fast wrong answer.
            if (gpu.LastSplitProgram is { } split)
                ok &= CheckSplit(runtime, label, split, spec, host, want);

            return ok;
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    /// <summary>
    /// Runs the two-kernel split route and requires it to agree with the same reference
    /// the single kernel was held to.
    /// </summary>
    private static bool CheckSplit(
        DirectPtxRuntime runtime, string label, PtxSplitProgram split,
        CodegenKernelSpec spec, float[][] host, double[] want)
    {
        var buffers = new List<DirectPtxBuffer>();
        try
        {
            using var partialModule = runtime.LoadModule(split.PartialSource, allowExperimentalJitFallback: true);
            using var combineModule = runtime.LoadModule(split.CombineSource, allowExperimentalJitFallback: true);
            IntPtr partialFn = partialModule.GetFunction(split.PartialName, out _);
            IntPtr combineFn = combineModule.GetFunction(split.CombineName, out _);

            var plan = split.Plan;
            long count = want.Length;

            // The partial pass takes only the PRODUCT operands; the epilogue operands
            // moved to the combine, so binding by position would feed it the bias.
            var partialArgs = new IntPtr[plan.Partial.ParameterCount];
            for (int i = 0; i < spec.ProductInputs.Count; i++)
            {
                int source = spec.ProductInputs[i];
                var buffer = runtime.AllocateBytes((nuint)(host[source].Length * sizeof(float)));
                buffer.Upload<float>(host[source]);
                buffers.Add(buffer);
                partialArgs[i] = buffer.Pointer;
            }

            var temp = runtime.AllocateBytes((nuint)(split.TempElements * sizeof(float)));
            buffers.Add(temp);
            partialArgs[partialArgs.Length - 1] = temp.Pointer;

            var combineArgs = new IntPtr[plan.Combine.ParameterCount];
            combineArgs[0] = temp.Pointer;
            if (plan.Combine.BiasInput is { } bias)
            {
                int source = spec.BiasInput!.Value;
                var buffer = runtime.AllocateBytes((nuint)(host[source].Length * sizeof(float)));
                buffer.Upload<float>(host[source]);
                buffers.Add(buffer);
                combineArgs[bias] = buffer.Pointer;
            }
            if (plan.Combine.ScaleInput is { } scale)
            {
                int source = spec.ScaleInput!.Value;
                var buffer = runtime.AllocateBytes((nuint)(host[source].Length * sizeof(float)));
                buffer.Upload<float>(host[source]);
                buffers.Add(buffer);
                combineArgs[scale] = buffer.Pointer;
            }

            var outBuffer = runtime.AllocateBytes((nuint)(count * sizeof(float)));
            buffers.Add(outBuffer);
            combineArgs[combineArgs.Length - 1] = outBuffer.Pointer;

            void LaunchSplit()
            {
                Launch(partialModule, partialFn, partialArgs,
                    split.PartialBlocks, split.PartialBlockX, split.PartialBlockY);
                Launch(combineModule, combineFn, combineArgs,
                    split.CombineBlocks, split.CombineBlockX, split.CombineBlockY);
            }

            LaunchSplit();
            runtime.Synchronize();

            var got = new float[count];
            outBuffer.Download<float>(got);

            double worst = 0, scale2 = 0;
            for (long e = 0; e < count; e++)
            {
                worst = Math.Max(worst, Math.Abs(got[e] - want[e]));
                scale2 = Math.Max(scale2, Math.Abs(want[e]));
            }
            double deviation = scale2 > 0 ? worst / scale2 : worst;
            bool ok = deviation <= 1e-6;

            // The emitter ADVERTISES this as the faster route, so the claim is measured
            // rather than asserted. A split offered on a shape it does not help is a bug
            // in the threshold, not a free option -- it costs a temporary and a launch.
            double splitUs = Measure(runtime.Synchronize, LaunchSplit);

            string axes = string.Join("+", plan.PromotedAxes);
            Console.WriteLine(("  split on axis " + axes).PadRight(38) +
                split.TempElements.ToString("N0", CultureInfo.InvariantCulture).PadLeft(10) +
                deviation.ToString("E3", CultureInfo.InvariantCulture).PadLeft(14) +
                "  split" + (ok ? "PASS" : "FAIL").PadLeft(7) +
                Us(splitUs) +
                ("  " + split.PartialBlocks + "+" + split.CombineBlocks + "blk").PadLeft(14));
            return ok;
        }
        catch (Exception ex)
        {
            Console.WriteLine("  split                                      -             -   ERROR");
            Console.WriteLine("    " + ex.GetType().Name + ": " + ex.Message.Split('\n')[0]);
            return false;
        }
        finally { foreach (var b in buffers) b.Dispose(); }
    }

    /// <summary>
    /// Median of medians over three runs, the p4 shape at a smaller sample; NaN when a
    /// foreign GPU workload makes the number meaningless.
    /// </summary>
    private static double Measure(Action synchronize, Action launch)
    {
        if (!_timingAllowed) return double.NaN;

        const int Warmup = 20, Samples = 15, PerSample = 50;
        double best = double.MaxValue;
        for (int run = 0; run < 3; run++)
        {
            for (int i = 0; i < Warmup; i++) launch();
            synchronize();

            var samples = new double[Samples];
            for (int i = 0; i < Samples; i++)
            {
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                for (int k = 0; k < PerSample; k++) launch();
                synchronize();
                samples[i] = System.Diagnostics.Stopwatch.GetElapsedTime(start)
                    .TotalMilliseconds / PerSample * 1000.0;
            }
            Array.Sort(samples);
            best = Math.Min(best, samples[samples.Length / 2]);
        }
        return best;
    }

    /// <summary>Formats a timing, or blanks it when it was not measurable.</summary>
    private static string Us(double value) => double.IsNaN(value)
        ? "        -   "
        : value.ToString("F1", CultureInfo.InvariantCulture).PadLeft(9) + " us";

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
