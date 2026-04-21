using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 2B scientific-method drill-down for the ResNet-50 Conv2D
/// bottleneck. <see cref="OpLevelPerfHarness"/> showed Conv <c>3×3×C→C</c>
/// is now the dominant cost after Phase 2A crushed the BERT MatMul path:
/// Conv <c>3×3×512→512</c> on <c>[1,512,7,7]</c> is 102× slower than ORT
/// per call, <c>3×3×256→256</c> on <c>[1,256,14,14]</c> is 77×.
///
/// <para>Same 3-tier evidence format as
/// <see cref="MatMulRootCauseDiag"/>: time the direct engine API, time the
/// full ONNX plan path, and dump per-step breakdown so we can see WHERE
/// the time goes before proposing fixes.</para>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class Conv2DRootCauseDiag
{
    private readonly ITestOutputHelper _output;
    public Conv2DRootCauseDiag(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 3;
    private const int Iters  = 10;

    [SkippableFact]
    public void LocaliseResNetConvBottleneck()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");

        _output.WriteLine($"Avx512Sgemm.CanUse = {Avx512Sgemm.CanUse}");
        _output.WriteLine($"CPU cores = {Environment.ProcessorCount}");

        // Disable AutoTracer to isolate whether its state accumulation is the
        // cause of the plan's per-step overhead. If Plan.Execute speeds up
        // without AutoTracer, we've found the culprit.
        bool disableAutoTracer = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_AUTOTRACER") == "1";
        if (disableAutoTracer)
        {
            AutoTracer.Enabled = false;
            _output.WriteLine($"AutoTracer.Enabled = false (set via env)");
        }
        _output.WriteLine("");

        // ResNet-50 hot convolutions in descending per-call cost.
        (int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label)[] cases =
        {
            (1, 512, 7,   7,  512, 3, 3, 1, 1, "stage4 3x3 [1,512,7,7]->[1,512,7,7]"),
            (1, 256, 14, 14,  256, 3, 3, 1, 1, "stage3 3x3 [1,256,14,14]->[1,256,14,14]"),
            (1, 128, 28, 28,  128, 3, 3, 1, 1, "stage2 3x3 [1,128,28,28]->[1,128,28,28]"),
            (1,  64, 56, 56,   64, 3, 3, 1, 1, "stage1 3x3 [1,64,56,56]->[1,64,56,56]"),
            (1,  64, 56, 56,   64, 1, 1, 1, 0, "bottleneck 1x1 [1,64,56,56]->[1,64,56,56]"),
        };

        foreach (var cs in cases)
        {
            _output.WriteLine($"=== {cs.label} ===");
            // Theoretical FLOPs = 2 × batch × Cin × Cout × kH × kW × outH × outW
            int outH = (cs.h + 2 * cs.pad - cs.kH) / cs.stride + 1;
            int outW = (cs.w + 2 * cs.pad - cs.kW) / cs.stride + 1;
            double flops = 2.0 * cs.n * cs.c * cs.co * cs.kH * cs.kW * outH * outW;
            double gflop = flops / 1e9;

            double t1 = TimeDirectEngineConv2D(cs);
            _output.WriteLine($"  [1] Direct engine.Conv2D:            {t1 * 1000:F1} µs  ({gflop / t1:F1} G·op/ms)");

            double t1b = TimeClosureDirectEngineConv2D(cs);
            _output.WriteLine($"  [1b] via Action<eng,out> + CopyTo:   {t1b * 1000:F1} µs  ({gflop / t1b:F1} G·op/ms)");

            double t1c = TimeDirectConvPreConvertedNchwc(cs);
            _output.WriteLine($"  [1c] Direct with NCHWc input pre-pack:{t1c * 1000:F1} µs  ({gflop / t1c:F1} G·op/ms)");

            double t2 = TimeOnnxCompiledPlan(cs, out int stepCount, out double[] perIter);
            _output.WriteLine($"  [2] OnnxImport → Plan.Execute:      {t2 * 1000:F1} µs  ({gflop / t2:F1} G·op/ms)  StepCount={stepCount}");
            _output.WriteLine($"      per-iter ms: {string.Join(", ", Array.ConvertAll(perIter, v => v.ToString("F1")))}");

            // [2b] Use the plan's CAPTURED tensors directly (bypass the
            // plan's closure/step machinery) and call engine.Conv2D.
            // If this is fast, the overhead lives in the plan's Execute
            // loop. If this is slow (matches [2]), the captured tensors
            // themselves carry context that slows Conv2D.
            double t2b = TimeWithCapturedTensors(cs);
            _output.WriteLine($"  [2b] Plan's captured tensors direct:{t2b * 1000:F1} µs  ({gflop / t2b:F1} G·op/ms)");

            var steps = ProfilePerStepOf(cs);
            _output.WriteLine("      Per-step breakdown:");
            for (int i = 0; i < steps.Length; i++)
                _output.WriteLine($"        step[{i}] {steps[i].OpName,-30} {steps[i].AvgMs * 1000:F1} µs");

            _output.WriteLine($"  Ratio [2]/[1] = {t2 / t1:F1}x  (plan-layer overhead)");
            _output.WriteLine("");
        }
    }

    // ─── timed paths ────────────────────────────────────────────────────────

    private static double TimeDirectEngineConv2D((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { cs.n, cs.c, cs.h, cs.w });
        var kernel = new Tensor<float>(new[] { cs.co, cs.c, cs.kH, cs.kW });
        Rand(0xC01, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(input.AsWritableSpan());
        Rand(0xC02, cs.co * cs.c * cs.kH * cs.kW).AsSpan().CopyTo(kernel.AsWritableSpan());

        for (int i = 0; i < Warmup; i++)
            _ = engine.Conv2D(input, kernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            _ = engine.Conv2D(input, kernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    /// <summary>
    /// Timed path equivalent to the plan's closure: call Conv2D, then
    /// CopyTo a pre-allocated output — mirror of the LazyNode Execute
    /// delegate pattern. Tests whether the delegate + CopyTo itself is
    /// what adds the observed plan overhead (vs. something plan-specific).
    /// </summary>
    private static double TimeClosureDirectEngineConv2D((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { cs.n, cs.c, cs.h, cs.w });
        var kernel = new Tensor<float>(new[] { cs.co, cs.c, cs.kH, cs.kW });
        Rand(0xC01, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(input.AsWritableSpan());
        Rand(0xC02, cs.co * cs.c * cs.kH * cs.kW).AsSpan().CopyTo(kernel.AsWritableSpan());

        int outH = (cs.h + 2 * cs.pad - cs.kH) / cs.stride + 1;
        int outW = (cs.w + 2 * cs.pad - cs.kW) / cs.stride + 1;
        var preAllocOutput = new Tensor<float>(new[] { cs.n, cs.co, outH, outW });

        Action<IEngine, Tensor<float>> closure = (eng, output) =>
        {
            var eager = eng.Conv2D(input, kernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });
            eager.AsSpan().CopyTo(output.AsWritableSpan());
        };

        for (int i = 0; i < Warmup; i++) closure(engine, preAllocOutput);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) closure(engine, preAllocOutput);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    /// <summary>
    /// Direct call with input PRE-CONVERTED to NCHWc8 layout so the int[]
    /// Conv2D overload's NCHWc fast path fires. Compare to direct NCHW
    /// to see how much NCHWc buys vs the im2col+GEMM fallback.
    /// </summary>
    private static double TimeDirectConvPreConvertedNchwc((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { cs.n, cs.c, cs.h, cs.w });
        var kernel = new Tensor<float>(new[] { cs.co, cs.c, cs.kH, cs.kW });
        Rand(0xC01, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(input.AsWritableSpan());
        Rand(0xC02, cs.co * cs.c * cs.kH * cs.kW).AsSpan().CopyTo(kernel.AsWritableSpan());

        // Only cb=8 covers these channel counts (64, 128, 256, 512 all divide 8).
        if (cs.c % 8 != 0 || cs.co % 8 != 0) return double.NaN;

        var packedInput = engine.ReorderToNchwc(input, LinearAlgebra.TensorLayout.Nchwc8);

        for (int i = 0; i < Warmup; i++)
            _ = engine.Conv2D(packedInput, kernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            _ = engine.Conv2D(packedInput, kernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static double TimeOnnxCompiledPlan((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs, out int stepCount, out double[] perIter)
    {
        var model = BuildSingleConvModel(cs);
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var engine = new CpuEngine();
        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        Rand(0xC03, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(result.Inputs["X"].AsWritableSpan());
        stepCount = result.Plan!.StepCount;

        for (int i = 0; i < Warmup; i++) result.Plan.Execute();

        perIter = new double[Iters];
        double tickToMs = 1000.0 / Stopwatch.Frequency;
        long totalTicks = 0;
        for (int i = 0; i < Iters; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            result.Plan.Execute();
            long t1 = Stopwatch.GetTimestamp();
            perIter[i] = (t1 - t0) * tickToMs;
            totalTicks += (t1 - t0);
        }
        return totalTicks * tickToMs / Iters;
    }

    /// <summary>
    /// Time engine.Conv2D using the tensors captured in the plan's step[0].Inputs.
    /// If this is as fast as [1], the plan-execution loop itself adds overhead.
    /// If this is as slow as [2], the captured-tensor references carry something
    /// that slows Conv2D (layout flag, pinning state, shape metadata, etc.).
    /// </summary>
    private static double TimeWithCapturedTensors((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        var model = BuildSingleConvModel(cs);
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var engine = new CpuEngine();
        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        Rand(0xC03, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(result.Inputs["X"].AsWritableSpan());
        var plan = (CompiledInferencePlan<float>)result.Plan!;

        // Access the plan's step[0] (Conv2D) input tensors directly via
        // the InternalsVisibleTo hook. step[0].Inputs[0] = input, [1] = kernel.
        var stepsField = typeof(CompiledInferencePlan<float>).GetField("_steps",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var steps = (CompiledStep<float>[])stepsField.GetValue(plan)!;
        var capturedInput = steps[0].Inputs[0];
        var capturedKernel = steps[0].Inputs[1];

        // Sanity — shapes should match the test case.
        Assert.Equal(cs.n, capturedInput._shape[0]);
        Assert.Equal(cs.c, capturedInput._shape[1]);

        for (int i = 0; i < Warmup; i++)
            _ = engine.Conv2D(capturedInput, capturedKernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++)
            _ = engine.Conv2D(capturedInput, capturedKernel, new[] { cs.stride, cs.stride }, new[] { cs.pad, cs.pad }, new[] { 1, 1 });
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / Iters;
    }

    private static (string OpName, double AvgMs)[] ProfilePerStepOf((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        var model = BuildSingleConvModel(cs);
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var engine = new CpuEngine();
        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        Rand(0xC04, cs.n * cs.c * cs.h * cs.w).AsSpan().CopyTo(result.Inputs["X"].AsWritableSpan());
        var plan = (CompiledInferencePlan<float>)result.Plan!;
        return plan.ProfilePerStep(Warmup, Iters);
    }

    // ─── model builder ──────────────────────────────────────────────────────

    private static ModelProto BuildSingleConvModel((int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, string label) cs)
    {
        int outH = (cs.h + 2 * cs.pad - cs.kH) / cs.stride + 1;
        int outW = (cs.w + 2 * cs.pad - cs.kW) / cs.stride + 1;
        var W = Rand(0xC05, cs.co * cs.c * cs.kH * cs.kW);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "Conv",
            inputs: new[] { (name: "X", shape: new[] { cs.n, cs.c, cs.h, cs.w }, elemType: OnnxTestHelpers.FLOAT) },
            output: (name: "Y", shape: new[] { cs.n, cs.co, outH, outW }, elemType: OnnxTestHelpers.FLOAT),
            attributes: new[]
            {
                IntsAttr("kernel_shape", cs.kH, cs.kW),
                IntsAttr("strides",      cs.stride, cs.stride),
                IntsAttr("pads",         cs.pad, cs.pad, cs.pad, cs.pad),
                IntsAttr("dilations",    1, 1),
            },
            initializers: new[] { (name: "W", shape: new[] { cs.co, cs.c, cs.kH, cs.kW }, data: W) });
        return model;
    }

    private static AttributeProto IntsAttr(string name, params int[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) a.Ints.Add(v);
        return a;
    }

    private static float[] Rand(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }
}
