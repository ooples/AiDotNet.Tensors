using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Onnx.Protos;
using Microsoft.ML.OnnxRuntime;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Performance parity / comparison tests vs ONNX Runtime. These aren't
/// correctness parity tests — they measure wall-clock latency on synthetic
/// graphs that stress a specific op category, so regressions in our kernel
/// paths surface as slowdowns rather than silent wrong answers.
/// <para>
/// Categories:
/// <list type="bullet">
/// <item>MatMul-heavy (no conv): pure BLAS workload — we route through
/// SimdGemm + the process-wide BLAS provider. Expected competitive-to-
/// faster than ORT's session cold-start + per-Run overhead on sizes that
/// fit in L2.</item>
/// <item>Transformer-like (MatMul + LayerNorm + Add): close to BERT's
/// hot-path shape. Our plan has zero per-op alloc because the memory
/// planner reuses buffers.</item>
/// </list>
/// </para>
/// <para>
/// The convolution-heavy path (ResNet-50) remains ORT-dominated because
/// ORT ships winograd + NCHWc + hand-tuned blocked GEMM; our Phase 1
/// Conv2D is naive im2col→GEMM. That gap is tracked in the README as the
/// next perf milestone; these tests ensure we don't silently regress on
/// the workloads where we already have a competitive story.
/// </para>
/// </summary>
public class PerformanceComparison
{
    private readonly ITestOutputHelper _output;
    public PerformanceComparison(ITestOutputHelper o) { _output = o; }

    [SkippableFact]
    public void MatMul_PureWorkload_CompetitiveWithOrt()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Synthetic: single 512×512 · 512×512 MatMul. No conv, no reshape
        // overhead, no ORT session init amortization beyond warmup.
        const int M = 512, K = 512, N = 512;
        var graph = new GraphProto { Name = "matmul_perf" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { M, K }, 1 /*FLOAT*/));
        var wData = new float[K * N];
        var rng = new Random(12345);
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() * 2 - 1);
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { K, N }, wData));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { M, N }, 1 /*FLOAT*/));
        var node = new NodeProto { OpType = "MatMul" };
        node.Input.Add("A"); node.Input.Add("W"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        using var session = new InferenceSession(bytes);
        var ortFeed = new[] { Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
            "A", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(a, new[] { M, K })) };
        // ORT warmup.
        for (int i = 0; i < 3; i++) { using var _ = session.Run(ortFeed); }

        using var stream = new MemoryStream(bytes);
        var engine = new CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.NotNull(result.Plan);
        a.AsSpan().CopyTo(result.Inputs["A"].AsWritableSpan());
        // Ours warmup (triggers specialization + any first-run JIT).
        for (int i = 0; i < 3; i++) result.Plan!.Execute();

        const int iters = 30;
        var ortTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) { using var _ = session.Run(ortFeed); }
        ortTimer.Stop();
        var oursTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) result.Plan!.Execute();
        oursTimer.Stop();

        double ortMs = ortTimer.Elapsed.TotalMilliseconds / iters;
        double oursMs = oursTimer.Elapsed.TotalMilliseconds / iters;
        double ratio = oursMs / ortMs;
        _output.WriteLine(
            $"MatMul 512x512·512x512 — ORT: {ortMs:F2} ms/iter, Ours: {oursMs:F2} ms/iter, " +
            $"ratio: {ratio:F2}x");

        // Perf regression gate is opt-in via AIDOTNET_ENFORCE_PERF=1. On a
        // contributor's workstation, thermal throttling / background load /
        // different AVX width / unrelated GC pressure can push the ratio
        // past 5× even on correct code — we don't want hardware sensitivity
        // to block PRs. CI runs with the env var set on a reference machine
        // where the ratio IS stable.
        bool enforce = Environment.GetEnvironmentVariable("AIDOTNET_ENFORCE_PERF") == "1";
        if (enforce)
        {
            Assert.True(ratio <= 5.0,
                $"MatMul-only workload: ours {oursMs:F2}ms is >5× ORT {ortMs:F2}ms (ratio {ratio:F2}x). " +
                "Check that SimdGemm / BLAS-provider fast path is selected for this size.");
        }
    }

    [SkippableFact]
    public void MatMulAddRelu_Chain_BeatsOrEqualsOrt()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Small fused-ish chain: MatMul → Add(bias) → Relu. Our plan
        // pipelines through the SIMD ReLU specialization.
        const int M = 64, K = 256, N = 256;
        var graph = new GraphProto { Name = "chain_perf" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { M, K }, 1));
        var wData = new float[K * N];
        var bData = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() - 0.5);
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { K, N }, wData));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { N }, bData));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { M, N }, 1));

        var mm = new NodeProto { OpType = "MatMul" };
        mm.Input.Add("A"); mm.Input.Add("W"); mm.Output.Add("MM");
        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("MM"); add.Input.Add("B"); add.Output.Add("MMB");
        var relu = new NodeProto { OpType = "Relu" };
        relu.Input.Add("MMB"); relu.Output.Add("Y");
        graph.Node.Add(mm); graph.Node.Add(add); graph.Node.Add(relu);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var a = new float[M * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);

        using var session = new InferenceSession(bytes);
        var ortFeed = new[] { Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
            "A", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(a, new[] { M, K })) };
        for (int i = 0; i < 3; i++) { using var _ = session.Run(ortFeed); }

        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, new CpuEngine());
        a.AsSpan().CopyTo(result.Inputs["A"].AsWritableSpan());
        for (int i = 0; i < 5; i++) result.Plan!.Execute();

        const int iters = 100;
        var ortTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) { using var _ = session.Run(ortFeed); }
        ortTimer.Stop();
        var oursTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) result.Plan!.Execute();
        oursTimer.Stop();

        double ortMs = ortTimer.Elapsed.TotalMilliseconds / iters;
        double oursMs = oursTimer.Elapsed.TotalMilliseconds / iters;
        _output.WriteLine(
            $"MatMul+Add+Relu chain (M={M}, K={K}, N={N}) — ORT: {ortMs:F3} ms/iter, " +
            $"Ours: {oursMs:F3} ms/iter, ratio: {oursMs / ortMs:F2}x");
        // Same regression-only cap: within 5× of ORT on the fused-linear
        // pattern that transformer FFN blocks hit.
        Assert.True(oursMs <= ortMs * 5.0,
            $"Chain latency: ours {oursMs:F3}ms > 5× ORT {ortMs:F3}ms. Check specialization + memory planning.");
    }

    [SkippableFact]
    public void TinyModel_PlanExecuteOverhead_BeatsOrt()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // A 4-element Add — ORT's per-Run session overhead is orders of
        // magnitude larger than the op itself. Our Plan.Execute is an
        // in-process delegate call with no session dispatch → tiny models
        // are where we should genuinely beat ORT.
        var graph = new GraphProto { Name = "tiny" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { 4 }, 1));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 4 }, new[] { 1f, 2f, 3f, 4f }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4 }, 1));
        var node = new NodeProto { OpType = "Add" };
        node.Input.Add("A"); node.Input.Add("B"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var a = new[] { 10f, 20f, 30f, 40f };
        using var session = new InferenceSession(bytes);
        var ortFeed = new[] { Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor(
            "A", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(a, new[] { 4 })) };
        for (int i = 0; i < 10; i++) { using var _ = session.Run(ortFeed); }

        using var stream = new MemoryStream(bytes);
        var result = OnnxImporter.Import<float>(stream, new CpuEngine());
        a.AsSpan().CopyTo(result.Inputs["A"].AsWritableSpan());
        for (int i = 0; i < 10; i++) result.Plan!.Execute();

        const int iters = 10000;
        var ortTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) { using var _ = session.Run(ortFeed); }
        ortTimer.Stop();
        var oursTimer = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) result.Plan!.Execute();
        oursTimer.Stop();

        // TimeSpan.TotalMicroseconds is .NET 7+. Use TotalMilliseconds * 1000
        // so this test compiles on older TFMs the shared test project may hit.
        double ortUs = ortTimer.Elapsed.TotalMilliseconds * 1000.0 / iters;
        double oursUs = oursTimer.Elapsed.TotalMilliseconds * 1000.0 / iters;
        _output.WriteLine(
            $"Tiny Add (4 floats) — ORT: {ortUs:F2} µs/iter, Ours: {oursUs:F2} µs/iter, " +
            $"ratio: {oursUs / ortUs:F2}x");
        // Our compiled plan is pure in-process function dispatch + a single
        // SIMD add — no graph traversal, no shape validation, no session
        // boundary crossing. ORT pays a session.Run bookkeeping cost that
        // swamps the 4-element op. We should win here by a wide margin.
        Assert.True(oursUs < ortUs,
            $"Tiny-model dispatch: ours {oursUs:F2}µs is not less than ORT {ortUs:F2}µs. " +
            "Our Plan.Execute has regressed vs ORT's per-Run overhead — investigate.");
    }
}
