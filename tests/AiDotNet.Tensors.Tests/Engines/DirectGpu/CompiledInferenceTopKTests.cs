// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// An inference plan records TensorTopK, and the row selection that reads its indices, as a top-k router does: scores
/// = x · W, the top k per row, and the rows of a table those indices pick, weighted by the top values. Every
/// execution must read that execution's indices, on the CPU engine, on the GPU engine, and replayed from a captured
/// CUDA graph. The inference trace used to refuse TensorTopK, so the plan could not be compiled at all.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CompiledInferenceTopKTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private const int N = 6, E = 8, X = 10, K = 2, D = 5;
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CompiledInferenceTopKTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // [N, K, D] + [N, K, X]: row n's k-th pick of the table scaled by its score, plus the one-hot of that pick (as a
    // router's gates are built), so both index readers are exercised.
    private static Tensor<float> Route(IEngine e, Tensor<float> x, Tensor<float> w, Tensor<float> table)
    {
        var scores = e.TensorMatMul(x, w);                                   // [N, X]
        var top = e.TensorTopK(scores, K, 1, out Tensor<int> idx);           // [N, K]
        var picked = e.TensorIndexSelect(table, idx, 0);                     // [N * K, D]
        var weighted = e.TensorMultiply(e.Reshape(picked, [N, K, D]), e.Reshape(top, [N, K, 1]));
        var oneHot = e.TensorOneHot<float>(idx, X);                      // [N * K, X] (reads idx flat)
        var gated = e.TensorMultiply(e.Reshape(oneHot, [N, K, X]), e.Reshape(scores, [N, 1, X]));
        return e.TensorConcatenate(new[] { e.Reshape(weighted, [N * K * D]), e.Reshape(gated, [N * K * X]) }, 0);
    }

    private static void Run(IEngine engine, bool capture)
    {
        var w = Rand([E, X], 1);
        var table = Rand([X, D], 2);
        var x = Rand([N, E], 3);
        var cpu = new CpuEngine();
        CompiledInferencePlan<float> plan;
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var scope = GraphMode.EnableInference();
            var traced = Route(engine, x, w, table);
            plan = scope.CompileInference(traced, new[] { x });
        }
        finally { AiDotNetEngine.Current = priorEngine; }
        var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
        CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
        try
        {
            using (plan)
            {
                for (int call = 0; call < 6; call++)
                {
                    // Each call's input picks different rows: a replay reading the trace's indices would be wrong.
                    var input = Rand([N, E], 10 + call);
                    var expected = Route(cpu, input, w, table).ToArray();
                    plan.SetInputs(new[] { input });
                    var got = plan.Execute().ToArray();
                    Assert.Equal(expected.Length, got.Length);
                    for (int i = 0; i < got.Length; i++)
                        Assert.True(Math.Abs(expected[i] - got[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                            $"call {call} [{i}]: eager {expected[i]}, plan {got[i]}");
                }
                if (capture)
                {
                    var exec = typeof(CompiledInferencePlan<float>)
                        .GetField("_graphExec", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
                    Assert.True((IntPtr)exec.GetValue(plan)! != IntPtr.Zero, "the plan did not replay a captured graph");
                }
            }
        }
        finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
    }

    [Fact]
    public void A_top_k_router_compiles_into_an_inference_plan_on_the_cpu_engine() => Run(new CpuEngine(), capture: false);

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_top_k_router_compiles_into_an_inference_plan_on_the_gpu(bool capture)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        if (capture)
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "Inference graph capture is CUDA-only.");
        Run(gpu, capture);
    }
}
