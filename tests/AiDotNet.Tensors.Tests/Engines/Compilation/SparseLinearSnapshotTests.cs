using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #1235: <c>SparseLinearFusionPattern</c> snapshotted a mostly-zero MatMul weight to CSR at compile time and used the
/// snapshot forever, so a weight written after compile (an ONNX graph input fed per call, a fine-tuned or reloaded
/// weight) kept producing the trace-time result. The fused node now re-checks the weight's mutation version each
/// execute: a sparse rewrite rebuilds the CSR, a dense rewrite switches to the dense GEMM.
/// </summary>
public class SparseLinearSnapshotTests
{
    private const int Batch = 3, In = 8, Out = 6;

    private static float[] Reference(float[] x, float[] w)
    {
        var y = new float[Batch * Out];
        for (int b = 0; b < Batch; b++)
            for (int o = 0; o < Out; o++)
            {
                float s = 0;
                for (int i = 0; i < In; i++) s += x[b * In + i] * w[i * Out + o];
                y[b * Out + o] = s;
            }
        return y;
    }

    private static void SetAll(Tensor<float> t, float[] values)
    {
        for (int i = 0; i < values.Length; i++) t[i] = values[i]; // the public indexer bumps the version
    }

    [Fact]
    public void WeightWrittenAfterCompile_IsUsed_SparseThenDenseThenSparse()
    {
        var engine = new CpuEngine();
        var xData = Enumerable.Range(0, Batch * In).Select(i => (float)Math.Sin(i + 1)).ToArray();
        var x = new Tensor<float>((float[])xData.Clone(), new[] { Batch, In });
        var w = new Tensor<float>(new float[In * Out], new[] { In, Out });
        w[0] = 1f; // mostly zero at compile time: the sparse fusion takes it

        // The ONNX importer's options: dataflow fusion off, so the sparse-linear pattern sees the MatMul.
        CompiledInferencePlan<float> plan;
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions
        {
            EnableDataflowFusion = false,
            EnableSpectralDecomposition = false,
        });
        try
        {
            using var scope = GraphMode.EnableInference();
            var y = engine.TensorMatMul(x, w);
            plan = scope.CompileInference<float>(y);
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(null);
        }
        Assert.Contains("FusedSparseLinear", StepNames(plan)); // the guarded node is what runs

        // Still sparse after the write: the CSR is rebuilt.
        var sparse = new float[In * Out];
        sparse[3 * Out + 2] = 2f; sparse[7 * Out + 5] = -1.5f;
        SetAll(w, sparse);
        Assert.Equal(Reference(xData, sparse), plan.Execute().ToArray());

        // Dense now: the dense path, on the current values.
        var dense = Enumerable.Range(0, In * Out).Select(i => (float)Math.Cos(i) + 0.1f).ToArray();
        SetAll(w, dense);
        var got = plan.Execute().ToArray();
        var want = Reference(xData, dense);
        for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(got[i] - want[i]) < 1e-4f, $"dense [{i}] {got[i]} vs {want[i]}");

        // Back to sparse.
        SetAll(w, sparse);
        Assert.Equal(Reference(xData, sparse), plan.Execute().ToArray());
    }

    private static string[] StepNames(CompiledInferencePlan<float> plan)
    {
        var steps = typeof(CompiledInferencePlan<float>)
            .GetField("_steps", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
            ?.GetValue(plan) as Array;
        const System.Reflection.BindingFlags all = System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
        return steps?.Cast<object>().Select(s => s.GetType().GetField("OpName", all)?.GetValue(s) as string ?? "").ToArray()
            ?? Array.Empty<string>();
    }
}
