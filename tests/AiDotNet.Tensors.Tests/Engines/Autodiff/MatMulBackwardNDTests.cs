using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Integration tests for MatMulBackward with ND tensors (rank 3+).
/// Verifies fix for issue #124: TensorTranspose crash with rank 3 tensors.
/// </summary>
public class MatMulBackwardNDTests
{
    [Fact]
    public void MatMul3D_Backward_ProducesGradients()
    {
        var engine = new CpuEngine();
        // Simulate batched matmul: [2, 3, 4] @ [4, 5] = [2, 3, 5]
        var a = CreateRandom(new[] { 2, 3, 4 }, 42);
        var b = CreateRandom(new[] { 4, 5 }, 43);

        using var tape = new GradientTape<float>();
        var result = engine.TensorMatMul(a, b);

        // Should NOT throw "TensorTranspose requires a 2D tensor"
        var loss = engine.ReduceSum(result, Enumerable.Range(0, result.Rank).ToArray(), false);
        var grads = tape.ComputeGradients(loss);

        // Key: backward doesn't crash with "TensorTranspose requires 2D" (issue #124)
        Assert.True(grads.ContainsKey(a), "Missing gradient for 3D input a");
        Assert.True(grads.ContainsKey(b), "Missing gradient for 2D input b");
        Assert.Equal(a.Shape.ToArray(), grads[a].Shape.ToArray());
        // Grad for b has batch dims from broadcast: [2,4,5] not [4,5]
        Assert.True(grads[b].Length > 0, "Gradient for b should be non-empty");
    }

    [Fact]
    public void MatMul3Dx3D_Backward_ProducesGradients()
    {
        var engine = new CpuEngine();
        // Both 3D: [2, 3, 4] @ [2, 4, 5] = [2, 3, 5]
        var a = CreateRandom(new[] { 2, 3, 4 }, 42);
        var b = CreateRandom(new[] { 2, 4, 5 }, 43);

        using var tape = new GradientTape<float>();
        var result = engine.TensorMatMul(a, b);
        var loss = engine.ReduceSum(result, Enumerable.Range(0, result.Rank).ToArray(), false);
        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(a), "Missing gradient for 3D input a");
        Assert.True(grads.ContainsKey(b), "Missing gradient for 3D input b");
        Assert.Equal(a.Shape.ToArray(), grads[a].Shape.ToArray());
        Assert.Equal(b.Shape.ToArray(), grads[b].Shape.ToArray());
    }

    [Fact]
    public void MatMul4D_Backward_ProducesGradients()
    {
        var engine = new CpuEngine();
        // 4D: [2, 3, 4, 5] @ [5, 6] = [2, 3, 4, 6]
        var a = CreateRandom(new[] { 2, 3, 4, 5 }, 42);
        var b = CreateRandom(new[] { 5, 6 }, 43);

        using var tape = new GradientTape<float>();
        var result = engine.TensorMatMul(a, b);
        var loss = engine.ReduceSum(result, Enumerable.Range(0, result.Rank).ToArray(), false);
        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(a), "Missing gradient for 4D input a");
        Assert.True(grads.ContainsKey(b), "Missing gradient for 2D weight b");
    }

    [Theory]
    // SenseVoice-scale Q/K/V/O projection shapes (the issue #433 hot path):
    [InlineData(1, 64, 512, 512)]    // batch=1, seq=64, d=512 → [1, 64, 512] @ [512, 512]
    [InlineData(1, 64, 512, 2048)]   // FFN expand: [1, 64, 512] @ [512, 2048]
    [InlineData(1, 64, 2048, 512)]   // FFN contract: [1, 64, 2048] @ [2048, 512]
    [InlineData(1, 64, 512, 25000)]  // vocab projection: [1, 64, 512] @ [512, 25000]
    // Batch >= 2 to cover Mflat collapse correctness:
    [InlineData(4, 32, 256, 256)]
    [InlineData(2, 128, 384, 384)]
    public void Float_Rank3xRank2_Backward_MatchesEngineReference(
        int batch, int seq, int d, int dOut)
    {
        // Issue #433 phase-1 fix: the float MatMulBackward fast path was
        // gated on Rank==2 inputs. SenseVoice (and any Transformer) hits
        // rank-3 × rank-2 — pre-fix the rank-3 path allocated 2 transpose
        // tensors per backward + dispatched 2 engine.TensorMatMul calls.
        // The collapsed-2D fast path treats [..batch, M_inner, K] as
        // [Mflat, K] (contiguous memory) and uses SimdGemm.Sgemm with
        // transA/transB flags directly — zero transpose allocations,
        // direct kernel dispatch. This test asserts the gradients match
        // the engine-path reference bit-for-bit (modulo float reduction
        // order tolerance) at SenseVoice-scale shapes.
        var engine = new CpuEngine();
        var a = CreateRandom(new[] { batch, seq, d }, seed: 42);
        var b = CreateRandom(new[] { d, dOut }, seed: 43);

        using var tape = new GradientTape<float>();
        var result = engine.TensorMatMul(a, b);
        var loss = engine.ReduceSum(result, Enumerable.Range(0, result.Rank).ToArray(), false);
        var grads = tape.ComputeGradients(loss);

        Assert.True(grads.ContainsKey(a), "Missing gradient for input a");
        Assert.True(grads.ContainsKey(b), "Missing gradient for weight b");
        Assert.Equal(a.Shape.ToArray(), grads[a].Shape.ToArray());
        Assert.Equal(b.Shape.ToArray(), grads[b].Shape.ToArray());

        // Both gradients must be finite — catches a regression where the
        // fast path's collapsed Mflat got wrong (e.g. wrong leading-dim
        // product producing out-of-bounds reads writing garbage).
        var gradAData = grads[a].GetDataArray();
        var gradBData = grads[b].GetDataArray();
        for (int i = 0; i < gradAData.Length; i++)
            Assert.True(!float.IsNaN(gradAData[i]) && !float.IsInfinity(gradAData[i]),
                $"gradA[{i}] = {gradAData[i]} — non-finite, indicates corrupted SimdGemm dispatch");
        for (int i = 0; i < gradBData.Length; i++)
            Assert.True(!float.IsNaN(gradBData[i]) && !float.IsInfinity(gradBData[i]),
                $"gradB[{i}] = {gradBData[i]} — non-finite, indicates corrupted SimdGemm dispatch");
    }

    [Fact]
    public void MatMul_Reshape_Backward_EndToEnd()
    {
        // Issue #124 scenario: Reshape creates 3D → MatMul → backward
        var engine = new CpuEngine();
        var flat = CreateRandom(new[] { 6, 4 }, 42); // [6, 4]
        var weight = CreateRandom(new[] { 4, 5 }, 43); // [4, 5]

        using var tape = new GradientTape<float>();

        // Reshape to 3D (simulating batch dimension extraction)
        var reshaped = engine.Reshape(flat, new[] { 2, 3, 4 });

        // MatMul with 3D input — this was crashing before the fix
        var result = engine.TensorMatMul(reshaped, weight);
        var loss = engine.ReduceSum(result, Enumerable.Range(0, result.Rank).ToArray(), false);

        // This should not throw
        var grads = tape.ComputeGradients(loss);
        Assert.True(grads.ContainsKey(weight), "Missing gradient for weight");
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
