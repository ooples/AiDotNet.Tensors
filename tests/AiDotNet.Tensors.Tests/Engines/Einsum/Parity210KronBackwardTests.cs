using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210KronBackwardTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float tol = 1e-4f) => System.MathF.Abs(a - b) <= tol * (1 + System.MathF.Abs(a) + System.MathF.Abs(b));

    [Fact]
    public void Kron_Backward_ScalarLoss_ProducesCorrectGrads()
    {
        // y = kron(A, B), L = Σ y. dL/dA[i,j] = Σ_{k,l} B[k,l] = sum(B).
        //                          dL/dB[k,l] = Σ_{i,j} A[i,j] = sum(A).
        var a = T(new[] { 1f, 2f, 3f, 4f }, 2, 2);
        var b = T(new[] { 0.5f, 1f, 1.5f, 2f }, 2, 2);

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorKron(a, b);
        var loss = E.ReduceSum(y, null);
        var grads = tape.ComputeGradients(loss, new[] { a, b });

        float sumB = 0.5f + 1f + 1.5f + 2f;  // 5
        float sumA = 1f + 2f + 3f + 4f;      // 10
        var ga = grads[a];
        var gb = grads[b];
        for (int i = 0; i < 4; i++) Assert.True(Close(sumB, ga.AsSpan()[i]));
        for (int i = 0; i < 4; i++) Assert.True(Close(sumA, gb.AsSpan()[i]));
    }

    [Fact]
    public void Kron_Backward_WeightedLoss_ProducesCorrectGrads()
    {
        // L = y[0,0] = A[0,0] · B[0,0] → dL/dA[0,0] = B[0,0], dL/dB[0,0] = A[0,0].
        var a = T(new[] { 2f, 0f, 0f, 0f }, 2, 2);
        var b = T(new[] { 3f, 0f, 0f, 0f }, 2, 2);

        using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });
        var y = E.TensorKron(a, b);
        // Extract y[0,0] via slice... simpler: weighted sum with 1 at [0,0], 0 elsewhere.
        var mask = T(new float[16], 4, 4);
        mask[0, 0] = 1f;
        var weighted = E.TensorMultiply(y, mask);
        var loss = E.ReduceSum(weighted, null);
        var grads = tape.ComputeGradients(loss, new[] { a, b });
        // dL/dA[0,0] = B[0,0] = 3; dL/dB[0,0] = A[0,0] = 2.
        Assert.True(Close(3f, grads[a][0, 0]));
        Assert.True(Close(2f, grads[b][0, 0]));
    }
}
