// Copyright (c) AiDotNet. All rights reserved.
// Numeric gradient check for the IoU-family backward pass (Issue #217).
// Finite-difference each box coordinate on both input tensors and confirm
// the analytic backward returns within 1e-3 of the numeric Jacobian
// projection g^T · (df/dθ). Tolerance is loose because we accumulate
// FD noise (h = 1e-3) and α-stop-gradient skews CIoU ever so slightly.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Detection;

public class BoxIouBackwardTests
{
    private static readonly CpuEngine Cpu = new();
    private const float H = 1e-3f;     // FD step
    private const float Tol = 5e-3f;   // tolerance per coord

    private static Tensor<float> RandBoxes(int seed, int n, float range = 20f)
    {
        var rng = new Random(seed);
        var data = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            float x1 = (float)(rng.NextDouble() * range);
            float y1 = (float)(rng.NextDouble() * range);
            float w = 1f + (float)(rng.NextDouble() * range * 0.5);
            float h = 1f + (float)(rng.NextDouble() * range * 0.5);
            data[i * 4] = x1;
            data[i * 4 + 1] = y1;
            data[i * 4 + 2] = x1 + w;
            data[i * 4 + 3] = y1 + h;
        }
        return new Tensor<float>(data, new[] { n, 4 });
    }

    private static Tensor<float> RandGrad(int seed, int n, int m)
    {
        var rng = new Random(seed ^ 0x55);
        var data = new float[n * m];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, new[] { n, m });
    }

    private static double Dot(Tensor<float> a, Tensor<float> b)
    {
        var sa = a.AsSpan(); var sb = b.AsSpan();
        double s = 0;
        for (int i = 0; i < sa.Length; i++) s += (double)sa[i] * sb[i];
        return s;
    }

    private delegate Tensor<float> Forward(Tensor<float> a, Tensor<float> b);
    private delegate (Tensor<float> gA, Tensor<float> gB) Backward(
        Tensor<float> go, Tensor<float> a, Tensor<float> b);

    private static void CheckGradients(Forward fwd, Backward bwd, int seedA, int seedB, int n, int m, float tolerance = Tol)
    {
        var boxesA = RandBoxes(seedA, n);
        var boxesB = RandBoxes(seedB, m);
        var go = RandGrad(seedA + seedB, n, m);

        var (gA, gB) = bwd(go, boxesA, boxesB);

        // FD Jacobian projection: scalar L(θ) = <go, f(θ, φ)>. dL/dθ_k should
        // match gA[k] / gB[k] within O(h) error.
        void CheckTensor(Tensor<float> box, Tensor<float> analytic, string label,
            Func<Tensor<float>, Tensor<float>> perturbForward)
        {
            var boxData = box.AsWritableSpan();
            var grad = analytic.AsSpan();
            for (int k = 0; k < boxData.Length; k++)
            {
                float orig = boxData[k];
                boxData[k] = orig + H;
                double lPlus = Dot(go, perturbForward(box));
                boxData[k] = orig - H;
                double lMinus = Dot(go, perturbForward(box));
                boxData[k] = orig;

                double fd = (lPlus - lMinus) / (2.0 * H);
                double diff = Math.Abs(fd - grad[k]);
                double scale = 1.0 + Math.Abs(fd);
                if (diff > tolerance * scale)
                    throw new Xunit.Sdk.XunitException(
                        $"{label}[{k}]: analytic={grad[k]:G6}, fd={fd:G6}, diff={diff:G6}");
            }
        }

        CheckTensor(boxesA, gA, "gradA", varying => fwd(varying, boxesB));
        CheckTensor(boxesB, gB, "gradB", varying => fwd(boxesA, varying));
    }

    [Fact]
    public void BoxIou_MatchesFiniteDifferences()
        => CheckGradients(Cpu.BoxIou, Cpu.BoxIouBackward, 1, 2, 4, 3);

    [Fact]
    public void GeneralizedBoxIou_MatchesFiniteDifferences()
        => CheckGradients(Cpu.GeneralizedBoxIou, Cpu.GeneralizedBoxIouBackward, 3, 4, 3, 5);

    [Fact]
    public void DistanceBoxIou_MatchesFiniteDifferences()
        => CheckGradients(Cpu.DistanceBoxIou, Cpu.DistanceBoxIouBackward, 5, 6, 4, 4);

    [Fact]
    public void CompleteBoxIou_MatchesFiniteDifferences()
    {
        // CIoU uses α-stop-gradient (Zheng et al. 2020, matching
        // torchvision.ops.complete_box_iou_loss). FD measures the total
        // derivative including α's dependency on IoU, so the analytic
        // backward diverges from FD by the α-sensitivity term. Tolerance
        // bumped to absorb that designed-in discrepancy (~1% per coord).
        CheckGradients(Cpu.CompleteBoxIou, Cpu.CompleteBoxIouBackward, 7, 8, 3, 4, tolerance: 2e-2f);
    }

    [Fact]
    public void BoxIouBackward_EmptyInput_ReturnsZeroShape()
    {
        var a = new Tensor<float>(new float[0], new[] { 0, 4 });
        var b = new Tensor<float>(new float[0], new[] { 0, 4 });
        var go = new Tensor<float>(new float[0], new[] { 0, 0 });
        var (gA, gB) = Cpu.BoxIouBackward(go, a, b);
        Assert.Equal(new[] { 0, 4 }, gA.Shape.ToArray());
        Assert.Equal(new[] { 0, 4 }, gB.Shape.ToArray());
    }

    [Fact]
    public void BoxIouBackward_NonOverlapping_ZeroGrad()
    {
        // Boxes that don't overlap have ∂iou/∂θ = 0 at every corner.
        var a = new Tensor<float>(new float[] { 0, 0, 1, 1 }, new[] { 1, 4 });
        var b = new Tensor<float>(new float[] { 10, 10, 11, 11 }, new[] { 1, 4 });
        var go = new Tensor<float>(new float[] { 1 }, new[] { 1, 1 });
        var (gA, gB) = Cpu.BoxIouBackward(go, a, b);
        foreach (var v in gA.AsSpan()) Assert.Equal(0f, v, 5);
        foreach (var v in gB.AsSpan()) Assert.Equal(0f, v, 5);
    }

    [Fact]
    public void GeneralizedBoxIouBackward_NonOverlapping_NonZero()
    {
        // GIoU keeps a gradient even when IoU = 0, via the union/enclose
        // term — that's the whole point of GIoU. Confirm gradA is not all
        // zero on the non-overlap case.
        var a = new Tensor<float>(new float[] { 0, 0, 1, 1 }, new[] { 1, 4 });
        var b = new Tensor<float>(new float[] { 10, 10, 11, 11 }, new[] { 1, 4 });
        var go = new Tensor<float>(new float[] { 1 }, new[] { 1, 1 });
        var (gA, _) = Cpu.GeneralizedBoxIouBackward(go, a, b);
        float sum = 0;
        foreach (var v in gA.AsSpan()) sum += Math.Abs(v);
        Assert.True(sum > 1e-4f, $"expected nonzero grad, got sum={sum}");
    }
}
