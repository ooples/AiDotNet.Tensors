using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for IoU/GIoU/DIoU/CIoU loss operations:
/// - Forward correctness (perfect overlap, no overlap, partial overlap)
/// - Gradient existence and finite values
/// - Gradient correctness via finite difference comparison
/// </summary>
public class IoULossTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> MakeBoxes(float[,] data)
    {
        int n = data.GetLength(0);
        var arr = new float[n * 4];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 4; j++)
                arr[i * 4 + j] = data[i, j];
        return new Tensor<float>(arr, new[] { n, 4 });
    }

    // ── Forward correctness ──

    [Fact]
    public void IoULoss_PerfectOverlap_ReturnsZero()
    {
        var boxes = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var loss = _engine.TensorIoULoss(boxes, boxes);
        Assert.Equal(0f, loss[0], 4);
    }

    [Fact]
    public void IoULoss_NoOverlap_ReturnsOne()
    {
        var pred = MakeBoxes(new float[,] { { 0, 0, 5, 5 } });
        var targ = MakeBoxes(new float[,] { { 10, 10, 20, 20 } });
        var loss = _engine.TensorIoULoss(pred, targ);
        // No intersection → IoU = 0 → loss = 1
        Assert.True(loss[0] > 0.99f);
    }

    [Fact]
    public void IoULoss_HalfOverlap()
    {
        // pred [0,0,10,10] area=100, targ [5,0,15,10] area=100
        // intersection [5,0,10,10] area=50, union=150, IoU=1/3
        var pred = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var targ = MakeBoxes(new float[,] { { 5, 0, 15, 10 } });
        var loss = _engine.TensorIoULoss(pred, targ);
        float expectedIoU = 50f / 150f;
        Assert.Equal(1f - expectedIoU, loss[0], 4);
    }

    [Fact]
    public void IoULoss_BatchOfBoxes()
    {
        var pred = MakeBoxes(new float[,] { { 0, 0, 10, 10 }, { 0, 0, 5, 5 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 }, { 10, 10, 20, 20 } });
        var loss = _engine.TensorIoULoss(pred, targ);
        Assert.Equal(2, loss.Length);
        Assert.Equal(0f, loss[0], 4); // perfect overlap
        Assert.True(loss[1] > 0.99f); // no overlap
    }

    [Fact]
    public void GIoULoss_PerfectOverlap_ReturnsZero()
    {
        var boxes = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var loss = _engine.TensorGIoULoss(boxes, boxes);
        Assert.Equal(0f, loss[0], 4);
    }

    [Fact]
    public void GIoULoss_NoOverlap_GreaterThanIoU()
    {
        var pred = MakeBoxes(new float[,] { { 0, 0, 5, 5 } });
        var targ = MakeBoxes(new float[,] { { 10, 10, 20, 20 } });
        var iouLoss = _engine.TensorIoULoss(pred, targ);
        var giouLoss = _engine.TensorGIoULoss(pred, targ);
        // GIoU penalizes distant boxes more → GIoU loss >= IoU loss
        Assert.True(giouLoss[0] >= iouLoss[0] - 0.001f);
    }

    [Fact]
    public void DIoULoss_PerfectOverlap_ReturnsZero()
    {
        var boxes = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var loss = _engine.TensorDIoULoss(boxes, boxes);
        Assert.Equal(0f, loss[0], 4);
    }

    [Fact]
    public void CIoULoss_PerfectOverlap_ReturnsZero()
    {
        var boxes = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var loss = _engine.TensorCIoULoss(boxes, boxes);
        Assert.Equal(0f, loss[0], 4);
    }

    [Fact]
    public void CIoULoss_DifferentAspectRatio_PenalizesMore()
    {
        // Same center/overlap but different aspect ratios
        var pred = MakeBoxes(new float[,] { { 0, 0, 10, 10 } }); // square
        var targ = MakeBoxes(new float[,] { { 0, 0, 20, 5 } });  // wide rectangle
        var diouLoss = _engine.TensorDIoULoss(pred, targ);
        var ciouLoss = _engine.TensorCIoULoss(pred, targ);
        // CIoU adds aspect ratio penalty on top of DIoU
        Assert.True(ciouLoss[0] >= diouLoss[0] - 0.001f);
    }

    // ── Target validation ──

    [Fact]
    public void IoULoss_WrongTargetShape_Throws()
    {
        var pred = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var badTarget = new Tensor<float>(new float[3], new[] { 1, 3 }); // [1,3] not [1,4]
        Assert.Throws<ArgumentException>(() => _engine.TensorIoULoss(pred, badTarget));
    }

    [Fact]
    public void IoULoss_MismatchedBatchSize_Throws()
    {
        var pred = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 }, { 5, 5, 15, 15 } });
        Assert.Throws<ArgumentException>(() => _engine.TensorIoULoss(pred, targ));
    }

    // ── Gradient correctness ──

    [Fact]
    public void IoULoss_GradientsExistAndAreFinite()
    {
        var pred = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();
        var loss = _engine.TensorIoULoss(pred, targ);
        var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalarLoss);

        Assert.True(grads.ContainsKey(pred));
        var grad = grads[pred];
        Assert.Equal(pred.Length, grad.Length);
        for (int i = 0; i < grad.Length; i++)
            Assert.False(float.IsNaN(grad[i]) || float.IsInfinity(grad[i]),
                $"Gradient at index {i} is {grad[i]}");
    }

    [Fact]
    public void IoULoss_GradientApproximatesFiniteDifference()
    {
        var predData = new float[,] { { 2, 2, 8, 8 } };
        var targData = new float[,] { { 0, 0, 10, 10 } };
        var targ = MakeBoxes(targData);
        float eps = 1e-3f;

        // Compute analytical gradient (don't specify sources — let tape find all)
        var pred = MakeBoxes(predData);
        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var loss = _engine.TensorIoULoss(pred, targ);
            var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
            grads = tape.ComputeGradients(scalarLoss);
        }
        Assert.True(grads.ContainsKey(pred), "Gradient not found for predicted tensor");
        var analyticalGrad = grads[pred];

        // Finite difference for each coordinate
        for (int c = 0; c < 4; c++)
        {
            var plusData = (float[,])predData.Clone();
            plusData[0, c] += eps;
            var minusData = (float[,])predData.Clone();
            minusData[0, c] -= eps;

            var lossPlus = _engine.TensorIoULoss(MakeBoxes(plusData), targ);
            var lossMinus = _engine.TensorIoULoss(MakeBoxes(minusData), targ);
            float fdGrad = (lossPlus[0] - lossMinus[0]) / (2 * eps);

            Assert.Equal(fdGrad, analyticalGrad[c], 2);
        }
    }

    [Fact]
    public void GIoULoss_GradientsExistAndAreFinite()
    {
        var pred = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();
        var loss = _engine.TensorGIoULoss(pred, targ);
        var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalarLoss);

        Assert.True(grads.ContainsKey(pred));
        var grad = grads[pred];
        for (int i = 0; i < grad.Length; i++)
            Assert.False(float.IsNaN(grad[i]) || float.IsInfinity(grad[i]));
    }

    [Fact]
    public void DIoULoss_GradientsExistAndAreFinite()
    {
        var pred = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();
        var loss = _engine.TensorDIoULoss(pred, targ);
        var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalarLoss);

        Assert.True(grads.ContainsKey(pred));
        var grad = grads[pred];
        for (int i = 0; i < grad.Length; i++)
            Assert.False(float.IsNaN(grad[i]) || float.IsInfinity(grad[i]));
    }

    [Fact]
    public void CIoULoss_GradientsExistAndAreFinite()
    {
        var pred = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var targ = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();
        var loss = _engine.TensorCIoULoss(pred, targ);
        var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalarLoss);

        Assert.True(grads.ContainsKey(pred));
        var grad = grads[pred];
        for (int i = 0; i < grad.Length; i++)
            Assert.False(float.IsNaN(grad[i]) || float.IsInfinity(grad[i]));
    }
}
