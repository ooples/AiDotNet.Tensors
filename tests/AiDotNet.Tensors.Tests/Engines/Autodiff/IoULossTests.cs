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

    // ── Slice backward sanity check ──

    [Fact]
    public void SliceBackward_GradientFlowsToOriginalTensor()
    {
        var input = MakeBoxes(new float[,] { { 1, 2, 3, 4 } });
        using var tape = new GradientTape<float>();
        var col0 = _engine.TensorSlice(input, new[] { 0, 0 }, new[] { 1, 1 });
        var col1 = _engine.TensorSlice(input, new[] { 0, 1 }, new[] { 1, 1 });
        var sum = _engine.TensorAdd(col0, col1);
        var loss = _engine.ReduceSum(sum, new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss);
        Assert.True(grads.ContainsKey(input), $"Gradient not found for input. Keys: {grads.Count}");
    }

    // ── Gradient correctness ──

    [Fact]
    public void IoULoss_SimpleManualChain_GradientFlows()
    {
        var predicted = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var target = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();
        var px1 = _engine.TensorSlice(predicted, new[] { 0, 0 }, new[] { 1, 1 });
        var px2 = _engine.TensorSlice(predicted, new[] { 0, 2 }, new[] { 1, 1 });
        var width = _engine.TensorSubtract(px2, px1);
        var loss = _engine.ReduceSum(width, new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss);
        Assert.True(grads.ContainsKey(predicted), $"No grad for predicted. Keys: {grads.Count}");
    }

    [Fact]
    public void ScalarMinusTensor_GradientFlows()
    {
        var x = new Tensor<float>(new float[] { 3f }, new[] { 1, 1 });
        using var tape = new GradientTape<float>();
        var y = _engine.ScalarMinusTensor(1f, x);
        var loss = _engine.ReduceSum(y, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(loss);
        Assert.True(grads.ContainsKey(x), $"No grad for x. Keys: {grads.Count}");
        // d(1-x)/dx = -1
        Assert.Equal(-1f, grads[x][0], 4);
    }

    [Fact]
    public void DivideAfterScalarMinus_GradientFlows()
    {
        var a = new Tensor<float>(new float[] { 6f }, new[] { 1, 1 });
        var b = new Tensor<float>(new float[] { 10f }, new[] { 1, 1 });
        using var tape = new GradientTape<float>();
        var ratio = _engine.TensorDivide(a, b);
        var y = _engine.ScalarMinusTensor(1f, ratio);
        var loss = _engine.ReduceSum(y, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(loss);
        Assert.True(grads.ContainsKey(a), $"No grad for a. Keys: {grads.Count}");
        // d(1 - a/b)/da = -1/b = -0.1
        Assert.Equal(-0.1f, grads[a][0], 3);
    }

    [Fact]
    public void IoUChain_DivideScalarMinusReduceSum_GradientFlows()
    {
        // Minimal IoU-like chain: iou = a/b, loss = 1-iou, scalar = sum(loss)
        var a = new Tensor<float>(new float[] { 36f }, new[] { 1, 1 });
        var b = new Tensor<float>(new float[] { 100f }, new[] { 1, 1 });
        using var tape = new GradientTape<float>();
        var iou = _engine.TensorDivide(a, b);
        var loss = _engine.ScalarMinusTensor(1f, iou);
        var scalar = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalar);

        // d(1 - a/b)/da = -1/b = -0.01
        Assert.True(grads.ContainsKey(a), $"No grad for a");
        Assert.Equal(-0.01f, grads[a][0], 3);
    }

    [Fact]
    public void IoUChain_SliceThroughMultiplyDivide_GradientFlows()
    {
        var predicted = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        using var tape = new GradientTape<float>();
        var px1 = _engine.TensorSlice(predicted, new[] { 0, 0 }, new[] { 1, 1 });
        var px2 = _engine.TensorSlice(predicted, new[] { 0, 2 }, new[] { 1, 1 });
        // width = px2 - px1 = 8
        var width = _engine.TensorSubtract(px2, px1);
        // Test: does gradient flow through multiply?
        var area = _engine.TensorMultiply(width, width);
        var loss = _engine.ReduceSum(area, new[] { 0, 1 }, keepDims: false);
        var grads = tape.ComputeGradients(loss);
        // Check each intermediate
        Assert.True(grads.ContainsKey(area), $"No grad for area");
        Assert.True(grads.ContainsKey(width), $"No grad for width. area grad exists={grads.ContainsKey(area)}");
        Assert.True(grads.ContainsKey(px1), $"No grad for px1. width grad exists={grads.ContainsKey(width)}");
        Assert.True(grads.ContainsKey(predicted), $"No grad for predicted. px1 grad exists={grads.ContainsKey(px1)}");
        // d(w^2)/dpx1 = 2w * (-1) = 2*8*(-1) = -16
        Assert.Equal(-16f, grads[predicted][0], 1); // px1
        Assert.Equal(0f, grads[predicted][1], 1);    // py1 (not used)
        Assert.Equal(16f, grads[predicted][2], 1);   // px2
        Assert.Equal(0f, grads[predicted][3], 1);    // py2 (not used)
    }

    [Fact]
    public void IoULoss_FullManualIoU_GradientFlows()
    {
        // Replicate the full IoU composition to find where gradient chain breaks
        var predicted = MakeBoxes(new float[,] { { 1, 1, 9, 9 } });
        var target = MakeBoxes(new float[,] { { 0, 0, 10, 10 } });

        using var tape = new GradientTape<float>();

        var px1 = _engine.TensorSlice(predicted, new[] { 0, 0 }, new[] { 1, 1 });
        var py1 = _engine.TensorSlice(predicted, new[] { 0, 1 }, new[] { 1, 1 });
        var px2 = _engine.TensorSlice(predicted, new[] { 0, 2 }, new[] { 1, 1 });
        var py2 = _engine.TensorSlice(predicted, new[] { 0, 3 }, new[] { 1, 1 });

        var tx1 = _engine.TensorSlice(target, new[] { 0, 0 }, new[] { 1, 1 });
        var ty1 = _engine.TensorSlice(target, new[] { 0, 1 }, new[] { 1, 1 });
        var tx2 = _engine.TensorSlice(target, new[] { 0, 2 }, new[] { 1, 1 });
        var ty2 = _engine.TensorSlice(target, new[] { 0, 3 }, new[] { 1, 1 });

        // Intersection
        var interX1 = _engine.TensorMax(px1, tx1);
        var interY1 = _engine.TensorMax(py1, ty1);
        var interX2 = _engine.TensorNegate(_engine.TensorMax(_engine.TensorNegate(px2), _engine.TensorNegate(tx2)));
        var interY2 = _engine.TensorNegate(_engine.TensorMax(_engine.TensorNegate(py2), _engine.TensorNegate(ty2)));
        var interW = _engine.ReLU(_engine.TensorSubtract(interX2, interX1));
        var interH = _engine.ReLU(_engine.TensorSubtract(interY2, interY1));
        var interArea = _engine.TensorMultiply(interW, interH);

        // Areas
        var predW = _engine.ReLU(_engine.TensorSubtract(px2, px1));
        var predH = _engine.ReLU(_engine.TensorSubtract(py2, py1));
        var predArea = _engine.TensorMultiply(predW, predH);
        var targW = _engine.ReLU(_engine.TensorSubtract(tx2, tx1));
        var targH = _engine.ReLU(_engine.TensorSubtract(ty2, ty1));
        var targArea = _engine.TensorMultiply(targW, targH);
        var unionArea = _engine.TensorAddScalar(_engine.TensorSubtract(_engine.TensorAdd(predArea, targArea), interArea), 1e-7f);

        // IoU
        var iou = _engine.TensorDivide(interArea, unionArea);
        var loss = _engine.ScalarMinusTensor(1f, iou);
        var scalarLoss = _engine.ReduceSum(loss, new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(scalarLoss);

        Assert.True(grads.ContainsKey(predicted),
            $"No grad for predicted. Grads has {grads.Count} keys. " +
            $"Has px1={grads.ContainsKey(px1)}, predW={grads.ContainsKey(predW)}, predArea={grads.ContainsKey(predArea)}, " +
            $"iou={grads.ContainsKey(iou)}, interArea={grads.ContainsKey(interArea)}, interX1={grads.ContainsKey(interX1)}");
        // Verify gradient value for px1 coordinate (index 0 in [1,4] tensor)
        var predGrad = grads[predicted];
        Assert.False(predGrad[0] == 0f, $"px1 grad is zero but should be ~0.06. Full grad: [{predGrad[0]},{predGrad[1]},{predGrad[2]},{predGrad[3]}]");
    }

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
        // Debug: check what keys exist
        if (!grads.ContainsKey(pred))
        {
            var keyShapes = string.Join(", ", grads.Keys.Select(k => $"{k.Shape[0]}x{(k.Shape.Length > 1 ? k.Shape[1].ToString() : "?")}"));
            Assert.Fail($"Gradient not found for predicted tensor (shape {pred.Shape[0]}x{pred.Shape[1]}). " +
                        $"Dict has {grads.Count} keys: [{keyShapes}]");
        }
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

            Assert.True(Math.Abs(fdGrad - analyticalGrad[c]) < 0.05f,
                $"Coord {c}: fd={fdGrad:F6}, analytical={analyticalGrad[c]:F6}, diff={Math.Abs(fdGrad - analyticalGrad[c]):F6}");
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
