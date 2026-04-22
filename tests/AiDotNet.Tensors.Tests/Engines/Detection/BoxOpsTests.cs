using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Detection;

/// <summary>
/// Reference-value tests for the Vision Detection family (Issue #217):
/// BoxConvert, BoxArea, BoxIoU, GeneralizedBoxIoU, DistanceBoxIoU,
/// CompleteBoxIoU, NMS, BatchedNMS, MasksToBoxes. Reference values are
/// computed from the documented torchvision formulae and cross-checked
/// by hand for each input.
/// </summary>
public class BoxOpsTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private const double Tol = 1e-6;

    // ---- BoxConvert -----------------------------------------------------

    [Fact]
    public void BoxConvert_XYXY_To_XYWH_RoundTrips()
    {
        // Shape [3, 4] — three boxes, all formats round-trip exactly.
        var xyxy = new Tensor<double>(new double[]
        {
            10, 20, 30, 50,    // x1=10, y1=20, x2=30, y2=50
             0,  0, 100, 80,
            -5, -3,  5, 17,
        }, new[] { 3, 4 });

        var xywh = _engine.BoxConvert(xyxy, BoxFormat.XYXY, BoxFormat.XYWH);
        var s = xywh.AsSpan();
        Assert.Equal(10, s[0]); Assert.Equal(20, s[1]); Assert.Equal(20, s[2]); Assert.Equal(30, s[3]);
        Assert.Equal( 0, s[4]); Assert.Equal( 0, s[5]); Assert.Equal(100, s[6]); Assert.Equal(80, s[7]);
        Assert.Equal(-5, s[8]); Assert.Equal(-3, s[9]); Assert.Equal(10, s[10]); Assert.Equal(20, s[11]);

        var roundTrip = _engine.BoxConvert(xywh, BoxFormat.XYWH, BoxFormat.XYXY);
        var rt = roundTrip.AsSpan();
        for (int i = 0; i < 12; i++) Assert.Equal(xyxy.AsSpan()[i], rt[i]);
    }

    [Fact]
    public void BoxConvert_XYXY_To_CXCYWH_KnownValues()
    {
        var xyxy = new Tensor<double>(new double[] { 10, 20, 30, 50 }, new[] { 1, 4 });
        var c = _engine.BoxConvert(xyxy, BoxFormat.XYXY, BoxFormat.CXCYWH).AsSpan();
        Assert.Equal(20, c[0]);  // cx = (10+30)/2
        Assert.Equal(35, c[1]);  // cy = (20+50)/2
        Assert.Equal(20, c[2]);  // w
        Assert.Equal(30, c[3]);  // h
    }

    [Fact]
    public void BoxConvert_AllRoundTrips_CXCYWH()
    {
        var xyxy = new Tensor<double>(new double[] { 5, 7, 25, 17 }, new[] { 1, 4 });
        var cxcywh = _engine.BoxConvert(xyxy, BoxFormat.XYXY, BoxFormat.CXCYWH);
        var back = _engine.BoxConvert(cxcywh, BoxFormat.CXCYWH, BoxFormat.XYXY);
        var orig = xyxy.AsSpan();
        var b = back.AsSpan();
        for (int i = 0; i < 4; i++) Assert.True(Math.Abs(orig[i] - b[i]) < Tol);
    }

    // ---- BoxArea --------------------------------------------------------

    [Fact]
    public void BoxArea_KnownValues()
    {
        var boxes = new Tensor<double>(new double[]
        {
            10, 20, 30, 50,    // 20 × 30 = 600
             0,  0,  1,  1,    // 1
             5,  5,  5,  5,    // 0 (degenerate)
            10, 10,  0,  0,    // 0 (clamped — negative w/h)
        }, new[] { 4, 4 });
        var area = _engine.BoxArea(boxes).AsSpan();
        Assert.Equal(600, area[0]);
        Assert.Equal(  1, area[1]);
        Assert.Equal(  0, area[2]);
        Assert.Equal(  0, area[3]);
    }

    // ---- BoxIoU ---------------------------------------------------------

    [Fact]
    public void BoxIou_Identity_IsOnes()
    {
        var boxes = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,
            5, 5, 15, 15,
        }, new[] { 2, 4 });
        var iou = _engine.BoxIou(boxes, boxes);
        Assert.Equal(new[] { 2, 2 }, iou._shape);
        var s = iou.AsSpan();
        Assert.True(Math.Abs(s[0] - 1.0) < Tol);  // self
        Assert.True(Math.Abs(s[3] - 1.0) < Tol);  // self
    }

    [Fact]
    public void BoxIou_HalfOverlap_Quarter()
    {
        // Two 10×10 boxes overlap on a 5×10 strip.
        // Intersection = 5*10 = 50. Each area = 100. Union = 100 + 100 − 50 = 150.
        // IoU = 50/150 = 1/3.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 5, 0, 15, 10 }, new[] { 1, 4 });
        var iou = _engine.BoxIou(a, b).AsSpan();
        Assert.True(Math.Abs(iou[0] - (1.0 / 3.0)) < Tol);
    }

    [Fact]
    public void BoxIou_NoOverlap_Zero()
    {
        var a = new Tensor<double>(new double[] { 0, 0, 5, 5 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 100, 100, 110, 110 }, new[] { 1, 4 });
        Assert.Equal(0.0, _engine.BoxIou(a, b).AsSpan()[0]);
    }

    [Fact]
    public void BoxIou_TouchingEdge_Zero()
    {
        // Boxes share an edge but no area.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 10, 0, 20, 10 }, new[] { 1, 4 });
        Assert.Equal(0.0, _engine.BoxIou(a, b).AsSpan()[0]);
    }

    [Fact]
    public void BoxIou_OneInsideOther_RatioOfAreas()
    {
        // Inner 5×5 fully inside outer 10×10.
        // Intersection = 25. Union = 100 (outer dominates). IoU = 0.25.
        var inner = new Tensor<double>(new double[] { 2, 2, 7, 7 }, new[] { 1, 4 });
        var outer = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        Assert.True(Math.Abs(_engine.BoxIou(inner, outer).AsSpan()[0] - 0.25) < Tol);
    }

    [Fact]
    public void BoxIou_PairwiseShape_NxM()
    {
        var a = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,
            5, 5, 15, 15,
            -5, -5, 5, 5,
        }, new[] { 3, 4 });
        var b = new Tensor<double>(new double[]
        {
            0, 0, 20, 20,
            8, 8, 12, 12,
        }, new[] { 2, 4 });
        var iou = _engine.BoxIou(a, b);
        Assert.Equal(new[] { 3, 2 }, iou._shape);
    }

    // ---- Generalized / Distance / Complete BoxIoU -----------------------

    [Fact]
    public void GeneralizedBoxIou_NonOverlapping_Negative()
    {
        // Two 10×10 boxes far apart. IoU = 0. GIoU < 0 because the
        // enclosing box is much larger than the union.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 30, 30, 40, 40 }, new[] { 1, 4 });
        // enclose = (40-0)*(40-0) = 1600. union = 200.
        // GIoU = 0 - (1600 - 200)/1600 = -0.875
        double giou = _engine.GeneralizedBoxIou(a, b).AsSpan()[0];
        Assert.True(Math.Abs(giou - (-0.875)) < Tol, $"expected -0.875, got {giou}");
    }

    [Fact]
    public void GeneralizedBoxIou_Identical_OneAndOnly()
    {
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        Assert.True(Math.Abs(_engine.GeneralizedBoxIou(a, a).AsSpan()[0] - 1.0) < Tol);
    }

    [Fact]
    public void DistanceBoxIou_Identical_OneAndOnly()
    {
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        Assert.True(Math.Abs(_engine.DistanceBoxIou(a, a).AsSpan()[0] - 1.0) < Tol);
    }

    [Fact]
    public void CompleteBoxIou_Identical_OneAndOnly()
    {
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        Assert.True(Math.Abs(_engine.CompleteBoxIou(a, a).AsSpan()[0] - 1.0) < Tol);
    }

    [Fact]
    public void DistanceBoxIou_OffsetByCentre_LowersIoU()
    {
        // Two 10×10 boxes offset so centres are 5 apart along x.
        var a = new Tensor<double>(new double[] { 0, 0, 10, 10 }, new[] { 1, 4 });
        var b = new Tensor<double>(new double[] { 5, 0, 15, 10 }, new[] { 1, 4 });
        double iou = _engine.BoxIou(a, b).AsSpan()[0];
        double diou = _engine.DistanceBoxIou(a, b).AsSpan()[0];
        // DIoU = IoU − ρ²/c² where ρ²=25, c²=15²+10²=325, so subtract 25/325 ≈ 0.0769.
        Assert.True(diou < iou);
        Assert.True(Math.Abs(diou - (iou - 25.0 / 325.0)) < Tol,
            $"DIoU expected {iou - 25.0/325.0}, got {diou}");
    }

    // ---- NMS ------------------------------------------------------------

    [Fact]
    public void Nms_KeepsHighestThenNonOverlapping()
    {
        // Three boxes:
        // 0: low score, low IoU with #1.
        // 1: high score (kept first).
        // 2: high IoU with #1 → suppressed.
        var boxes = new Tensor<double>(new double[]
        {
            100, 100, 110, 110,   // box 0 — far away
              0,   0,  10,  10,   // box 1 — kept
              1,   1,   9,   9,   // box 2 — overlaps box 1, lower score
        }, new[] { 3, 4 });
        var scores = new Tensor<double>(new double[] { 0.5, 0.9, 0.8 }, new[] { 3 });
        var keep = _engine.Nms(boxes, scores, iouThreshold: 0.5).AsSpan();

        Assert.Equal(2, keep.Length);
        Assert.Equal(1, keep[0]);   // score 0.9 first
        Assert.Equal(0, keep[1]);   // score 0.5, no overlap with #1
    }

    [Fact]
    public void Nms_HighThreshold_KeepsAll()
    {
        var boxes = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,
            1, 1,  9,  9,
            5, 5, 15, 15,
        }, new[] { 3, 4 });
        var scores = new Tensor<double>(new double[] { 0.9, 0.5, 0.7 }, new[] { 3 });
        var keep = _engine.Nms(boxes, scores, iouThreshold: 0.99).AsSpan();
        Assert.Equal(3, keep.Length);
        // Score-descending order: 0.9, 0.7, 0.5 → indices 0, 2, 1.
        Assert.Equal(0, keep[0]);
        Assert.Equal(2, keep[1]);
        Assert.Equal(1, keep[2]);
    }

    [Fact]
    public void Nms_LowThreshold_KeepsOnlyTop()
    {
        var boxes = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,
            0, 0, 11, 11,    // huge IoU with box 0
            0, 0,  9,  9,    // huge IoU with box 0
        }, new[] { 3, 4 });
        var scores = new Tensor<double>(new double[] { 0.9, 0.5, 0.4 }, new[] { 3 });
        var keep = _engine.Nms(boxes, scores, iouThreshold: 0.1).AsSpan();
        Assert.Single(keep.ToArray());
        Assert.Equal(0, keep[0]);
    }

    [Fact]
    public void Nms_EmptyInput_ReturnsEmpty()
    {
        var boxes = new Tensor<double>(new[] { 0, 4 });
        var scores = new Tensor<double>(new[] { 0 });
        var keep = _engine.Nms(boxes, scores, 0.5);
        Assert.Equal(0, keep._shape[0]);
    }

    // ---- BatchedNMS -----------------------------------------------------

    [Fact]
    public void BatchedNms_PerClassSuppression()
    {
        // Two highly-overlapping boxes in two classes — both must survive
        // because BatchedNMS only suppresses within a class.
        var boxes = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,    // class 0
            1, 1,  9,  9,    // class 1 — would be suppressed by global NMS
        }, new[] { 2, 4 });
        var scores = new Tensor<double>(new double[] { 0.9, 0.8 }, new[] { 2 });
        var classes = new Tensor<int>(new[] { 0, 1 }, new[] { 2 });
        var keep = _engine.BatchedNms(boxes, scores, classes, 0.5).AsSpan();
        Assert.Equal(2, keep.Length);
    }

    [Fact]
    public void BatchedNms_SameClass_SuppressedNormally()
    {
        var boxes = new Tensor<double>(new double[]
        {
            0, 0, 10, 10,
            1, 1,  9,  9,
        }, new[] { 2, 4 });
        var scores = new Tensor<double>(new double[] { 0.9, 0.8 }, new[] { 2 });
        var classes = new Tensor<int>(new[] { 0, 0 }, new[] { 2 });
        var keep = _engine.BatchedNms(boxes, scores, classes, 0.5).AsSpan();
        Assert.Single(keep.ToArray());
        Assert.Equal(0, keep[0]);
    }

    // ---- MasksToBoxes ---------------------------------------------------

    [Fact]
    public void MasksToBoxes_TightBoundingBox()
    {
        // Two 4×4 masks. Mask 0 has a single pixel at (1, 2). Mask 1 has
        // a 2×3 rectangle from (1,1)-(2,3).
        var data = new double[2 * 4 * 4];
        // mask 0: (y=1, x=2) only.
        data[0 * 16 + 1 * 4 + 2] = 1;
        // mask 1: rect rows y∈{1,2}, cols x∈{1,2,3}.
        for (int y = 1; y <= 2; y++)
            for (int x = 1; x <= 3; x++)
                data[1 * 16 + y * 4 + x] = 1;
        var masks = new Tensor<double>(data, new[] { 2, 4, 4 });
        var boxes = _engine.MasksToBoxes(masks).AsSpan();

        // mask 0: point (x=2, y=1) → bbox [2, 1, 2, 1].
        Assert.Equal(2, boxes[0]); Assert.Equal(1, boxes[1]);
        Assert.Equal(2, boxes[2]); Assert.Equal(1, boxes[3]);

        // mask 1: x∈[1,3], y∈[1,2] → bbox [1, 1, 3, 2].
        Assert.Equal(1, boxes[4]); Assert.Equal(1, boxes[5]);
        Assert.Equal(3, boxes[6]); Assert.Equal(2, boxes[7]);
    }

    [Fact]
    public void MasksToBoxes_EmptyMask_AllZeros()
    {
        var masks = new Tensor<double>(new double[1 * 4 * 4], new[] { 1, 4, 4 });
        var b = _engine.MasksToBoxes(masks).AsSpan();
        for (int i = 0; i < 4; i++) Assert.Equal(0, b[i]);
    }
}
