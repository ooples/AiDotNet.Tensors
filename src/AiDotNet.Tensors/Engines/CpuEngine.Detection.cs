using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Encoding for axis-aligned bounding boxes. The four channels of the
/// last axis carry different quantities depending on the format.
/// </summary>
public enum BoxFormat
{
    /// <summary>
    /// <c>[x1, y1, x2, y2]</c> — top-left and bottom-right corners.
    /// torchvision's default.
    /// </summary>
    XYXY,
    /// <summary>
    /// <c>[x, y, w, h]</c> — top-left corner and size.
    /// </summary>
    XYWH,
    /// <summary>
    /// <c>[cx, cy, w, h]</c> — centre point and size. Common for YOLO
    /// and DETR exports.
    /// </summary>
    CXCYWH,
}

/// <summary>
/// Vision Detection ops: bounding-box arithmetic, NMS, mask → box.
/// All operate on contiguous <c>Tensor&lt;T&gt;</c> with the last axis = 4.
/// Implementations are CPU-only here; backends override as needed.
/// Issue #217.
/// </summary>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> BoxConvert<T>(Tensor<T> boxes, BoxFormat from, BoxFormat to)
    {
        ValidateBoxes(boxes, nameof(boxes));
        // CXCYWH uses ½ in element type; integral T silently truncates to 0.
        if (from == BoxFormat.CXCYWH || to == BoxFormat.CXCYWH)
            RequireFloatingPoint<T>(nameof(BoxConvert));
        if (from == to) return boxes.Clone();

        var ops = MathHelper.GetNumericOperations<T>();
        int n = boxes.Length / 4;
        var result = new Tensor<T>(boxes._shape);
        var src = boxes.AsSpan();
        var dst = result.AsWritableSpan();
        T half = ops.Divide(ops.One, ops.FromDouble(2.0));

        for (int i = 0; i < n; i++)
        {
            int o = i * 4;
            // First normalise to xyxy.
            T x1, y1, x2, y2;
            switch (from)
            {
                case BoxFormat.XYXY:
                    x1 = src[o]; y1 = src[o + 1]; x2 = src[o + 2]; y2 = src[o + 3];
                    break;
                case BoxFormat.XYWH:
                    x1 = src[o]; y1 = src[o + 1];
                    x2 = ops.Add(x1, src[o + 2]);
                    y2 = ops.Add(y1, src[o + 3]);
                    break;
                case BoxFormat.CXCYWH:
                    T cx = src[o], cy = src[o + 1], w = src[o + 2], h = src[o + 3];
                    T hw = ops.Multiply(w, half);
                    T hh = ops.Multiply(h, half);
                    x1 = ops.Subtract(cx, hw);
                    y1 = ops.Subtract(cy, hh);
                    x2 = ops.Add(cx, hw);
                    y2 = ops.Add(cy, hh);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(from));
            }
            // Then encode into requested format.
            switch (to)
            {
                case BoxFormat.XYXY:
                    dst[o] = x1; dst[o + 1] = y1; dst[o + 2] = x2; dst[o + 3] = y2;
                    break;
                case BoxFormat.XYWH:
                    dst[o] = x1; dst[o + 1] = y1;
                    dst[o + 2] = ops.Subtract(x2, x1);
                    dst[o + 3] = ops.Subtract(y2, y1);
                    break;
                case BoxFormat.CXCYWH:
                    T wOut = ops.Subtract(x2, x1);
                    T hOut = ops.Subtract(y2, y1);
                    dst[o] = ops.Add(x1, ops.Multiply(wOut, half));
                    dst[o + 1] = ops.Add(y1, ops.Multiply(hOut, half));
                    dst[o + 2] = wOut;
                    dst[o + 3] = hOut;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(to));
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> BoxArea<T>(Tensor<T> boxes)
    {
        ValidateBoxes(boxes, nameof(boxes));
        var ops = MathHelper.GetNumericOperations<T>();
        int n = boxes.Length / 4;
        // Output shape is the input shape minus the trailing 4.
        var outShape = boxes._shape.Take(boxes.Rank - 1).ToArray();
        if (outShape.Length == 0) outShape = new[] { 1 };
        var result = new Tensor<T>(outShape);
        var src = boxes.AsSpan();
        var dst = result.AsWritableSpan();
        T zero = ops.Zero;
        for (int i = 0; i < n; i++)
        {
            T w = ops.Subtract(src[i * 4 + 2], src[i * 4 + 0]);
            T h = ops.Subtract(src[i * 4 + 3], src[i * 4 + 1]);
            // torchvision clamps negative widths/heights at zero so
            // degenerate boxes don't contribute negative area to IoU
            // calculations.
            if (ops.LessThan(w, zero)) w = zero;
            if (ops.LessThan(h, zero)) h = zero;
            dst[i] = ops.Multiply(w, h);
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> BoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        ValidateBoxes(boxesA, nameof(boxesA));
        ValidateBoxes(boxesB, nameof(boxesB));
        if (boxesA.Rank != 2 || boxesB.Rank != 2)
            throw new ArgumentException("BoxIou requires rank-2 boxes [N, 4] and [M, 4].");
        var (iou, _, _, _) = ComputePairwiseIoU(boxesA, boxesB);
        DifferentiableOps.RecordBinary("BoxIou", iou, boxesA, boxesB, BackwardFunctions<T>.BoxIouBackward);
        return iou;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GeneralizedBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        ValidateBoxes(boxesA, nameof(boxesA));
        ValidateBoxes(boxesB, nameof(boxesB));
        if (boxesA.Rank != 2 || boxesB.Rank != 2)
            throw new ArgumentException("GeneralizedBoxIou requires rank-2 boxes [N, 4] and [M, 4].");
        var ops = MathHelper.GetNumericOperations<T>();
        var (iou, union, _, _) = ComputePairwiseIoU(boxesA, boxesB);
        // GIoU = IoU − (|enclosing| − |union|) / |enclosing|.
        var a = boxesA.AsSpan();
        var b = boxesB.AsSpan();
        int N = boxesA._shape[0], M = boxesB._shape[0];
        var iouSpan = iou.AsWritableSpan();
        var unionSpan = union.AsSpan();
        for (int i = 0; i < N; i++)
        {
            T ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
            for (int j = 0; j < M; j++)
            {
                T bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];
                T ex1 = ops.LessThan(ax1, bx1) ? ax1 : bx1;
                T ey1 = ops.LessThan(ay1, by1) ? ay1 : by1;
                T ex2 = ops.LessThan(ax2, bx2) ? bx2 : ax2;
                T ey2 = ops.LessThan(ay2, by2) ? by2 : ay2;
                T ew = ops.Subtract(ex2, ex1);
                T eh = ops.Subtract(ey2, ey1);
                T enclose = ops.Multiply(ew, eh);
                int idx = i * M + j;
                T u = unionSpan[idx];
                if (ops.GreaterThan(enclose, ops.Zero))
                {
                    T penalty = ops.Divide(ops.Subtract(enclose, u), enclose);
                    iouSpan[idx] = ops.Subtract(iouSpan[idx], penalty);
                }
            }
        }
        DifferentiableOps.RecordBinary("GeneralizedBoxIou", iou, boxesA, boxesB,
            BackwardFunctions<T>.GeneralizedBoxIouBackward);
        return iou;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> DistanceBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        var result = DiouLikeImpl(boxesA, boxesB, includeAspect: false);
        DifferentiableOps.RecordBinary("DistanceBoxIou", result, boxesA, boxesB,
            BackwardFunctions<T>.DistanceBoxIouBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> CompleteBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        var result = DiouLikeImpl(boxesA, boxesB, includeAspect: true);
        DifferentiableOps.RecordBinary("CompleteBoxIou", result, boxesA, boxesB,
            BackwardFunctions<T>.CompleteBoxIouBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<int> Nms<T>(Tensor<T> boxes, Tensor<T> scores, double iouThreshold)
    {
        ValidateBoxes(boxes, nameof(boxes));
        if (boxes.Rank != 2) throw new ArgumentException("Nms requires rank-2 boxes [N, 4].");
        if (scores.Rank != 1 || scores._shape[0] != boxes._shape[0])
            throw new ArgumentException("scores must be rank-1 with length N.");

        int n = boxes._shape[0];
        if (n == 0) return new Tensor<int>(new[] { 0 });

        var ops = MathHelper.GetNumericOperations<T>();
        var b = boxes.AsSpan();
        var s = scores.AsSpan();

        // Sort indices by score descending — Array.Sort with custom comparer
        // is allocation-free for int[] keys.
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        var scoreCopy = new double[n];
        for (int i = 0; i < n; i++) scoreCopy[i] = ops.ToDouble(s[i]);
        Array.Sort(order, (x, y) => scoreCopy[y].CompareTo(scoreCopy[x]));

        // Pre-compute areas so we avoid repeating the subtraction in the
        // inner loop.
        var areas = new double[n];
        for (int i = 0; i < n; i++)
        {
            double w = ops.ToDouble(b[i * 4 + 2]) - ops.ToDouble(b[i * 4]);
            double h = ops.ToDouble(b[i * 4 + 3]) - ops.ToDouble(b[i * 4 + 1]);
            areas[i] = (w > 0 && h > 0) ? w * h : 0;
        }

        var keep = new System.Collections.Generic.List<int>(n);
        var suppressed = new bool[n];
        for (int oi = 0; oi < n; oi++)
        {
            int i = order[oi];
            if (suppressed[i]) continue;
            keep.Add(i);
            double ix1 = ops.ToDouble(b[i * 4]), iy1 = ops.ToDouble(b[i * 4 + 1]);
            double ix2 = ops.ToDouble(b[i * 4 + 2]), iy2 = ops.ToDouble(b[i * 4 + 3]);
            double ai = areas[i];

            for (int oj = oi + 1; oj < n; oj++)
            {
                int j = order[oj];
                if (suppressed[j]) continue;
                double jx1 = ops.ToDouble(b[j * 4]), jy1 = ops.ToDouble(b[j * 4 + 1]);
                double jx2 = ops.ToDouble(b[j * 4 + 2]), jy2 = ops.ToDouble(b[j * 4 + 3]);
                double iw = Math.Max(0, Math.Min(ix2, jx2) - Math.Max(ix1, jx1));
                double ih = Math.Max(0, Math.Min(iy2, jy2) - Math.Max(iy1, jy1));
                double inter = iw * ih;
                double union = ai + areas[j] - inter;
                if (union > 0 && inter / union > iouThreshold) suppressed[j] = true;
            }
        }

        var result = new Tensor<int>(new[] { keep.Count });
        var rs = result.AsWritableSpan();
        for (int i = 0; i < keep.Count; i++) rs[i] = keep[i];
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<int> BatchedNms<T>(Tensor<T> boxes, Tensor<T> scores, Tensor<int> classIds, double iouThreshold)
    {
        ValidateBoxes(boxes, nameof(boxes));
        if (boxes.Rank != 2) throw new ArgumentException("BatchedNms requires rank-2 boxes [N, 4].");
        int n = boxes._shape[0];
        if (n == 0) return new Tensor<int>(new[] { 0 });
        if (classIds.Length != n)
            throw new ArgumentException("classIds must have length N.");

        // torchvision trick: offset boxes by class * offsetUnit so boxes
        // from different classes never overlap and a single global NMS
        // effectively runs per-class. torchvision uses (max + 1) but that
        // breaks when coords are entirely negative (offset can be ≤ 0 or
        // smaller than box extent). We use the full span (max − min + 1),
        // which dominates any pairwise distance regardless of sign.
        var ops = MathHelper.GetNumericOperations<T>();
        var b = boxes.AsSpan();
        double maxCoord = double.NegativeInfinity;
        double minCoord = double.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            double x1 = ops.ToDouble(b[i * 4]), y1 = ops.ToDouble(b[i * 4 + 1]);
            double x2 = ops.ToDouble(b[i * 4 + 2]), y2 = ops.ToDouble(b[i * 4 + 3]);
            maxCoord = Math.Max(Math.Max(maxCoord, x2), Math.Max(y2, Math.Max(x1, y1)));
            minCoord = Math.Min(Math.Min(minCoord, x1), Math.Min(y1, Math.Min(x2, y2)));
        }
        double offsetUnit = (maxCoord - minCoord) + 1.0;

        var ids = classIds.AsSpan();
        var offsetBoxes = new Tensor<T>(boxes._shape);
        var ob = offsetBoxes.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            T off = ops.FromDouble(ids[i] * offsetUnit);
            ob[i * 4] = ops.Add(b[i * 4], off);
            ob[i * 4 + 1] = ops.Add(b[i * 4 + 1], off);
            ob[i * 4 + 2] = ops.Add(b[i * 4 + 2], off);
            ob[i * 4 + 3] = ops.Add(b[i * 4 + 3], off);
        }
        return Nms(offsetBoxes, scores, iouThreshold);
    }

    /// <inheritdoc/>
    public virtual Tensor<int> MasksToBoxes<T>(Tensor<T> masks)
    {
        if (masks.Rank != 3)
            throw new ArgumentException("MasksToBoxes requires rank-3 masks [N, H, W].");
        var ops = MathHelper.GetNumericOperations<T>();
        int N = masks._shape[0], H = masks._shape[1], W = masks._shape[2];
        var src = masks.AsSpan();
        var result = new Tensor<int>(new[] { N, 4 });
        var dst = result.AsWritableSpan();
        T zero = ops.Zero;

        for (int n = 0; n < N; n++)
        {
            int xMin = int.MaxValue, yMin = int.MaxValue;
            int xMax = -1, yMax = -1;
            int planeOff = n * H * W;
            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; x++)
                {
                    if (!ops.Equals(src[planeOff + y * W + x], zero))
                    {
                        if (x < xMin) xMin = x;
                        if (x > xMax) xMax = x;
                        if (y < yMin) yMin = y;
                        if (y > yMax) yMax = y;
                    }
                }
            }
            int o = n * 4;
            if (xMax < 0)
            {
                // Empty mask — torchvision returns [0, 0, 0, 0].
                dst[o] = 0; dst[o + 1] = 0; dst[o + 2] = 0; dst[o + 3] = 0;
            }
            else
            {
                dst[o] = xMin; dst[o + 1] = yMin; dst[o + 2] = xMax; dst[o + 3] = yMax;
            }
        }
        return result;
    }

    // ---- helpers --------------------------------------------------------

    /// <summary>
    /// Pairwise IoU + intersection + per-side area, all in one pass.
    /// Returns (iou, union, areaA, areaB). Reused by every IoU variant.
    /// </summary>
    private (Tensor<T> iou, Tensor<T> union, Tensor<T> areaA, Tensor<T> areaB) ComputePairwiseIoU<T>(Tensor<T> boxesA, Tensor<T> boxesB)
    {
        // The IoU family divides (inter/union, enclose penalty, DIoU term,
        // CIoU α) in element type T — integral T silently truncates to 0/1.
        RequireFloatingPoint<T>("IoU family");
        var ops = MathHelper.GetNumericOperations<T>();
        int N = boxesA._shape[0], M = boxesB._shape[0];
        var areaA = (Tensor<T>)BoxArea(boxesA);
        var areaB = (Tensor<T>)BoxArea(boxesB);
        var aSpan = boxesA.AsSpan();
        var bSpan = boxesB.AsSpan();
        var aaSpan = areaA.AsSpan();
        var abSpan = areaB.AsSpan();
        var iou = new Tensor<T>(new[] { N, M });
        var union = new Tensor<T>(new[] { N, M });
        var iouSpan = iou.AsWritableSpan();
        var uSpan = union.AsWritableSpan();
        T zero = ops.Zero;
        for (int i = 0; i < N; i++)
        {
            T ax1 = aSpan[i * 4], ay1 = aSpan[i * 4 + 1], ax2 = aSpan[i * 4 + 2], ay2 = aSpan[i * 4 + 3];
            T areaAi = aaSpan[i];
            for (int j = 0; j < M; j++)
            {
                T bx1 = bSpan[j * 4], by1 = bSpan[j * 4 + 1], bx2 = bSpan[j * 4 + 2], by2 = bSpan[j * 4 + 3];
                T ix1 = ops.GreaterThan(ax1, bx1) ? ax1 : bx1;
                T iy1 = ops.GreaterThan(ay1, by1) ? ay1 : by1;
                T ix2 = ops.LessThan(ax2, bx2) ? ax2 : bx2;
                T iy2 = ops.LessThan(ay2, by2) ? ay2 : by2;
                T iw = ops.Subtract(ix2, ix1);
                T ih = ops.Subtract(iy2, iy1);
                if (ops.LessThan(iw, zero)) iw = zero;
                if (ops.LessThan(ih, zero)) ih = zero;
                T inter = ops.Multiply(iw, ih);
                T u = ops.Subtract(ops.Add(areaAi, abSpan[j]), inter);
                int idx = i * M + j;
                uSpan[idx] = u;
                iouSpan[idx] = ops.GreaterThan(u, zero) ? ops.Divide(inter, u) : zero;
            }
        }
        return (iou, union, areaA, areaB);
    }

    /// <summary>
    /// DIoU and CIoU share everything but the aspect-ratio penalty —
    /// switch via <paramref name="includeAspect"/>.
    /// </summary>
    private Tensor<T> DiouLikeImpl<T>(Tensor<T> boxesA, Tensor<T> boxesB, bool includeAspect)
    {
        ValidateBoxes(boxesA, nameof(boxesA));
        ValidateBoxes(boxesB, nameof(boxesB));
        RequireFloatingPoint<T>(includeAspect ? "CompleteBoxIou" : "DistanceBoxIou");
        if (boxesA.Rank != 2 || boxesB.Rank != 2)
            throw new ArgumentException("DistanceBoxIou/CompleteBoxIou require rank-2 boxes.");

        var ops = MathHelper.GetNumericOperations<T>();
        var (iou, _, _, _) = ComputePairwiseIoU(boxesA, boxesB);
        var iouSpan = iou.AsWritableSpan();
        var a = boxesA.AsSpan();
        var b = boxesB.AsSpan();
        int N = boxesA._shape[0], M = boxesB._shape[0];
        T half = ops.Divide(ops.One, ops.FromDouble(2.0));
        T four = ops.FromDouble(4.0);
        T pi = ops.FromDouble(Math.PI);
        T piSq = ops.Multiply(pi, pi);
        T zero = ops.Zero;

        for (int i = 0; i < N; i++)
        {
            T ax1 = a[i * 4], ay1 = a[i * 4 + 1], ax2 = a[i * 4 + 2], ay2 = a[i * 4 + 3];
            T acx = ops.Multiply(ops.Add(ax1, ax2), half);
            T acy = ops.Multiply(ops.Add(ay1, ay2), half);
            T aw = ops.Subtract(ax2, ax1);
            T ah = ops.Subtract(ay2, ay1);
            for (int j = 0; j < M; j++)
            {
                T bx1 = b[j * 4], by1 = b[j * 4 + 1], bx2 = b[j * 4 + 2], by2 = b[j * 4 + 3];
                T bcx = ops.Multiply(ops.Add(bx1, bx2), half);
                T bcy = ops.Multiply(ops.Add(by1, by2), half);
                T dx = ops.Subtract(acx, bcx);
                T dy = ops.Subtract(acy, bcy);
                T centreSq = ops.Add(ops.Multiply(dx, dx), ops.Multiply(dy, dy));

                // Enclosing box diagonal squared.
                T ex1 = ops.LessThan(ax1, bx1) ? ax1 : bx1;
                T ey1 = ops.LessThan(ay1, by1) ? ay1 : by1;
                T ex2 = ops.LessThan(ax2, bx2) ? bx2 : ax2;
                T ey2 = ops.LessThan(ay2, by2) ? by2 : ay2;
                T ew = ops.Subtract(ex2, ex1);
                T eh = ops.Subtract(ey2, ey1);
                T diagSq = ops.Add(ops.Multiply(ew, ew), ops.Multiply(eh, eh));

                int idx = i * M + j;
                if (ops.GreaterThan(diagSq, zero))
                {
                    T diouTerm = ops.Divide(centreSq, diagSq);
                    iouSpan[idx] = ops.Subtract(iouSpan[idx], diouTerm);
                }
                if (includeAspect)
                {
                    // v = (4/π²) · (atan(wA/hA) − atan(wB/hB))²  — only valid when both heights > 0.
                    T bw = ops.Subtract(bx2, bx1);
                    T bh = ops.Subtract(by2, by1);
                    if (ops.GreaterThan(ah, zero) && ops.GreaterThan(bh, zero))
                    {
                        double aspectA = Math.Atan(ops.ToDouble(aw) / ops.ToDouble(ah));
                        double aspectB = Math.Atan(ops.ToDouble(bw) / ops.ToDouble(bh));
                        double v = (4.0 / (Math.PI * Math.PI)) * (aspectA - aspectB) * (aspectA - aspectB);
                        // α = v / ((1 − iou) + v) — denominator stays positive
                        // because we add v ≥ 0 to (1 − iou) ≥ 0.
                        double iouVal = ops.ToDouble(iouSpan[idx]);
                        // After the DIoU subtraction iouSpan[idx] holds DIoU
                        // (which can be negative). For α we need raw IoU,
                        // not DIoU — recompute the raw IoU value here.
                        double rawIou = iouVal + ops.ToDouble(ops.Divide(centreSq, diagSq));
                        double denom = (1.0 - rawIou) + v;
                        double alpha = denom > 0 ? v / denom : 0;
                        iouSpan[idx] = ops.Subtract(iouSpan[idx], ops.FromDouble(alpha * v));
                    }
                }
            }
        }
        return iou;
    }

    // ========================================================================
    // IoU family backward (Issue #217). All four variants share the same
    // coordinate-level chain rule for the IoU-proper path; GIoU / DIoU /
    // CIoU each add their own gradient contributions on top. We route
    // everything through ComputeIouBackwardCell so the derivation stays
    // in one place.
    // ========================================================================

    /// <inheritdoc/>
    public virtual (Tensor<T> gradA, Tensor<T> gradB) BoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => IouFamilyBackward(gradOutput, boxesA, boxesB, IouVariant.Iou);

    /// <inheritdoc/>
    public virtual (Tensor<T> gradA, Tensor<T> gradB) GeneralizedBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => IouFamilyBackward(gradOutput, boxesA, boxesB, IouVariant.GIoU);

    /// <inheritdoc/>
    public virtual (Tensor<T> gradA, Tensor<T> gradB) DistanceBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => IouFamilyBackward(gradOutput, boxesA, boxesB, IouVariant.DIoU);

    /// <inheritdoc/>
    public virtual (Tensor<T> gradA, Tensor<T> gradB) CompleteBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => IouFamilyBackward(gradOutput, boxesA, boxesB, IouVariant.CIoU);

    private enum IouVariant { Iou, GIoU, DIoU, CIoU }

    private (Tensor<T> gradA, Tensor<T> gradB) IouFamilyBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB, IouVariant variant)
    {
        ValidateBoxes(boxesA, nameof(boxesA));
        ValidateBoxes(boxesB, nameof(boxesB));
        if (boxesA.Rank != 2 || boxesB.Rank != 2)
            throw new ArgumentException("IoU backward requires rank-2 boxes [N, 4] and [M, 4].");
        int N = boxesA._shape[0], M = boxesB._shape[0];
        if (gradOutput.Rank != 2 || gradOutput._shape[0] != N || gradOutput._shape[1] != M)
            throw new ArgumentException($"gradOutput must be [N={N}, M={M}].");

        var gradA = new Tensor<T>(new[] { N, 4 });
        var gradB = new Tensor<T>(new[] { M, 4 });
        if (N == 0 || M == 0) return (gradA, gradB);

        var ops = MathHelper.GetNumericOperations<T>();
        var a = boxesA.AsSpan();
        var b = boxesB.AsSpan();
        var go = gradOutput.AsSpan();
        var gA = gradA.AsWritableSpan();
        var gB = gradB.AsWritableSpan();

        // Work in double for numerical stability — the forward converts via
        // ops.ToDouble() for DIoU/CIoU anyway, so keeping backward in the
        // same precision keeps finite-difference tests reproducible.
        const double INV_PI_SQ = 4.0 / (Math.PI * Math.PI);

        for (int i = 0; i < N; i++)
        {
            double ax1 = ops.ToDouble(a[i * 4]);
            double ay1 = ops.ToDouble(a[i * 4 + 1]);
            double ax2 = ops.ToDouble(a[i * 4 + 2]);
            double ay2 = ops.ToDouble(a[i * 4 + 3]);

            for (int j = 0; j < M; j++)
            {
                double bx1 = ops.ToDouble(b[j * 4]);
                double by1 = ops.ToDouble(b[j * 4 + 1]);
                double bx2 = ops.ToDouble(b[j * 4 + 2]);
                double by2 = ops.ToDouble(b[j * 4 + 3]);

                double awRaw = ax2 - ax1, ahRaw = ay2 - ay1;
                double bwRaw = bx2 - bx1, bhRaw = by2 - by1;
                double aw = Math.Max(awRaw, 0);
                double ah = Math.Max(ahRaw, 0);
                double bw = Math.Max(bwRaw, 0);
                double bh = Math.Max(bhRaw, 0);
                double areaA = aw * ah;
                double areaB = bw * bh;

                double ix1 = Math.Max(ax1, bx1), ix2 = Math.Min(ax2, bx2);
                double iy1 = Math.Max(ay1, by1), iy2 = Math.Min(ay2, by2);
                double iwRaw = ix2 - ix1, ihRaw = iy2 - iy1;
                double iw = Math.Max(iwRaw, 0), ih = Math.Max(ihRaw, 0);
                double inter = iw * ih;
                double union = areaA + areaB - inter;
                double iou = union > 0 ? inter / union : 0;

                double g = ops.ToDouble(go[i * M + j]);
                if (g == 0.0) continue;

                // Accumulators per corner for this (i, j) cell.
                double gAx1 = 0, gAy1 = 0, gAx2 = 0, gAy2 = 0;
                double gBx1 = 0, gBy1 = 0, gBx2 = 0, gBy2 = 0;

                // g_iou gets the IoU-path contribution plus anything that
                // routes back through iou in GIoU/DIoU/CIoU. We accumulate
                // into gIou first then propagate once.
                double gIou = g;
                double gInter = 0, gAreaA = 0, gAreaB = 0;

                if (variant == IouVariant.GIoU)
                {
                    // GIoU = IoU + union/enclose − 1. The constant −1 drops out.
                    double ex1 = Math.Min(ax1, bx1), ex2 = Math.Max(ax2, bx2);
                    double ey1 = Math.Min(ay1, by1), ey2 = Math.Max(ay2, by2);
                    double ewRaw = ex2 - ex1, ehRaw = ey2 - ey1;
                    double ew = Math.Max(ewRaw, 0), eh = Math.Max(ehRaw, 0);
                    double enclose = ew * eh;
                    if (enclose > 0)
                    {
                        double gUnion = g / enclose;
                        double gEnclose = g * (-union / (enclose * enclose));
                        // union = areaA + areaB − inter
                        gAreaA += gUnion; gAreaB += gUnion; gInter += -gUnion;
                        // enclose = ew · eh
                        double gEw = gEnclose * eh;
                        double gEh = gEnclose * ew;
                        // ew = max(ewRaw, 0)
                        double gEwRaw = ewRaw > 0 ? gEw : 0;
                        double gEhRaw = ehRaw > 0 ? gEh : 0;
                        // ex1 = min(ax1, bx1); ex2 = max(ax2, bx2)
                        double gEx1 = -gEwRaw, gEx2 = gEwRaw;
                        double gEy1 = -gEhRaw, gEy2 = gEhRaw;
                        if (ax1 <= bx1) gAx1 += gEx1; else gBx1 += gEx1;
                        if (ay1 <= by1) gAy1 += gEy1; else gBy1 += gEy1;
                        if (ax2 >= bx2) gAx2 += gEx2; else gBx2 += gEx2;
                        if (ay2 >= by2) gAy2 += gEy2; else gBy2 += gEy2;
                    }
                }
                else if (variant == IouVariant.DIoU || variant == IouVariant.CIoU)
                {
                    // DIoU = IoU − centreSq/diagSq.
                    double acx = (ax1 + ax2) * 0.5, acy = (ay1 + ay2) * 0.5;
                    double bcx = (bx1 + bx2) * 0.5, bcy = (by1 + by2) * 0.5;
                    double dcx = acx - bcx, dcy = acy - bcy;
                    double centreSq = dcx * dcx + dcy * dcy;
                    double ex1 = Math.Min(ax1, bx1), ex2 = Math.Max(ax2, bx2);
                    double ey1 = Math.Min(ay1, by1), ey2 = Math.Max(ay2, by2);
                    double ew = ex2 - ex1, eh = ey2 - ey1;
                    double diagSq = ew * ew + eh * eh;

                    double gCentreSq = 0, gDiagSq = 0;
                    if (diagSq > 0)
                    {
                        gCentreSq = g * (-1.0 / diagSq);
                        gDiagSq = g * centreSq / (diagSq * diagSq);
                    }

                    // CIoU adds −α·v (α treated as stop-gradient per Zheng 2020).
                    if (variant == IouVariant.CIoU && ah > 0 && bh > 0)
                    {
                        double aspectA = Math.Atan(aw / ah);
                        double aspectB = Math.Atan(bw / bh);
                        double diff = aspectA - aspectB;
                        double v = INV_PI_SQ * diff * diff;
                        double denom = (1.0 - iou) + v;
                        double alpha = denom > 0 ? v / denom : 0;
                        // CIoU = DIoU − α·v, so dCIoU/dv = −α (α constant).
                        double gV = g * (-alpha);
                        // v = INV_PI_SQ · diff². dv/dDiff = 2·INV_PI_SQ·diff.
                        double gDiff = gV * 2.0 * INV_PI_SQ * diff;
                        double gAspectA = gDiff, gAspectB = -gDiff;
                        // atan(r): d/dr = 1/(1+r²). For atan(w/h), dr/dw = 1/h
                        // and dr/dh = −w/h². Combined, d(atan(w/h))/dw = h/(w²+h²)
                        // and d(atan(w/h))/dh = −w/(w²+h²).
                        double aDen = aw * aw + ah * ah;
                        double bDen = bw * bw + bh * bh;
                        if (aDen > 0)
                        {
                            double gAw = gAspectA * (ah / aDen);
                            double gAh = gAspectA * (-aw / aDen);
                            // aw = max(awRaw, 0); ah = max(ahRaw, 0).
                            if (awRaw > 0) { gAx2 += gAw; gAx1 += -gAw; }
                            if (ahRaw > 0) { gAy2 += gAh; gAy1 += -gAh; }
                        }
                        if (bDen > 0)
                        {
                            double gBw = gAspectB * (bh / bDen);
                            double gBh = gAspectB * (-bw / bDen);
                            if (bwRaw > 0) { gBx2 += gBw; gBx1 += -gBw; }
                            if (bhRaw > 0) { gBy2 += gBh; gBy1 += -gBh; }
                        }
                        // α is stop-gradient, so no contribution through gIou
                        // from the α·v factor.
                    }

                    // Propagate gCentreSq through the centre distance.
                    // centreSq = (acx-bcx)² + (acy-bcy)².
                    double gAcx = gCentreSq * 2.0 * dcx;
                    double gAcy = gCentreSq * 2.0 * dcy;
                    // acx = (ax1+ax2)/2, bcx = (bx1+bx2)/2.
                    gAx1 += gAcx * 0.5; gAx2 += gAcx * 0.5;
                    gAy1 += gAcy * 0.5; gAy2 += gAcy * 0.5;
                    gBx1 += -gAcx * 0.5; gBx2 += -gAcx * 0.5;
                    gBy1 += -gAcy * 0.5; gBy2 += -gAcy * 0.5;

                    // Propagate gDiagSq through enclose dimensions.
                    // diagSq = ew² + eh². dew/dex1 = −1 (ew = ex2 − ex1),
                    // dew/dex2 = +1. Same for eh.
                    double gEw = gDiagSq * 2.0 * ew;
                    double gEh = gDiagSq * 2.0 * eh;
                    // ex1 = min(ax1, bx1); ex2 = max(ax2, bx2). No max(...,0)
                    // clamp on ew/eh for DIoU — diagSq uses the raw span.
                    double gEx1 = -gEw, gEx2 = gEw;
                    double gEy1 = -gEh, gEy2 = gEh;
                    if (ax1 <= bx1) gAx1 += gEx1; else gBx1 += gEx1;
                    if (ay1 <= by1) gAy1 += gEy1; else gBy1 += gEy1;
                    if (ax2 >= bx2) gAx2 += gEx2; else gBx2 += gEx2;
                    if (ay2 >= by2) gAy2 += gEy2; else gBy2 += gEy2;
                }

                // IoU-proper contribution (shared across all variants).
                if (union > 0)
                {
                    gInter += gIou * (union + inter) / (union * union);
                    gAreaA += gIou * (-inter) / (union * union);
                    gAreaB += gIou * (-inter) / (union * union);
                }

                // inter = iw · ih.
                double gIw = gInter * ih;
                double gIh = gInter * iw;
                double gIwRaw = iwRaw > 0 ? gIw : 0;
                double gIhRaw = ihRaw > 0 ? gIh : 0;
                double gIx2 = gIwRaw, gIx1 = -gIwRaw;
                double gIy2 = gIhRaw, gIy1 = -gIhRaw;
                // ix1 = max(ax1, bx1); ix2 = min(ax2, bx2).
                if (ax1 >= bx1) gAx1 += gIx1; else gBx1 += gIx1;
                if (ay1 >= by1) gAy1 += gIy1; else gBy1 += gIy1;
                if (ax2 <= bx2) gAx2 += gIx2; else gBx2 += gIx2;
                if (ay2 <= by2) gAy2 += gIy2; else gBy2 += gIy2;

                // areaA = aw · ah.
                double gAw2 = gAreaA * ah;
                double gAh2 = gAreaA * aw;
                double gBw2 = gAreaB * bh;
                double gBh2 = gAreaB * bw;
                if (awRaw > 0) { gAx2 += gAw2; gAx1 += -gAw2; }
                if (ahRaw > 0) { gAy2 += gAh2; gAy1 += -gAh2; }
                if (bwRaw > 0) { gBx2 += gBw2; gBx1 += -gBw2; }
                if (bhRaw > 0) { gBy2 += gBh2; gBy1 += -gBh2; }

                // Accumulate into output tensors.
                gA[i * 4] = ops.Add(gA[i * 4], ops.FromDouble(gAx1));
                gA[i * 4 + 1] = ops.Add(gA[i * 4 + 1], ops.FromDouble(gAy1));
                gA[i * 4 + 2] = ops.Add(gA[i * 4 + 2], ops.FromDouble(gAx2));
                gA[i * 4 + 3] = ops.Add(gA[i * 4 + 3], ops.FromDouble(gAy2));
                gB[j * 4] = ops.Add(gB[j * 4], ops.FromDouble(gBx1));
                gB[j * 4 + 1] = ops.Add(gB[j * 4 + 1], ops.FromDouble(gBy1));
                gB[j * 4 + 2] = ops.Add(gB[j * 4 + 2], ops.FromDouble(gBx2));
                gB[j * 4 + 3] = ops.Add(gB[j * 4 + 3], ops.FromDouble(gBy2));
            }
        }
        return (gradA, gradB);
    }

    private static void ValidateBoxes<T>(Tensor<T> boxes, string paramName)
    {
        if (boxes is null) throw new ArgumentNullException(paramName);
        if (boxes.Rank < 1 || boxes._shape[boxes.Rank - 1] != 4)
            throw new ArgumentException(
                $"Box tensor's last axis must be 4; got shape [{string.Join(", ", boxes._shape)}].",
                paramName);
    }

    /// <summary>
    /// Detection ops use ratios (inter/union, centre²/diag², ½ in CXCYWH),
    /// which degrade to integer 0/1 arithmetic on integral T and silently
    /// corrupt results. Reject integral T up front rather than producing
    /// garbage.
    /// </summary>
    private static void RequireFloatingPoint<T>(string op)
    {
        var t = typeof(T);
        if (t == typeof(float) || t == typeof(double) || t == typeof(decimal)) return;
        throw new NotSupportedException(
            $"{op} requires a floating-point element type (float/double/decimal); got {t.Name}.");
    }
}
