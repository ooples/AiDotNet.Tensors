using System;
using System.Linq;
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
        return iou;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> DistanceBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => DiouLikeImpl(boxesA, boxesB, includeAspect: false);

    /// <inheritdoc/>
    public virtual Tensor<T> CompleteBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => DiouLikeImpl(boxesA, boxesB, includeAspect: true);

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

        // torchvision trick: offset boxes by class * (max_coord + 1) so
        // boxes from different classes never overlap and a single global
        // NMS effectively runs per-class.
        var ops = MathHelper.GetNumericOperations<T>();
        var b = boxes.AsSpan();
        double maxCoord = 0;
        for (int i = 0; i < n; i++)
        {
            double x1 = ops.ToDouble(b[i * 4]), y1 = ops.ToDouble(b[i * 4 + 1]);
            double x2 = ops.ToDouble(b[i * 4 + 2]), y2 = ops.ToDouble(b[i * 4 + 3]);
            maxCoord = Math.Max(Math.Max(maxCoord, x2), Math.Max(y2, Math.Max(x1, y1)));
        }
        double offsetUnit = maxCoord + 1.0;

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
    public virtual Tensor<T> MasksToBoxes<T>(Tensor<T> masks)
    {
        if (masks.Rank != 3)
            throw new ArgumentException("MasksToBoxes requires rank-3 masks [N, H, W].");
        var ops = MathHelper.GetNumericOperations<T>();
        int N = masks._shape[0], H = masks._shape[1], W = masks._shape[2];
        var src = masks.AsSpan();
        var result = new Tensor<T>(new[] { N, 4 });
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
                dst[o] = zero; dst[o + 1] = zero; dst[o + 2] = zero; dst[o + 3] = zero;
            }
            else
            {
                dst[o] = ops.FromDouble(xMin);
                dst[o + 1] = ops.FromDouble(yMin);
                dst[o + 2] = ops.FromDouble(xMax);
                dst[o + 3] = ops.FromDouble(yMax);
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

    private static void ValidateBoxes<T>(Tensor<T> boxes, string paramName)
    {
        if (boxes is null) throw new ArgumentNullException(paramName);
        if (boxes.Rank < 1 || boxes._shape[boxes.Rank - 1] != 4)
            throw new ArgumentException(
                $"Box tensor's last axis must be 4; got shape [{string.Join(", ", boxes._shape)}].",
                paramName);
    }
}
