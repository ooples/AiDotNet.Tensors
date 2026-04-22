// Copyright (c) AiDotNet. All rights reserved.
// GPU dispatch for the vision detection ops added by Issue #217.
// Pattern matches DirectGpuTensorEngine.Parity210.cs:
//   - if active backend implements IDetectionBackend, dispatch to its
//     native kernel
//   - otherwise (or on any failure) fall through to the inherited
//     CpuEngine implementation
// The CpuEngine fallback gives correct (slower) results on every
// non-OpenCL backend until they ship their own kernels.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> BoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuBoxIou(boxesA, boxesB, "BoxIou", static (b, a, x, o, n, m) => b.BoxIou(a, x, o, n, m))
           ?? base.BoxIou(boxesA, boxesB);

    /// <inheritdoc/>
    public override Tensor<T> GeneralizedBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuBoxIou(boxesA, boxesB, "GeneralizedBoxIou",
                        static (b, a, x, o, n, m) => b.GeneralizedBoxIou(a, x, o, n, m))
           ?? base.GeneralizedBoxIou(boxesA, boxesB);

    /// <inheritdoc/>
    public override Tensor<T> DistanceBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuBoxIou(boxesA, boxesB, "DistanceBoxIou",
                        static (b, a, x, o, n, m) => b.DistanceBoxIou(a, x, o, n, m))
           ?? base.DistanceBoxIou(boxesA, boxesB);

    /// <inheritdoc/>
    public override Tensor<T> CompleteBoxIou<T>(Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuBoxIou(boxesA, boxesB, "CompleteBoxIou",
                        static (b, a, x, o, n, m) => b.CompleteBoxIou(a, x, o, n, m))
           ?? base.CompleteBoxIou(boxesA, boxesB);

    /// <inheritdoc/>
    public override Tensor<T> BoxArea<T>(Tensor<T> boxes)
    {
        // Restrict GPU dispatch to float (the kernel surface) and rank-2
        // boxes so the [N,4] → [N] reshape stays trivially correct.
        if (typeof(T) != typeof(float) || boxes.Rank != 2 || boxes._shape[1] != 4)
            return base.BoxArea(boxes);
        try
        {
            if (TryGetBackend(out var backend) && backend is IDetectionBackend det)
            {
                int n = boxes._shape[0];
                if (n == 0) return base.BoxArea(boxes);
                using var inBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, n);
                try
                {
                    det.BoxArea(inBuf.Buffer, outBuf.Buffer, n);
                    var arr = FinishGpuOp<T>(backend, outBuf, n);
                    return new Tensor<T>(arr, new[] { n });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.BoxArea(boxes);
    }

    /// <inheritdoc/>
    public override Tensor<T> BoxConvert<T>(Tensor<T> boxes, BoxFormat from, BoxFormat to)
    {
        if (from == to) return base.BoxConvert(boxes, from, to);
        if (typeof(T) != typeof(float) || boxes.Rank < 1 || boxes._shape[boxes.Rank - 1] != 4)
            return base.BoxConvert(boxes, from, to);
        try
        {
            if (TryGetBackend(out var backend) && backend is IDetectionBackend det)
            {
                int n = boxes.Length / 4;
                if (n == 0) return base.BoxConvert(boxes, from, to);
                using var inBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, boxes.Length);
                try
                {
                    det.BoxConvert(inBuf.Buffer, outBuf.Buffer, n, (int)from, (int)to);
                    var arr = FinishGpuOp<T>(backend, outBuf, boxes.Length);
                    return new Tensor<T>(arr, (int[])boxes._shape.Clone());
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        catch { }
        return base.BoxConvert(boxes, from, to);
    }

    /// <inheritdoc/>
    public override (Tensor<T> gradA, Tensor<T> gradB) BoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuIouBackward(gradOutput, boxesA, boxesB, 0) ?? base.BoxIouBackward(gradOutput, boxesA, boxesB);

    /// <inheritdoc/>
    public override (Tensor<T> gradA, Tensor<T> gradB) GeneralizedBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuIouBackward(gradOutput, boxesA, boxesB, 1) ?? base.GeneralizedBoxIouBackward(gradOutput, boxesA, boxesB);

    /// <inheritdoc/>
    public override (Tensor<T> gradA, Tensor<T> gradB) DistanceBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuIouBackward(gradOutput, boxesA, boxesB, 2) ?? base.DistanceBoxIouBackward(gradOutput, boxesA, boxesB);

    /// <inheritdoc/>
    public override (Tensor<T> gradA, Tensor<T> gradB) CompleteBoxIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB)
        => TryGpuIouBackward(gradOutput, boxesA, boxesB, 3) ?? base.CompleteBoxIouBackward(gradOutput, boxesA, boxesB);

    /// <summary>
    /// GPU dispatch for the shared IouFamilyBackward kernel. <paramref name="variant"/>
    /// matches the kernel's int code (0 IoU, 1 GIoU, 2 DIoU, 3 CIoU).
    /// Returns null to fall through to the CPU reference on any failure
    /// or when the backend doesn't implement <see cref="IDetectionBackend"/>.
    /// </summary>
    private (Tensor<T> gradA, Tensor<T> gradB)? TryGpuIouBackward<T>(
        Tensor<T> gradOutput, Tensor<T> boxesA, Tensor<T> boxesB, int variant)
    {
        if (typeof(T) != typeof(float)) return null;
        if (boxesA.Rank != 2 || boxesA._shape[1] != 4) return null;
        if (boxesB.Rank != 2 || boxesB._shape[1] != 4) return null;
        int n = boxesA._shape[0];
        int m = boxesB._shape[0];
        if (gradOutput.Rank != 2 || gradOutput._shape[0] != n || gradOutput._shape[1] != m) return null;
        if (n == 0 || m == 0) return null;
        try
        {
            if (TryGetBackend(out var backend) && backend is IDetectionBackend det)
            {
                using var goBuf = GetOrAllocateBuffer(backend, gradOutput);
                using var aBuf = GetOrAllocateBuffer(backend, boxesA);
                using var bBuf = GetOrAllocateBuffer(backend, boxesB);
                var gABuf = AllocateOutputBuffer(backend, n * 4);
                var gBBuf = AllocateOutputBuffer(backend, m * 4);
                try
                {
                    det.IouFamilyBackward(goBuf.Buffer, aBuf.Buffer, bBuf.Buffer,
                        gABuf.Buffer, gBBuf.Buffer, n, m, variant);
                    var arrA = FinishGpuOp<T>(backend, gABuf, n * 4);
                    var arrB = FinishGpuOp<T>(backend, gBBuf, m * 4);
                    return (new Tensor<T>(arrA, new[] { n, 4 }), new Tensor<T>(arrB, new[] { m, 4 }));
                }
                catch { gABuf.Dispose(); gBBuf.Dispose(); throw; }
            }
        }
        catch { }
        return null;
    }

    /// <summary>
    /// Shared dispatch for the four pairwise IoU variants — they only
    /// differ in which IDetectionBackend method handles the kernel
    /// launch, so factor out the buffer plumbing.
    /// Returns null on any GPU-eligible failure so the caller can fall
    /// back to the CPU implementation.
    /// </summary>
    private Tensor<T>? TryGpuBoxIou<T>(
        Tensor<T> boxesA, Tensor<T> boxesB, string opName,
        System.Action<IDetectionBackend, IGpuBuffer, IGpuBuffer, IGpuBuffer, int, int> kernel)
    {
        if (typeof(T) != typeof(float)) return null;
        if (boxesA.Rank != 2 || boxesA._shape[1] != 4) return null;
        if (boxesB.Rank != 2 || boxesB._shape[1] != 4) return null;
        try
        {
            if (TryGetBackend(out var backend) && backend is IDetectionBackend det)
            {
                int n = boxesA._shape[0];
                int m = boxesB._shape[0];
                if (n == 0 || m == 0) return null;
                using var aBuf = GetOrAllocateBuffer(backend, boxesA);
                using var bBuf = GetOrAllocateBuffer(backend, boxesB);
                var outBuf = AllocateOutputBuffer(backend, n * m);
                try
                {
                    kernel(det, aBuf.Buffer, bBuf.Buffer, outBuf.Buffer, n, m);
                    var arr = FinishGpuOp<T>(backend, outBuf, n * m);
                    return new Tensor<T>(arr, new[] { n, m });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        catch { }
        return null;
    }
}
