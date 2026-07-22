#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Closed admission table for issue #851. A public CUDA route must obtain a
/// key here before the direct-PTX runtime is consulted. Consequently arbitrary
/// dimensions never cause JIT work and every admitted dimension is baked into
/// the generated module.
/// </summary>
internal static class DirectPtxVisionSpecializations
{
    internal static bool IsAdmitted(DirectPtxVisionSpec requested)
    {
        DirectPtxVisionSpec canonical = default;
        bool admitted = requested.Operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou or
            DirectPtxVisionOperation.DistanceBoxIou or
            DirectPtxVisionOperation.CompleteBoxIou => TryPairwise(
                requested.Operation, requested.D0, requested.D1, out canonical),
            DirectPtxVisionOperation.BoxArea or
            DirectPtxVisionOperation.IoULoss or
            DirectPtxVisionOperation.GIoULoss or
            DirectPtxVisionOperation.DIoULoss or
            DirectPtxVisionOperation.CIoULoss or
            DirectPtxVisionOperation.IoULossBackward or
            DirectPtxVisionOperation.GIoULossBackward or
            DirectPtxVisionOperation.DIoULossBackward or
            DirectPtxVisionOperation.CIoULossBackward => TryVector(
                requested.Operation, requested.D0, out canonical),
            DirectPtxVisionOperation.BoxConvert => TryBoxConvert(
                requested.D0, requested.D1, requested.D2, out canonical),
            DirectPtxVisionOperation.IouFamilyBackwardA or
            DirectPtxVisionOperation.IouFamilyBackwardB => TryPairwiseBackward(
                requested.Operation == DirectPtxVisionOperation.IouFamilyBackwardA,
                requested.D0, requested.D1, requested.D2, out canonical),
            DirectPtxVisionOperation.Nms => TryNms(
                requested.D0, BitConverter.Int32BitsToSingle(requested.ScalarBits),
                (requested.Flags & 1) != 0, out canonical),
            DirectPtxVisionOperation.MasksToBoxes => TryMasksToBoxes(
                requested.D0, requested.D1, requested.D2, out canonical),
            DirectPtxVisionOperation.RoiAlign or
            DirectPtxVisionOperation.RoiPool or
            DirectPtxVisionOperation.PsRoiAlign or
            DirectPtxVisionOperation.PsRoiPool => TryRoi(
                requested.Operation, requested.D0, requested.D1, requested.D2,
                requested.D3, requested.D4, requested.D5, requested.D6,
                requested.D7, BitConverter.Int32BitsToSingle(requested.ScalarBits),
                requested.Flags & 0xff, (requested.Flags & 0x100) != 0,
                out canonical),
            DirectPtxVisionOperation.Cross3 => TryCross3(
                requested.D0, requested.D1, out canonical),
            DirectPtxVisionOperation.Meshgrid2D => TryMeshgrid2D(
                requested.D0, requested.D1, requested.Flags & 1,
                (requested.Flags & 2) != 0, out canonical),
            _ => false
        };
        return admitted && canonical == requested;
    }

    internal static bool TryPairwise(
        DirectPtxVisionOperation operation, int n, int m, out DirectPtxVisionSpec spec)
    {
        bool family = operation is DirectPtxVisionOperation.GeneralizedBoxIou or
            DirectPtxVisionOperation.DistanceBoxIou or
            DirectPtxVisionOperation.CompleteBoxIou;
        bool admitted = family && (n, m) is (256, 256) or (1024, 256) or
            (1024, 1024) or (4096, 256);
        spec = admitted ? new(operation, n, m) : default;
        return admitted;
    }

    internal static bool TryVector(
        DirectPtxVisionOperation operation, int n, out DirectPtxVisionSpec spec)
    {
        bool family = operation is DirectPtxVisionOperation.BoxArea or
            DirectPtxVisionOperation.IoULoss or
            DirectPtxVisionOperation.GIoULoss or
            DirectPtxVisionOperation.DIoULoss or
            DirectPtxVisionOperation.CIoULoss or
            DirectPtxVisionOperation.IoULossBackward or
            DirectPtxVisionOperation.GIoULossBackward or
            DirectPtxVisionOperation.DIoULossBackward or
            DirectPtxVisionOperation.CIoULossBackward;
        bool admitted = family && (n is 256 or 1024 or 4096);
        spec = admitted ? new(operation, n) : default;
        return admitted;
    }

    internal static bool TryBoxConvert(
        int n, int from, int to, out DirectPtxVisionSpec spec)
    {
        bool admitted = (n is 256 or 1024 or 4096) &&
            (uint)from <= 2 && (uint)to <= 2;
        spec = admitted
            ? new(DirectPtxVisionOperation.BoxConvert, n, from, to)
            : default;
        return admitted;
    }

    internal static bool TryPairwiseBackward(
        bool ownerA, int n, int m, int variant, out DirectPtxVisionSpec spec)
    {
        bool admitted = (n is 256 or 1024) && (m is 256 or 1024) &&
            (uint)variant <= 3;
        spec = admitted
            ? new(ownerA ? DirectPtxVisionOperation.IouFamilyBackwardA :
                DirectPtxVisionOperation.IouFamilyBackwardB, n, m, variant)
            : default;
        return admitted;
    }

    internal static bool TryNms(
        int length, float threshold, bool batched, out DirectPtxVisionSpec spec)
    {
        bool admitted = (length is 256 or 1024) &&
            BitConverter.SingleToInt32Bits(threshold) ==
            BitConverter.SingleToInt32Bits(0.5f);
        spec = admitted
            ? new(DirectPtxVisionOperation.Nms, length, Flags: batched ? 1 : 0,
                ScalarBits: BitConverter.SingleToInt32Bits(threshold))
            : default;
        return admitted;
    }

    internal static bool TryMasksToBoxes(
        int n, int h, int w, out DirectPtxVisionSpec spec)
    {
        bool admitted = (n, h, w) is (256, 28, 28) or (64, 64, 64);
        spec = admitted
            ? new(DirectPtxVisionOperation.MasksToBoxes, n, h, w)
            : default;
        return admitted;
    }

    internal static bool TryRoi(
        DirectPtxVisionOperation operation,
        int n, int c, int h, int w, int k, int outH, int outW,
        int outputChannels, float spatialScale, int samplingRatio, bool aligned,
        out DirectPtxVisionSpec spec)
    {
        bool roiOperation = operation is DirectPtxVisionOperation.RoiAlign or
            DirectPtxVisionOperation.RoiPool or
            DirectPtxVisionOperation.PsRoiAlign or
            DirectPtxVisionOperation.PsRoiPool;
        bool positionSensitive = operation is DirectPtxVisionOperation.PsRoiAlign or
            DirectPtxVisionOperation.PsRoiPool;
        bool align = operation is DirectPtxVisionOperation.RoiAlign or
            DirectPtxVisionOperation.PsRoiAlign;
        bool shape = positionSensitive
            ? (n, c, h, w, k, outH, outW, outputChannels) ==
              (1, 196, 56, 56, 256, 7, 7, 4)
            : (n, c, h, w, k, outH, outW) ==
              (1, 256, 56, 56, 256, 7, 7) && outputChannels == c;
        bool semantics = operation switch
        {
            DirectPtxVisionOperation.RoiAlign => samplingRatio == 2,
            DirectPtxVisionOperation.PsRoiAlign => samplingRatio == 2 && !aligned,
            DirectPtxVisionOperation.RoiPool or DirectPtxVisionOperation.PsRoiPool =>
                samplingRatio == 0 && !aligned,
            _ => false
        };
        bool admitted = roiOperation && shape && semantics &&
            BitConverter.SingleToInt32Bits(spatialScale) ==
            BitConverter.SingleToInt32Bits(0.25f);
        int flags = samplingRatio & 0xff;
        if (aligned) flags |= 0x100;
        spec = admitted
            ? new(operation, n, c, h, w, k, outH, outW, outputChannels,
                flags, BitConverter.SingleToInt32Bits(spatialScale))
            : default;
        return admitted;
    }

    internal static bool TryCross3(
        int outer, int inner, out DirectPtxVisionSpec spec)
    {
        bool admitted = (outer, inner) is (256, 1) or (1024, 1) or (256, 64);
        spec = admitted
            ? new(DirectPtxVisionOperation.Cross3, outer, inner)
            : default;
        return admitted;
    }

    internal static bool TryMeshgrid2D(
        int n0, int n1, int outputIndex, bool xy, out DirectPtxVisionSpec spec)
    {
        bool admitted = ((n0, n1) is (256, 256) or (1024, 256)) &&
            outputIndex is 0 or 1;
        spec = admitted
            ? new(DirectPtxVisionOperation.Meshgrid2D, n0, n1,
                Flags: outputIndex | (xy ? 2 : 0))
            : default;
        return admitted;
    }
}
#endif
