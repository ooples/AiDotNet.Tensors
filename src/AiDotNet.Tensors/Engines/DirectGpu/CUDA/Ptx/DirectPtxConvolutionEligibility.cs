using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxConvolutionShape(
    int Batch,
    int InputChannels,
    int InputHeight,
    int InputWidth,
    int OutputChannels,
    int OutputHeight,
    int OutputWidth,
    int KernelHeight,
    int KernelWidth,
    int StrideHeight,
    int StrideWidth,
    int PaddingHeight,
    int PaddingWidth,
    int DilationHeight,
    int DilationWidth);

/// <summary>
/// Single validation boundary for the issue-#841 experimental convolution ABI.
/// A successful result is the capability check that lets dispatch construct
/// stride-free tensor views and enter the shape-specialized kernel.
/// </summary>
internal static class DirectPtxConvolutionEligibility
{
    internal const string FeatureDisabled = "convolution-feature-disabled";
    internal const string BackendUnavailable = "convolution-backend-unavailable";
    internal const string ArchitectureNotImplemented = "convolution-architecture-not-implemented";
    internal const string ShapeNotImplemented = "convolution-shape-not-implemented";
    internal const string NullBuffer = "convolution-null-buffer";
    internal const string InvalidDevicePointer = "convolution-invalid-device-pointer";
    internal const string PhysicalExtentMismatch = "convolution-physical-extent-mismatch";
    internal const string AlignmentMismatch = "convolution-alignment-mismatch";
    internal const string AliasNotSupported = "convolution-alias-not-supported";

    internal static string? Validate(
        bool featureEnabled,
        bool backendAvailable,
        int computeCapabilityMajor,
        int computeCapabilityMinor,
        DirectPtxConvolutionShape shape,
        IGpuBuffer? input,
        IGpuBuffer? weights,
        IGpuBuffer? bias,
        IGpuBuffer? output)
    {
        if (!featureEnabled) return FeatureDisabled;
        if (!backendAvailable) return BackendUnavailable;
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            return ArchitectureNotImplemented;
        if (shape != PtxFusedConv2DNchwK1Kernel.Shape)
            return ShapeNotImplemented;
        if (input is null || weights is null || bias is null || output is null)
            return NullBuffer;
        if (input.Handle == IntPtr.Zero || weights.Handle == IntPtr.Zero ||
            bias.Handle == IntPtr.Zero || output.Handle == IntPtr.Zero)
            return InvalidDevicePointer;
        if (input.SizeInBytes != PtxFusedConv2DNchwK1Kernel.InputBytes ||
            weights.SizeInBytes != PtxFusedConv2DNchwK1Kernel.WeightBytes ||
            bias.SizeInBytes != PtxFusedConv2DNchwK1Kernel.BiasBytes ||
            output.SizeInBytes != PtxFusedConv2DNchwK1Kernel.OutputBytes)
            return PhysicalExtentMismatch;
        if (((PtxCompat.ToNuint(input.Handle) | PtxCompat.ToNuint(weights.Handle) |
              PtxCompat.ToNuint(bias.Handle) | PtxCompat.ToNuint(output.Handle)) & 15u) != 0)
            return AlignmentMismatch;
        if (Overlaps(input, output) || Overlaps(weights, output) || Overlaps(bias, output))
            return AliasNotSupported;
        return null;
    }

    private static bool Overlaps(IGpuBuffer left, IGpuBuffer right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Handle);
        nuint rightStart = PtxCompat.ToNuint(right.Handle);
        nuint leftBytes = (nuint)left.SizeInBytes;
        nuint rightBytes = (nuint)right.SizeInBytes;
        nuint leftEnd = leftStart + leftBytes;
        nuint rightEnd = rightStart + rightBytes;
        // Malformed pointer ranges (unsigned wraparound, where end < start) must
        // reject rather than escape the fail-closed admission boundary.
        if (leftEnd < leftStart || rightEnd < rightStart)
            return true;
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
