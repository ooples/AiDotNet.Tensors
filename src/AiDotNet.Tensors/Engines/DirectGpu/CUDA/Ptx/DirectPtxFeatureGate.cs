using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Opt-in gate for hand-emitted PTX kernels. The gate is deliberately off by
/// default until a specialization passes its correctness, no-spill, and
/// championship benchmarks on a supported SM.
/// </summary>
internal static class DirectPtxFeatureGate
{
    internal const string MasterEnvironmentVariable = "AIDOTNET_DIRECT_PTX";
    internal const string EnvironmentVariable = "AIDOTNET_DIRECT_PTX_ATTENTION";
    internal const string ResidualRmsNormEnvironmentVariable = "AIDOTNET_DIRECT_PTX_RESIDUAL_RMSNORM";
    internal const string FlashDecodeEnvironmentVariable = "AIDOTNET_DIRECT_PTX_FLASH_DECODE";
    internal const string PagedDecodeEnvironmentVariable = "AIDOTNET_DIRECT_PTX_PAGED_DECODE";
    internal const string PagedPrefillEnvironmentVariable = "AIDOTNET_DIRECT_PTX_PAGED_PREFILL";
    internal const string AttentionBackwardEnvironmentVariable = "AIDOTNET_DIRECT_PTX_ATTENTION_BACKWARD";
    internal const string FlashAttentionBackwardEnvironmentVariable = "AIDOTNET_DIRECT_PTX_FLASH_ATTENTION_BACKWARD";
    internal const string QkvRopeCacheEnvironmentVariable = "AIDOTNET_DIRECT_PTX_QKV_ROPE_CACHE";
    internal const string RecurrentStateEnvironmentVariable = "AIDOTNET_DIRECT_PTX_RECURRENT_STATE";
    internal const string AutotuneEnvironmentVariable = "AIDOTNET_DIRECT_PTX_AUTOTUNE";
    internal const string CacheCapacityEnvironmentVariable = "AIDOTNET_DIRECT_PTX_CACHE_CAPACITY";

    // Feature configuration is a process-start contract. Snapshot it once so
    // the resident launch path never allocates strings while re-reading the
    // environment. Tests retain an explicit dynamic override below.
    private static readonly bool EnvironmentMasterEnabled = ReadEnabled(MasterEnvironmentVariable);
    private static readonly bool EnvironmentAttentionEnabled = ReadEnabled(EnvironmentVariable);
    private static readonly bool EnvironmentResidualRmsNormEnabled = ReadEnabled(ResidualRmsNormEnvironmentVariable);
    private static readonly bool EnvironmentFlashDecodeEnabled = ReadEnabled(FlashDecodeEnvironmentVariable);
    private static readonly bool EnvironmentPagedDecodeEnabled = ReadEnabled(PagedDecodeEnvironmentVariable);
    private static readonly bool EnvironmentPagedPrefillEnabled = ReadEnabled(PagedPrefillEnvironmentVariable);
    private static readonly bool EnvironmentAttentionBackwardEnabled = ReadEnabled(AttentionBackwardEnvironmentVariable);
    private static readonly bool EnvironmentFlashAttentionBackwardEnabled = ReadEnabled(FlashAttentionBackwardEnvironmentVariable);
    private static readonly bool EnvironmentQkvRopeCacheEnabled = ReadEnabled(QkvRopeCacheEnvironmentVariable);
    private static readonly bool EnvironmentRecurrentStateEnabled = ReadEnabled(RecurrentStateEnvironmentVariable);
    private static readonly bool EnvironmentAutotuneEnabled =
        !string.Equals(Environment.GetEnvironmentVariable(AutotuneEnvironmentVariable), "0", StringComparison.Ordinal);
    private static readonly int EnvironmentCacheCapacity = ReadCacheCapacity();

    /// <summary>Test-only override. Null restores environment-based behavior.</summary>
    internal static bool? TestOverride { get; set; }

    internal static bool IsEnabled => IsAttentionEnabled;

    internal static bool IsAttentionEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentAttentionEnabled);

    internal static bool IsResidualRmsNormEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentResidualRmsNormEnabled);

    internal static bool IsFlashDecodeEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentFlashDecodeEnabled);

    internal static bool IsPagedDecodeEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentPagedDecodeEnabled);

    internal static bool IsPagedPrefillEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentPagedPrefillEnabled);

    internal static bool IsAttentionBackwardEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentAttentionBackwardEnabled);

    internal static bool IsFlashAttentionBackwardEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentFlashAttentionBackwardEnabled);

    internal static bool IsQkvRopeCacheEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentQkvRopeCacheEnabled);

    internal static bool IsRecurrentStateEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentRecurrentStateEnabled);

    internal static bool IsAutotuneEnabled => EnvironmentAutotuneEnabled;

    internal static int CacheCapacity => EnvironmentCacheCapacity;

    private static bool ReadEnabled(string variable) =>
        string.Equals(Environment.GetEnvironmentVariable(variable), "1", StringComparison.Ordinal);

    private static int ReadCacheCapacity()
    {
        string? text = Environment.GetEnvironmentVariable(CacheCapacityEnvironmentVariable);
        return int.TryParse(text, out int value) && value is >= 4 and <= 256 ? value : 32;
    }
}

internal enum DirectPtxPhysicalType
{
    Float16,
    BFloat16,
    Float32,
    Int32
}

internal enum DirectPtxPhysicalLayout
{
    /// <summary>Dense row-major [batch, head, sequence, dimension].</summary>
    Bhsd,
    /// <summary>Dense row-major [row, feature].</summary>
    RowMajor2D,
    /// <summary>Dense row-major [sequence, head, dimension].</summary>
    SequenceHeadDim,
    /// <summary>Dense [row, qkv, head, feature] projection output.</summary>
    PackedQkv,
    /// <summary>Output-major packed Q/K/V projection weights, [qkv,head,feature,input].</summary>
    PackedQkvWeights,
    /// <summary>Packed Q/K/V projection bias, [qkv,head,feature].</summary>
    PackedQkvBias,
    /// <summary>Dense additive attention bias, [H,Sq,Skv] or [B,H,Sq,Skv].</summary>
    AttentionBias,
    /// <summary>One-dimensional canonical vector.</summary>
    Vector,
    /// <summary>Block table plus packed pages for decode attention.</summary>
    PagedKv,
    /// <summary>Dense row-major [batch, sequence, feature].</summary>
    BatchSequenceFeature
}

/// <summary>
/// Capability token for an already-validated device allocation. It contains
/// no strides: construction proves that the pointer obeys the specialization's
/// canonical physical layout, dtype, byte extent, and alignment.
/// </summary>
internal readonly struct DirectPtxTensorView
{
    internal IntPtr Pointer { get; }
    internal nuint ByteLength { get; }
    internal nuint AllocationByteLength { get; }
    internal DirectPtxPhysicalType PhysicalType { get; }
    internal DirectPtxPhysicalLayout Layout { get; }
    internal DirectPtxExtent LogicalExtent { get; }
    internal DirectPtxExtent PhysicalExtent { get; }
    internal DirectPtxTensorAccess Access { get; }

    private DirectPtxTensorView(
        IntPtr pointer,
        nuint byteLength,
        nuint allocationByteLength,
        DirectPtxPhysicalType physicalType,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent logicalExtent,
        DirectPtxExtent physicalExtent,
        DirectPtxTensorAccess access)
    {
        Pointer = pointer;
        ByteLength = byteLength;
        AllocationByteLength = allocationByteLength;
        PhysicalType = physicalType;
        Layout = layout;
        LogicalExtent = logicalExtent;
        PhysicalExtent = physicalExtent;
        Access = access;
    }

    internal static DirectPtxTensorView Create(
        IGpuBuffer buffer,
        DirectPtxTensorContract contract,
        nuint byteOffset = 0)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        if (buffer.Handle == IntPtr.Zero)
            throw new ArgumentException("The GPU buffer has no device pointer.", nameof(buffer));
        nuint allocationBytes = checked((nuint)buffer.SizeInBytes);
        nuint end = checked(byteOffset + contract.RequiredBytes);
        if (end > allocationBytes ||
            (contract.ExtentMode == DirectPtxExtentMode.Exact && end != allocationBytes))
            throw new ArgumentException(
                $"Tensor '{contract.Name}' requires {contract.RequiredBytes} bytes at offset {byteOffset}; allocation has {allocationBytes}.",
                nameof(buffer));
        nuint pointerValue = checked(PtxCompat.ToNuint(buffer.Handle) + byteOffset);
        if ((pointerValue & (nuint)(contract.AlignmentBytes - 1)) != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' is not {contract.AlignmentBytes}-byte aligned.", nameof(buffer));
        if (byteOffset % (nuint)contract.ElementBytes != 0 || allocationBytes % (nuint)contract.ElementBytes != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' extent/offset is incompatible with {contract.PhysicalType}.", nameof(buffer));

        return new DirectPtxTensorView(
            PtxCompat.ToIntPtr(pointerValue),
            contract.RequiredBytes,
            allocationBytes,
            contract.PhysicalType,
            contract.Layout,
            contract.LogicalExtent,
            contract.PhysicalExtent,
            contract.Access);
    }

    internal static DirectPtxTensorView CreateBhsd(
        IGpuBuffer buffer,
        DirectPtxPhysicalType physicalType,
        nuint requiredBytes,
        int requiredAlignment = 16)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        if (buffer.Handle == IntPtr.Zero)
            throw new ArgumentException("The GPU buffer has no device pointer.", nameof(buffer));
        if (requiredBytes == 0 || checked((nuint)buffer.SizeInBytes) < requiredBytes)
            throw new ArgumentException(
                $"The GPU buffer has {buffer.SizeInBytes} bytes; the canonical BHSD view requires {requiredBytes}.",
                nameof(buffer));
        if (requiredAlignment <= 0 || (requiredAlignment & (requiredAlignment - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(requiredAlignment), "Alignment must be a power of two.");
        if ((PtxCompat.ToNuint(buffer.Handle) & (nuint)(requiredAlignment - 1)) != 0)
            throw new ArgumentException(
                $"The GPU pointer is not {requiredAlignment}-byte aligned.", nameof(buffer));

        long elementBytes = physicalType is DirectPtxPhysicalType.Float16 or DirectPtxPhysicalType.BFloat16 ? 2L : 4L;
        if (buffer.SizeInBytes % elementBytes != 0)
            throw new ArgumentException("The buffer byte extent is incompatible with its physical dtype.", nameof(buffer));

        int elements = checked((int)(requiredBytes / (nuint)elementBytes));
        return new DirectPtxTensorView(
            buffer.Handle, requiredBytes, checked((nuint)buffer.SizeInBytes), physicalType,
            DirectPtxPhysicalLayout.Bhsd, new DirectPtxExtent(elements),
            new DirectPtxExtent(elements), DirectPtxTensorAccess.ReadWrite);
    }

    internal static DirectPtxTensorView CreateOwned(
        DirectPtxBuffer buffer,
        DirectPtxPhysicalType physicalType,
        nuint requiredBytes)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        if (buffer.Pointer == IntPtr.Zero || buffer.ByteLength < requiredBytes)
            throw new ArgumentException("The direct PTX buffer is smaller than the canonical BHSD view.", nameof(buffer));
        if ((PtxCompat.ToNuint(buffer.Pointer) & 15u) != 0)
            throw new ArgumentException("The direct PTX buffer is not 16-byte aligned.", nameof(buffer));
        int elementBytes = physicalType is DirectPtxPhysicalType.Float16 or DirectPtxPhysicalType.BFloat16 ? 2 : 4;
        int elements = checked((int)(requiredBytes / (nuint)elementBytes));
        return new DirectPtxTensorView(
            buffer.Pointer, requiredBytes, buffer.ByteLength, physicalType,
            DirectPtxPhysicalLayout.Bhsd, new DirectPtxExtent(elements),
            new DirectPtxExtent(elements), DirectPtxTensorAccess.ReadWrite);
    }

    internal static DirectPtxTensorView CreateOwned(
        DirectPtxBuffer buffer,
        DirectPtxTensorContract contract,
        nuint byteOffset = 0)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        nuint end = checked(byteOffset + contract.RequiredBytes);
        if (buffer.Pointer == IntPtr.Zero || end > buffer.ByteLength ||
            (contract.ExtentMode == DirectPtxExtentMode.Exact && end != buffer.ByteLength))
            throw new ArgumentException(
                $"The direct PTX buffer does not satisfy tensor ABI '{contract.Name}'.", nameof(buffer));
        nuint pointer = checked(PtxCompat.ToNuint(buffer.Pointer) + byteOffset);
        if ((pointer & (nuint)(contract.AlignmentBytes - 1)) != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' is not {contract.AlignmentBytes}-byte aligned.", nameof(buffer));
        return new DirectPtxTensorView(
            PtxCompat.ToIntPtr(pointer), contract.RequiredBytes, buffer.ByteLength,
            contract.PhysicalType, contract.Layout, contract.LogicalExtent,
            contract.PhysicalExtent, contract.Access);
    }
}
