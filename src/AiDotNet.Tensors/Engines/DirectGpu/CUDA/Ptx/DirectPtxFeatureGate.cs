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
    internal const string CastFp16EnvironmentVariable = "AIDOTNET_DIRECT_PTX_CAST_FP16";
    internal const string CastFp32EnvironmentVariable = "AIDOTNET_DIRECT_PTX_CAST_FP32";
    internal const string Transpose2DEnvironmentVariable = "AIDOTNET_DIRECT_PTX_TRANSPOSE2D";
    internal const string SgdMomentumEnvironmentVariable = "AIDOTNET_DIRECT_PTX_SGD_MOMENTUM";
    internal const string GlobalAvgPoolEnvironmentVariable = "AIDOTNET_DIRECT_PTX_GLOBAL_AVGPOOL";
    internal const string ComplexMultiplyEnvironmentVariable = "AIDOTNET_DIRECT_PTX_COMPLEX_MULTIPLY";
    internal const string QkvRopeCacheEnvironmentVariable = "AIDOTNET_DIRECT_PTX_QKV_ROPE_CACHE";
    internal const string ReductionEnvironmentVariable = "AIDOTNET_DIRECT_PTX_REDUCTION";
    internal const string Cholesky4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_CHOLESKY_4X4";
    internal const string LuFactor4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_LU_FACTOR_4X4";
    internal const string Qr4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_QR_4X4";
    internal const string Eigh4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_EIGH_4X4";
    internal const string Svd4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_SVD_4X4";
    internal const string LuSolve4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_LU_SOLVE_4X4";
    internal const string LdlFactor4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_LDL_FACTOR_4X4";
    internal const string LdlSolve4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_LDL_SOLVE_4X4";
    internal const string Solve4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_SOLVE_4X4";
    internal const string TriangularSolve4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_TRIANGULAR_SOLVE_4X4";
    internal const string SolverBackward4x4EnvironmentVariable = "AIDOTNET_DIRECT_PTX_SOLVER_BACKWARD_4X4";
    internal const string RngDropoutEnvironmentVariable = "AIDOTNET_DIRECT_PTX_RNG_DROPOUT";
    internal const string VisionBoxIouEnvironmentVariable = "AIDOTNET_DIRECT_PTX_VISION_BOX_IOU";
    internal const string VisionEnvironmentVariable = "AIDOTNET_DIRECT_PTX_VISION";
    internal const string RecurrentStateEnvironmentVariable = "AIDOTNET_DIRECT_PTX_RECURRENT_STATE";
    internal const string ConvolutionEnvironmentVariable = "AIDOTNET_DIRECT_PTX_CONVOLUTION";
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
    private static readonly bool EnvironmentReductionEnabled = ReadEnabled(ReductionEnvironmentVariable);
    private static readonly bool EnvironmentCholesky4x4Enabled = ReadEnabled(Cholesky4x4EnvironmentVariable);
    private static readonly bool EnvironmentLuFactor4x4Enabled = ReadEnabled(LuFactor4x4EnvironmentVariable);
    private static readonly bool EnvironmentQr4x4Enabled = ReadEnabled(Qr4x4EnvironmentVariable);
    private static readonly bool EnvironmentEigh4x4Enabled = ReadEnabled(Eigh4x4EnvironmentVariable);
    private static readonly bool EnvironmentSvd4x4Enabled = ReadEnabled(Svd4x4EnvironmentVariable);
    private static readonly bool EnvironmentLuSolve4x4Enabled = ReadEnabled(LuSolve4x4EnvironmentVariable);
    private static readonly bool EnvironmentLdlFactor4x4Enabled = ReadEnabled(LdlFactor4x4EnvironmentVariable);
    private static readonly bool EnvironmentLdlSolve4x4Enabled = ReadEnabled(LdlSolve4x4EnvironmentVariable);
    private static readonly bool EnvironmentSolve4x4Enabled = ReadEnabled(Solve4x4EnvironmentVariable);
    private static readonly bool EnvironmentTriangularSolve4x4Enabled = ReadEnabled(TriangularSolve4x4EnvironmentVariable);
    private static readonly bool EnvironmentSolverBackward4x4Enabled = ReadEnabled(SolverBackward4x4EnvironmentVariable);
    private static readonly bool EnvironmentAttentionBackwardEnabled = ReadEnabled(AttentionBackwardEnvironmentVariable);
    private static readonly bool EnvironmentRngDropoutEnabled = ReadEnabled(RngDropoutEnvironmentVariable);
    private static readonly bool EnvironmentFlashAttentionBackwardEnabled = ReadEnabled(FlashAttentionBackwardEnvironmentVariable);
    private static readonly bool EnvironmentCastFp16Enabled = ReadEnabled(CastFp16EnvironmentVariable);
    private static readonly bool EnvironmentCastFp32Enabled = ReadEnabled(CastFp32EnvironmentVariable);
    private static readonly bool EnvironmentTranspose2DEnabled = ReadEnabled(Transpose2DEnvironmentVariable);
    private static readonly bool EnvironmentSgdMomentumEnabled = ReadEnabled(SgdMomentumEnvironmentVariable);
    private static readonly bool EnvironmentGlobalAvgPoolEnabled = ReadEnabled(GlobalAvgPoolEnvironmentVariable);
    private static readonly bool EnvironmentComplexMultiplyEnabled = ReadEnabled(ComplexMultiplyEnvironmentVariable);
    private static readonly bool EnvironmentQkvRopeCacheEnabled = ReadEnabled(QkvRopeCacheEnvironmentVariable);
    private static readonly bool EnvironmentVisionBoxIouEnabled = ReadEnabled(VisionBoxIouEnvironmentVariable);
    private static readonly bool EnvironmentVisionEnabled = ReadEnabled(VisionEnvironmentVariable);
    private static readonly bool[] EnvironmentVisionOperationEnabled = ReadVisionOperationGates();
    private static readonly bool EnvironmentRecurrentStateEnabled = ReadEnabled(RecurrentStateEnvironmentVariable);
    private static readonly bool EnvironmentConvolutionEnabled = ReadEnabled(ConvolutionEnvironmentVariable);
    private static readonly bool EnvironmentAutotuneEnabled =
        !string.Equals(Environment.GetEnvironmentVariable(AutotuneEnvironmentVariable), "0", StringComparison.Ordinal);
    private static readonly int EnvironmentCacheCapacity = ReadCacheCapacity();

    /// <summary>Test-only override. Null restores environment-based behavior.</summary>
    internal static bool? TestOverride { get; set; }
    /// <summary>Benchmark-only access to reduction cells that have not passed promotion.</summary>
    internal static bool ReductionExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to mean/max/min/sum-of-squares row cells that have not passed promotion.</summary>
    internal static bool RowReduceOpExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to cast cells that have not passed promotion.</summary>
    internal static bool CastFp16ExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to widening-cast cells that have not passed promotion.</summary>
    internal static bool CastFp32ExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to transpose cells that have not passed promotion.</summary>
    internal static bool Transpose2DExperimentOverride { get; set; }
    [ThreadStatic]
    private static bool s_sgdMomentumExperimentOverride;

    /// <summary>
    /// Benchmark-only access to SGD-momentum cells that have not passed promotion. Thread-local
    /// state prevents parallel tests or benchmarks from enabling an experimental route in an
    /// unrelated dispatcher, matching the global-average-pool and complex-multiply overrides.
    /// </summary>
    /// <remarks>
    /// As a process-global static this defeated the fail-closed guarantee: while one test held it
    /// set, every other thread could dispatch the unpromoted SGD route, which also made concurrent
    /// results order-dependent.
    /// </remarks>
    internal static bool SgdMomentumExperimentOverride
    {
        get => s_sgdMomentumExperimentOverride;
        set => s_sgdMomentumExperimentOverride = value;
    }
    [ThreadStatic]
    private static bool s_globalAvgPoolExperimentOverride;

    /// <summary>
    /// Benchmark-only access to global-average-pool cells that have not passed
    /// promotion. Thread-local state prevents parallel tests or benchmarks from
    /// enabling an experimental route in an unrelated dispatcher.
    /// </summary>
    internal static bool GlobalAvgPoolExperimentOverride
    {
        get => s_globalAvgPoolExperimentOverride;
        set => s_globalAvgPoolExperimentOverride = value;
    }
    /// <summary>Benchmark-only access to measured cells that have not passed promotion.</summary>
    internal static bool FusedLinearExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to mixed-precision cells that have not passed promotion.</summary>
    internal static bool MixedPrecisionLinearExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to quantized cells that have not passed promotion.</summary>
    internal static bool QuantizedLinearExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to normalization cells that have not passed promotion.</summary>
    internal static bool NormalizationExperimentOverride { get; set; }
    /// <summary>Benchmark-only access to convolution cells that have not passed promotion.</summary>
    internal static bool ConvolutionExperimentOverride { get; set; }

    [ThreadStatic]
    private static bool? _visionExperimentOverride;

    [ThreadStatic]
    private static bool? _visionGateOverride;

    /// <summary>Thread-isolated static/driver-test opt-in for the unpromoted specialization.</summary>
    internal static bool? VisionExperimentOverride
    {
        get => _visionExperimentOverride;
        set => _visionExperimentOverride = value;
    }

    /// <summary>Benchmark-only route selector; false forces the established backend.</summary>
    internal static bool? VisionGateOverride
    {
        get => _visionGateOverride;
        set => _visionGateOverride = value;
    }

    [ThreadStatic]
    private static bool s_complexMultiplyExperimentOverride;
    [ThreadStatic]
    private static bool? s_complexMultiplyGateOverride;

    /// <summary>Thread-local benchmark access to unpromoted complex-multiply cells.</summary>
    internal static bool ComplexMultiplyExperimentOverride
    {
        get => s_complexMultiplyExperimentOverride;
        set => s_complexMultiplyExperimentOverride = value;
    }

    /// <summary>
    /// Thread-local benchmark/test selection of the candidate or established
    /// route. Null restores process-start feature configuration.
    /// </summary>
    internal static bool? ComplexMultiplyGateOverride
    {
        get => s_complexMultiplyGateOverride;
        set => s_complexMultiplyGateOverride = value;
    }

    [ThreadStatic] private static bool? s_cholesky4x4ExperimentOverride;
    [ThreadStatic] private static bool? s_solver4x4ExperimentOverride;

    /// <summary>Thread-local benchmark/test override; null restores the process-start gate.</summary>
    internal static bool? Cholesky4x4ExperimentOverride
    {
        get => s_cholesky4x4ExperimentOverride;
        set => s_cholesky4x4ExperimentOverride = value;
    }

    /// <summary>Thread-local override for the non-Cholesky solver experiment family.</summary>
    internal static bool? Solver4x4ExperimentOverride
    {
        get => s_solver4x4ExperimentOverride;
        set => s_solver4x4ExperimentOverride = value;
    }

    internal static bool IsEnabled => IsAttentionEnabled;
    internal static bool IsRngDropoutEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentRngDropoutEnabled);

    internal static bool IsAttentionEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentAttentionEnabled);

    internal static bool IsResidualRmsNormEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentResidualRmsNormEnabled);

    internal static bool IsFlashDecodeEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentFlashDecodeEnabled);

    internal static bool IsReductionEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentReductionEnabled);

    internal static bool IsPagedDecodeEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentPagedDecodeEnabled);

    internal static bool IsPagedPrefillEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentPagedPrefillEnabled);

    internal static bool IsAttentionBackwardEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentAttentionBackwardEnabled);

    internal static bool IsFlashAttentionBackwardEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentFlashAttentionBackwardEnabled);

    internal static bool IsCastFp16Enabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentCastFp16Enabled);

    internal static bool IsCastFp32Enabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentCastFp32Enabled);

    internal static bool IsTranspose2DEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentTranspose2DEnabled);

    internal static bool IsSgdMomentumEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentSgdMomentumEnabled);
    internal static bool IsGlobalAvgPoolEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentGlobalAvgPoolEnabled);
    internal static bool IsComplexMultiplyEnabled => ComplexMultiplyGateOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentComplexMultiplyEnabled);

    internal static bool IsQkvRopeCacheEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentQkvRopeCacheEnabled);

    internal static bool IsCholesky4x4Enabled => Cholesky4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentCholesky4x4Enabled);

    internal static bool IsLuFactor4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentLuFactor4x4Enabled);

    internal static bool IsQr4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentQr4x4Enabled);

    internal static bool IsEigh4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentEigh4x4Enabled);

    internal static bool IsSvd4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentSvd4x4Enabled);

    internal static bool IsLuSolve4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentLuSolve4x4Enabled);

    internal static bool IsLdlFactor4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentLdlFactor4x4Enabled);

    internal static bool IsLdlSolve4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentLdlSolve4x4Enabled);

    internal static bool IsSolve4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentSolve4x4Enabled);

    internal static bool IsTriangularSolve4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentTriangularSolve4x4Enabled);

    internal static bool IsSolverBackward4x4Enabled => Solver4x4ExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentSolverBackward4x4Enabled);
    internal static bool IsVisionBoxIouEnabled => VisionGateOverride ??
        VisionExperimentOverride ?? TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentVisionEnabled || EnvironmentVisionBoxIouEnabled);

    internal static bool IsVisionOperationEnabled(DirectPtxVisionOperation operation)
    {
        int ordinal = (int)operation;
        bool operationEnabled = (uint)ordinal < (uint)EnvironmentVisionOperationEnabled.Length &&
            EnvironmentVisionOperationEnabled[ordinal];
        return VisionGateOverride ?? VisionExperimentOverride ?? TestOverride ??
            (EnvironmentMasterEnabled || EnvironmentVisionEnabled || operationEnabled);
    }
    internal static bool IsRecurrentStateEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentRecurrentStateEnabled);
    /// <summary>Softmax-family (issue #840) rollout gate; disabled by default.</summary>
    internal static bool IsSoftmaxEnabled => TestOverride ?? EnvironmentMasterEnabled;

    /// <summary>
    /// Opt-in switch that lets fail-closed softmax-family specializations dispatch for GPU
    /// validation before a shape is performance-promoted. Off in production.
    /// </summary>
    internal static bool SoftmaxExperimentOverride { get; set; }
    /// <summary>Specialized-scientific (issue #854) rollout gate; disabled by default.</summary>
    internal static bool IsScientificEnabled => TestOverride ?? EnvironmentMasterEnabled;

    /// <summary>
    /// Opt-in switch that lets fail-closed scientific specializations dispatch for GPU
    /// validation before a shape is performance-promoted. Off in production.
    /// </summary>
    internal static bool ScientificExperimentOverride { get; set; }

    internal static bool IsConvolutionEnabled => TestOverride ??
        (EnvironmentMasterEnabled || EnvironmentConvolutionEnabled);

    internal static bool IsAutotuneEnabled => EnvironmentAutotuneEnabled;

    internal static int CacheCapacity => EnvironmentCacheCapacity;

    private static bool ReadEnabled(string variable) =>
        string.Equals(Environment.GetEnvironmentVariable(variable), "1", StringComparison.Ordinal);

    private static bool[] ReadVisionOperationGates()
    {
        Array values = Enum.GetValues(typeof(DirectPtxVisionOperation));
        int maximum = 0;
        foreach (DirectPtxVisionOperation operation in values)
            maximum = Math.Max(maximum, (int)operation);
        var enabled = new bool[maximum + 1];
        foreach (DirectPtxVisionOperation operation in values)
        {
            string suffix = VisionGateSuffix(operation);
            enabled[(int)operation] = ReadEnabled(VisionEnvironmentVariable + "_" + suffix);
        }
        return enabled;
    }

    private static string VisionGateSuffix(DirectPtxVisionOperation operation) => operation switch
    {
        DirectPtxVisionOperation.GeneralizedBoxIou => "GENERALIZED_BOX_IOU",
        DirectPtxVisionOperation.DistanceBoxIou => "DISTANCE_BOX_IOU",
        DirectPtxVisionOperation.CompleteBoxIou => "COMPLETE_BOX_IOU",
        DirectPtxVisionOperation.BoxArea => "BOX_AREA",
        DirectPtxVisionOperation.BoxConvert => "BOX_CONVERT",
        DirectPtxVisionOperation.IoULoss => "IOU_LOSS",
        DirectPtxVisionOperation.GIoULoss => "GIOU_LOSS",
        DirectPtxVisionOperation.DIoULoss => "DIOU_LOSS",
        DirectPtxVisionOperation.CIoULoss => "CIOU_LOSS",
        DirectPtxVisionOperation.IoULossBackward => "IOU_LOSS_BACKWARD",
        DirectPtxVisionOperation.GIoULossBackward => "GIOU_LOSS_BACKWARD",
        DirectPtxVisionOperation.DIoULossBackward => "DIOU_LOSS_BACKWARD",
        DirectPtxVisionOperation.CIoULossBackward => "CIOU_LOSS_BACKWARD",
        DirectPtxVisionOperation.IouFamilyBackwardA => "IOU_FAMILY_BACKWARD_A",
        DirectPtxVisionOperation.IouFamilyBackwardB => "IOU_FAMILY_BACKWARD_B",
        DirectPtxVisionOperation.Nms => "NMS",
        DirectPtxVisionOperation.MasksToBoxes => "MASKS_TO_BOXES",
        DirectPtxVisionOperation.RoiAlign => "ROI_ALIGN",
        DirectPtxVisionOperation.RoiPool => "ROI_POOL",
        DirectPtxVisionOperation.PsRoiAlign => "PS_ROI_ALIGN",
        DirectPtxVisionOperation.PsRoiPool => "PS_ROI_POOL",
        DirectPtxVisionOperation.Cross3 => "CROSS3",
        DirectPtxVisionOperation.Meshgrid2D => "MESHGRID_2D",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    private static int ReadCacheCapacity()
    {
        string? text = Environment.GetEnvironmentVariable(CacheCapacityEnvironmentVariable);
        return int.TryParse(text, out int value) && value is >= 4 and <= 256 ? value : 32;
    }
}

internal enum DirectPtxPhysicalType
{
    Int8,
    Float16,
    BFloat16,
    Float32,
    Int32,
    UInt8
}

internal enum DirectPtxPhysicalLayout
{
    /// <summary>Dense row-major [batch, head, sequence, dimension].</summary>
    Bhsd,
    /// <summary>Dense row-major [row, feature].</summary>
    RowMajor2D,
    /// <summary>Dense row-major [batch, row, column] matrices.</summary>
    BatchedRowMajorMatrix,
    /// <summary>Dense row-major [dim0, dim1, dim2].</summary>
    RowMajor3D,
    /// <summary>Dense row-major [sequence, head, dimension].</summary>
    SequenceHeadDim,
    /// <summary>Dense [row, qkv, head, feature] projection output.</summary>
    PackedQkv,
    /// <summary>Output-major packed Q/K/V projection weights, [qkv,head,feature,input].</summary>
    PackedQkvWeights,
    /// <summary>Packed Q/K/V projection bias, [qkv,head,feature].</summary>
    PackedQkvBias,
    /// <summary>Input-major row-major linear weights, [inputFeature,outputFeature].</summary>
    LinearWeightInputMajor,
    /// <summary>Output-major row-major linear weights, [outputFeature,inputFeature].</summary>
    LinearWeightOutputMajor,
    /// <summary>Dense additive attention bias, [H,Sq,Skv] or [B,H,Sq,Skv].</summary>
    AttentionBias,
    /// <summary>One-dimensional canonical vector.</summary>
    Vector,
    /// <summary>Dense row-major bounding boxes in canonical XYXY order.</summary>
    BoxXyxy,
    /// <summary>Dense row-major bounding boxes in XYWH order.</summary>
    BoxXywh,
    /// <summary>Dense row-major bounding boxes in center-X/center-Y/width/height order.</summary>
    BoxCxcywh,
    /// <summary>Dense images with batch/channel/height/width order.</summary>
    Nchw,
    /// <summary>Dense images with batch/height/width/channel order.</summary>
    Nhwc,
    /// <summary>Dense normalized sampling coordinates ending in 2 or 3.</summary>
    SamplingGrid,
    /// <summary>ROI rows [batchIndex,x1,y1,x2,y2].</summary>
    RoiBoxes,
    /// <summary>Block table plus packed pages for decode attention.</summary>
    PagedKv,
    /// <summary>Dense row-major [batch, sequence, feature].</summary>
    BatchSequenceFeature,
    /// <summary>Dense output/input/spatial convolution weights [output, input, height, width].</summary>
    Oihw,
    /// <summary>Dense input/output/spatial transposed-convolution weights [input, output, height, width].</summary>
    Iohw
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
        DirectPtxTensorContract contract)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        nuint byteOffset = contract.ByteOffset;
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

    internal static DirectPtxTensorView Create(
        DirectPtxBuffer buffer,
        DirectPtxTensorContract contract)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        if (buffer.Pointer == IntPtr.Zero)
            throw new ArgumentException("The direct PTX buffer has no device pointer.", nameof(buffer));
        if (buffer.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' requires exactly {contract.RequiredBytes} bytes; allocation has {buffer.ByteLength}.",
                nameof(buffer));
        nuint pointerValue = PtxCompat.ToNuint(buffer.Pointer);
        if ((pointerValue & (nuint)(contract.AlignmentBytes - 1)) != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' is not {contract.AlignmentBytes}-byte aligned.", nameof(buffer));
        return new DirectPtxTensorView(
            buffer.Pointer,
            contract.RequiredBytes,
            buffer.ByteLength,
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
        DirectPtxTensorContract contract)
    {
        PtxCompat.ThrowIfNull(buffer, nameof(buffer));
        nuint byteOffset = contract.ByteOffset;
        nuint end = checked(byteOffset + contract.RequiredBytes);
        if (buffer.Pointer == IntPtr.Zero || end > buffer.ByteLength ||
            (contract.ExtentMode == DirectPtxExtentMode.Exact && end != buffer.ByteLength))
            throw new ArgumentException(
                $"The direct PTX buffer does not satisfy tensor ABI '{contract.Name}'.", nameof(buffer));
        nuint pointer = checked(PtxCompat.ToNuint(buffer.Pointer) + byteOffset);
        if ((pointer & (nuint)(contract.AlignmentBytes - 1)) != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' is not {contract.AlignmentBytes}-byte aligned.", nameof(buffer));
        if (byteOffset % (nuint)contract.ElementBytes != 0 ||
            buffer.ByteLength % (nuint)contract.ElementBytes != 0)
            throw new ArgumentException(
                $"Tensor '{contract.Name}' extent/offset is incompatible with {contract.PhysicalType}.", nameof(buffer));
        return new DirectPtxTensorView(
            PtxCompat.ToIntPtr(pointer), contract.RequiredBytes, buffer.ByteLength,
            contract.PhysicalType, contract.Layout, contract.LogicalExtent,
            contract.PhysicalExtent, contract.Access);
    }
}
