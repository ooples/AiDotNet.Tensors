using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Static catalog classifying every IEngine Tensor-returning method as Differentiable,
/// NonDifferentiable, or Delegator. Used by TapeCompletenessTests to enforce that no
/// future op can silently skip gradient tape recording.
/// </summary>
/// <remarks>
/// <para>When adding a new Tensor-returning method to IEngine, you MUST add it to exactly
/// one of the three sets below. The TapeCompletenessTests reflection test will fail CI
/// if any method is unclassified.</para>
/// </remarks>
internal static class OpRegistry
{
    /// <summary>
    /// Ops that have DifferentiableOps.Record* calls and BackwardFunction delegates.
    /// These participate in gradient computation when a tape is active.
    /// </summary>
    internal static readonly HashSet<string> DifferentiableOps = new(StringComparer.Ordinal)
    {
        // Arithmetic
        "TensorAdd", "TensorSubtract", "TensorMultiply", "TensorDivide",
        "TensorNegate", "TensorAbs", "TensorSign",
        "TensorMultiplyScalar", "TensorDivideScalar", "TensorAddScalar", "TensorSubtractScalar",
        "TensorAddMany", "TensorMultiplyMany", "TensorAddScaled",
        "TensorBroadcastAdd", "TensorBroadcastSubtract", "TensorBroadcastMultiply", "TensorBroadcastDivide",
        "TensorMax", "TensorMin", "TensorClamp",

        // Math
        "TensorExp", "TensorLog", "TensorSqrt", "TensorPower", "TensorPowerTensor",
        "TensorSin", "TensorCos", "TensorCosh", "TensorSinh",
        "TensorFrac", "TensorPow",

        // Matrix
        "TensorMatMul", "BatchMatMul", "TensorOuterProduct", "TensorBatchOuterProduct", "TensorOuter",

        // Activations
        "ReLU", "Sigmoid", "Tanh", "GELU", "LeakyReLU", "Swish", "Mish",
        "Softplus", "SELU", "ELU", "HardSigmoid", "HardSwish", "ReLU6",
        "Softmax", "LogSoftmax",
        "GLU", "Sparsemax", "TaylorSoftmax",

        // Shape
        "Reshape", "TensorTranspose", "TensorPermute", "TensorSqueeze", "TensorExpandDims",
        "TensorFlatten", "TensorSlice", "TensorSliceAxis", "TensorNarrow",
        "TensorConcatenate", "Concat", "TensorSplit", "TensorStack",
        "TensorTile", "TensorDiagonal",

        // Reduction
        "ReduceSum", "ReduceMean", "ReduceMax", "ReduceLogVariance",
        "TensorMean", "Std", "Var",

        // Pooling
        "MaxPool2D", "AvgPool2D", "MaxPool2DWithIndices",
        "MaxPool3DWithIndices", "AvgPool3D",
        "MaxPool1D", "AvgPool1D",
        "AdaptiveAvgPool2D", "AdaptiveMaxPool2D",

        // Convolution
        "Conv2D", "Conv1D", "Conv3D",
        "DepthwiseConv2D", "ConvTranspose2D", "ConvTranspose3D", "LocallyConnectedConv2D",

        // Normalization
        "BatchNorm", "LayerNorm", "GroupNorm", "RMSNorm", "InstanceNorm",

        // Attention/Embedding
        "Embedding", "Dropout",
        "GridSample", "Unfold", "Fold",

        // Spatial
        "Upsample", "Upsample3D", "PixelShuffle", "Crop", "Pad",
        "TensorCumSum", "TensorRepeatElements",

        // Scatter/Gather
        "Gather", "Scatter", "ScatterAdd", "ScatterMean", "ScatterMax", "ScatterSoftmax",
        "TensorIndexSelect", "TensorMaskedFill", "TensorMaskedSelect",

        // Parity-210 movement / cumulative / clamp
        "TensorRoll", "TensorFlip", "TensorRepeatInterleave",
        "TensorDot", "TensorVecDot", "TensorCross", "TensorMeshgrid",
        "TensorMultiDot", "TensorTrace", "TensorDiagEmbed", "TensorAddMM",
        "TensorCartesianProd", "TensorKron", "TensorInner",
        "TensorPDist", "TensorCDist", "TensorCosineSimilarity",
        "TensorFliplr", "TensorFlipud", "TensorRot90",
        "TensorSwapAxes", "TensorMoveDim",
        "TensorAtLeast1D", "TensorAtLeast2D", "TensorAtLeast3D",
        "TensorHStack", "TensorVStack", "TensorDStack",
        "TensorColumnStack", "TensorRowStack",
        "TensorHSplit", "TensorVSplit", "TensorDSplit",
        "TensorCumProd", "TensorCumMax", "TensorCumMin", "TensorLogCumSumExp",
        "TensorClampMin", "TensorClampMax", "TensorClampTensor",
        "TensorSelectScatter",

        // Parity-210 element-wise binary (forward only in v1; backward TBD)
        "TensorHypot", "TensorCopysign", "TensorFmod", "TensorRemainder",
        "TensorFloatPower", "TensorLdexp", "TensorNextAfter",
        "TensorLogAddExp", "TensorLogAddExp2",

        // Parity-210 special math (forward only in v1)
        "TensorErfc", "TensorXlogy", "TensorXlog1py",
        "TensorLgamma", "TensorDigamma",
        "TensorErfinv", "TensorI0", "TensorI1", "TensorI0e", "TensorI1e",
        "TensorFrexp", "TensorPolygamma",

        // Parity-210 sort / order stats (non-differentiable category below
        // also lists the scalar-returning ones; the tensor-returning ones
        // sit here as stubs with no backward).
        "TensorSort",

        // Parity-210 indexing family (forward only in v1)
        "TensorIndexAdd", "TensorIndexFill", "TensorIndexCopy", "TensorIndexPut",
        "TensorGatherPacked", "TensorScatterPacked",
        "TensorMaskedScatter", "TensorScatterReduce",
        // TensorBroadcastTo belongs to the Delegator set below (composed
        // from Reshape / TensorBroadcastAdd which record themselves) —
        // listing it here too tripped the duplicate-check in
        // TapeCompletenessTests.OpRegistry_HasNoDuplicates.
        "TensorExpandAs", "TensorBroadcastTensors",
        "TensorTake", "TensorTakeAlongDim", "TensorPut",
        "TensorBlockDiag", "TensorSliceScatter",

        // Parity-210 ops with autograd added in 2026-04
        "TensorTriu", "TensorTril",
        "TensorNanToNum",
        "TensorZeta", "TensorUnfold", "TensorPolygamma",
        "TensorFliplr", "TensorFlipud", "TensorRot90",
        "TensorSwapAxes", "TensorMoveDim",
        "TensorAtLeast1D", "TensorAtLeast2D", "TensorAtLeast3D",
        "TensorHStack", "TensorVStack", "TensorDStack",
        "TensorColumnStack", "TensorRowStack",
        "TensorCross", "TensorDiagEmbed",
        "TensorRoll", "TensorFlip", "TensorRepeatInterleave",
        "TensorCumProd", "TensorCumMax", "TensorCumMin", "TensorLogCumSumExp",
        "TensorHypot", "TensorCopysign", "TensorFmod", "TensorRemainder",
        "TensorFloatPower", "TensorLogAddExp", "TensorLogAddExp2",
        "TensorXlogy", "TensorXlog1py", "TensorLdexp",
        "TensorErfc", "TensorErfinv", "TensorLgamma", "TensorDigamma",
        "TensorI0", "TensorI1", "TensorI0e", "TensorI1e",
        "TensorKron", "TensorInner", "TensorAddMM", "TensorDot",
        "TensorCosineSimilarity", "TensorPDist", "TensorCDist",

        // Complex
        "TensorComplexMultiply", "TensorComplexConjugate", "TensorComplexMagnitude",
        "ComplexMagnitudeSquared",

        // Loss
        "TensorMSELoss", "TensorBCEWithLogitsLoss", "TensorCrossEntropyLoss",
        "TensorBinaryCrossEntropy", "TensorL1Loss", "TensorHuberLoss",
        "TensorKLDivLoss", "TensorNLLLoss", "TensorCTCLoss",

        // Fused
        "FusedLinearReLU", "FusedLinearSigmoid", "FusedLinearTanh",
        "FusedLinearGELU", "FusedLinearSwish",

        // Signal
        "RFFT", "IRFFT",

        // Other differentiable ops
        "TensorWhere", "TensorTrilinearInterpolate",
        "RBFKernel", "OctonionMatMulTensor",
        "CosineSimilarity", "TensorReciprocal",
        "ScalarMinusTensor",
        "TensorConstantPad",
        "TensorCosineSimilarityLoss",
        "TensorUpsampleBilinear",
    };

    /// <summary>
    /// Ops that are correctly non-differentiable — comparisons, constructors, backward helpers,
    /// random generators, masks, and explicit gradient stops.
    /// </summary>
    internal static readonly HashSet<string> NonDifferentiableOps = new(StringComparer.Ordinal)
    {
        // Comparison (return bool-like tensors, not differentiable)
        "TensorEquals", "TensorNotEquals", "TensorGreaterThan", "TensorLessThan",
        "TensorGreaterOrEqual", "TensorLessOrEqual",
        "TensorIsClose", "TensorAllClose", "TensorIsIn", "TensorAminmax",
        "TensorIsFinite", "TensorIsNan", "TensorIsInf",
        "TensorEqual", "TensorEq", "TensorEqScalar",
        "TensorLogicalAnd", "TensorLogicalOr", "TensorLogicalXor", "TensorLogicalNot",
        "TensorNonzero", "TensorCountNonzero",
        "TensorKthvalue", "TensorMedian", "TensorNanMedian", "TensorMode",
        "TensorUnique", "TensorUniqueConsecutive",
        "TensorUniqueWithInfo", "TensorUniqueConsecutiveWithInfo",
        "TensorSearchSorted", "TensorBucketize",
        "TensorHistogram", "TensorHistogramDD", "TensorBinCount", "TensorHistc",
        "TensorArgsort",

        // Constructors / initializers
        "TensorRandomUniform", "TensorRandomNormal", "TensorRandomUniformRange",
        "TensorEye", "TensorDiag", "TensorOneHot", "TensorLinspace",
        "TensorTriangularMask", "TensorDropoutMask",
        "PositionalEncoding", "CreateMelFilterbank", "CreateWindow",

        // Explicit gradient stop
        "StopGradient",

        // Stochastic (Gumbel uses STE)
        "GumbelSoftmax",

        // Arbitrary user function (cannot auto-diff)
        "TensorMap",

        // In-place mutation semantics
        "TensorSetSlice",

        // Discrete selection
        "TensorTopK", "TensorScatter",
        // Shape-splitting — each split output is a differentiable slice, but
        // the multi-tensor return type is currently treated as
        // non-differentiable at the tape level.
        "TensorTensorSplit",

        // Grid/geometry (non-learnable)
        "AffineGrid",

        // NeRF/3D ops (typically not differentiated through in training)
        "VolumeRendering", "ImportanceSampling", "RasterizeGaussians",
        "EvaluateSphericalHarmonics", "ComputeGaussianCovariance",
        "MultiresolutionHashEncoding", "UpdateOccupancyGrid",

        // Backward helper methods (implementation details, not forward ops)
        "ReluBackward", "SigmoidBackward", "TanhBackward", "GeluBackward",
        "LeakyReluBackward", "SwishBackward", "MishBackward", "SoftplusBackward",
        "HardswishBackward", "SeluBackward", "HardsigmoidBackward", "Relu6Backward",
        "EluBackward", "ThresholdBackward", "ReciprocalBackward",
        "SoftmaxBackward", "TensorSoftmaxBackward",
        "GLUBackward", "GeGLUBackward", "SwiGLUBackward", "ReGLUBackward",
        "SparsemaxBackward", "TaylorSoftmaxBackward", "SphericalSoftmaxBackward",
        "GumbelSoftmaxBackward",
        "MaxPool2DBackward", "AvgPool2DBackward", "MaxPool3DBackward", "AvgPool3DBackward",
        "Conv1DBackwardInput", "Conv1DBackwardKernel",
        "Conv2DBackwardInput", "Conv2DBackwardKernel",
        "Conv3DBackwardInput", "Conv3DBackwardKernel",
        "ConvTranspose2DBackwardInput", "ConvTranspose2DBackwardKernel",
        "ConvTranspose3DBackwardInput", "ConvTranspose3DBackwardKernel",
        "DepthwiseConv2DBackwardInput", "DepthwiseConv2DBackwardKernel",
        "LocallyConnectedConv2DBackwardInput", "LocallyConnectedConv2DBackwardWeights", "LocallyConnectedConv2DBackwardBias",
        "DeformableConv2DBackwardInput", "DeformableConv2DBackwardKernel",
        "DeformableConv2DBackwardOffset", "DeformableConv2DBackwardMask",
        "GridSampleBackwardInput", "GridSampleBackwardGrid",
        "BatchNormBackward", "LayerNormBackward", "GroupNormBackward", "RMSNormBackward", "InstanceNormBackward",
        "DropoutBackward", "EmbeddingBackward",
        "ReduceMaxBackward", "ReduceMeanBackward", "ReduceVarianceBackward", "ReduceLogVarianceBackward",
        "UpsampleBackward", "Upsample3DBackward",
        "CropBackward", "PadBackward", "PixelShuffleBackward",
        "ScatterAddBackward", "ScatterMeanBackward", "ScatterMaxBackward", "ScatterSoftmaxBackward",
        "RBFKernelBackward",
        "TensorBinaryCrossEntropyBackward", "TensorTrilinearInterpolateBackward",
        "TensorSquashBackward", "TensorEmbeddingLookupBackward",
        "CrossEntropyBackward", "MseBackward",
        "ScaledDotProductAttentionBackward", "FlashAttentionBackward",
        "GraphAttentionBackward", "GroupedQueryAttentionBackward",
        "FusedLinearBackward",
        "SpiralConvBackwardInput", "SpiralConvBackwardWeights", "SpiralConvBackwardBias",
        "DiffusionConvBackward",
        "PositionalEncodingBackward",
        "VolumeRenderingBackward", "RasterizeGaussiansBackward",
        "ComputeGaussianCovarianceBackward", "EvaluateSphericalHarmonicsBackward",
        "MultiresolutionHashEncodingBackward",
        "PReLUBackward",
        "TanhDerivative", "SigmoidDerivative", "ReLUDerivative",
        "VarBackward", "StdBackward",

        // Discrete / non-differentiable outputs
        "TensorArgMax", "TensorArgMin", "ArgSort",
        "GenerateSpiralIndices",

        // Signal processing (non-differentiable spectral ops)
        "ISTFT", "MelSpectrogram", "GriffinLim",

        // Native Complex<T> operations (no backward functions implemented yet)
        "NativeComplexFFT", "NativeComplexIFFT", "NativeComplexIFFTReal",
        "NativeComplexMultiply", "NativeComplexConjugate",
        "NativeComplexScale", "NativeComplexAdd",
        "NativeComplexMagnitude", "NativeComplexMagnitudeSquared",
        "NativeComplexPhase", "NativeComplexFromPolar",
        "NativeComplexFFTComplex", "NativeComplexTopK",
        "NativeComplexCrossSpectral",
        "NativeComplexFFT2D", "NativeComplexIFFT2DReal",
        "NativeComplexFFTND", "NativeComplexIFFTNDReal",
        "NativeAnalyticSignal",
        "NativeNormalizeRows",
        "NativeComplexFFTSpan", "NativeComplexIFFTSpan", "NativeComplexFFTComplexSpan", "NativeComplexIFFTRealSpan",
        "NativeTanh", "NativeExp", "NativeAtan2", "NativeMagnitudeAndPhase",
        "NativeBispectrum", "NativeTrispectrum",
        "NativeBatchedCavityForward",
        "NativeMfccFeatures", "NativeWidebandFeatures", "NativePacFeatures",
        "TensorSoftmaxRows",

        // Rounding (non-differentiable, STE would need explicit annotation)
        "TensorFloor", "TensorCeiling", "TensorRound",

        // Layout reorders (pure data movement, not a differentiable op —
        // any gradient propagates through the inverse layout reorder)
        "ReorderToNchwc", "ReorderToNchw",

        // Inference-only BatchNorm (training-mode BatchNorm uses the
        // separate BatchNormBackward path above)
        "BatchNormInference",

        // Vision Detection — Issue #217. Forward-only in v1; backward
        // candidates (BoxIoU family is continuous in box coords and is
        // used as the DETR matching loss) come in a follow-up. NMS /
        // BatchedNms / MasksToBoxes are inherently non-differentiable
        // (discrete output / argmin).
        "BoxConvert", "BoxArea",
        "BoxIou", "GeneralizedBoxIou", "DistanceBoxIou", "CompleteBoxIou",
        "Nms", "BatchedNms",
        "MasksToBoxes",
    };

    /// <summary>
    /// Ops that delegate to another method which already records.
    /// These do NOT need their own recording to avoid double-recording.
    /// Format: "DelegatorName" — the target method handles recording.
    /// </summary>
    internal static readonly HashSet<string> DelegatorOps = new(StringComparer.Ordinal)
    {
        // IEngine wrappers that delegate to internal methods
        "TensorSigmoid",     // -> Sigmoid (records)
        "TensorReLU",        // -> ReLU (records)
        "TensorGELU",        // -> GELU (records)
        "TensorSiLU",        // -> Swish (records)
        "TensorTanh",        // -> Tanh (records)
        "TensorLeakyReLU",   // -> LeakyReLU (records)
        "TensorMish",        // -> Mish (records)
        "TensorHardSwish",   // -> HardSwish (records)
        "TensorLayerNorm",   // -> LayerNorm (records)
        "TensorMaxPool2D",   // -> MaxPool2D (records)
        "TensorAvgPool2D",   // -> AvgPool2D (records)
        "TensorConv2D",      // -> Conv2D (records)

        // Composed from recorded sub-ops (backward through constituents)
        "TensorBroadcastTo", // -> Reshape or TensorBroadcastAdd (both record)
        "TensorLogSumExp",   // ReduceMax + BroadcastSubtract + TensorExp + ReduceSum + TensorLog + TensorAdd
        "TensorNorm",        // TensorMultiply + ReduceSum + TensorSqrt
        "TensorNormalize",   // TensorNorm + TensorDivide
        "TensorClip",        // TensorClamp (records)
        "TensorSquare",      // TensorMultiply(x, x) (records)
        "TensorSquash",      // TensorMultiply + TensorAdd + TensorDivide + ReduceSum (all record)
        "TensorSoftmax",     // -> Softmax (records)
        "ReduceStd",         // ReduceVariance + TensorSqrt
        "ReduceVariance",    // ReduceMean (records) + composed
        "GlobalAvgPool2D",   // reshape + ReduceMean
        "GlobalMaxPool2D",   // reshape + ReduceMax
        "TensorLerp",        // TensorAdd + TensorSubtract + TensorMultiplyScalar
        "FusedLinear",       // TensorMatMul + TensorBroadcastAdd (both record)
        "TensorEinsum",      // TensorMatMul / BatchMatMul / Split (record)
        "OctonionAddTensor",  // TensorAdd (records)
        "GeGLU",             // composed with GELU + Tanh
        "SwiGLU",            // composed with Swish
        "ReGLU",             // composed with ReLU
        "SphericalSoftmax",  // composed with Softmax
        "PairwiseDistance",   // TensorSqrt(PairwiseDistanceSquared)
        "PairwiseDistanceSquared", // direct computation, backward via engine ops

        // MaxPool3D scalar overload delegates to array overload
        "MaxPool3D",         // -> MaxPool3D(int[]) which records via MaxPool3DWithIndices

        // Attention ops (composed from recorded sub-ops)
        "TensorScaledDotProductAttention", // TensorMatMul + TensorMultiplyScalar + Softmax
        "ScaledDotProductAttention",       // composed
        "FlashAttention",                  // composed
        "GroupedQueryAttention",           // composed
        "GraphAttention",                  // composed
        "MultiHeadGraphAttention",         // composed

        // Loss functions (use NoGradScope + single composite record)
        "TensorIoULoss",    // composed from TensorMultiply + ReLU + etc.
        "TensorGIoULoss",
        "TensorDIoULoss",
        "TensorCIoULoss",

        // Mesh/graph ops (composed)
        "SpiralConv", "DiffusionConv", "ComputeMeshLaplacian",

        // Deformable conv (complex, delegates to sub-ops)
        "DeformableConv2D",

        // Additional delegators for IEngine wrappers
        "TensorSELU",          // -> SELU (records)
        "TensorHardSigmoid",   // -> HardSigmoid (records)
        "TensorReLU6",         // -> ReLU6 (records)
        "TensorPReLU",         // -> PReLU (records)
        "TensorRReLU",         // -> RReLU (records)
        "TensorThreshold",     // -> Threshold (records)
        "TensorAvgPool1D",     // -> AvgPool1D (records)
        "TensorMaxPool1D",     // -> MaxPool1D (records)
        "TensorAdaptiveMaxPool2D", // -> AdaptiveMaxPool2D (records)
        "TensorBatchMatMul",   // -> BatchMatMul (records)
        "TensorLogSoftmax",    // -> LogSoftmax (records)
        "TensorEmbeddingLookup", // -> Embedding (records)
        "TensorScatterAdd",    // -> ScatterAdd (records)
        "TensorGather",        // -> Gather (records)
        "TensorVar",           // -> Var (records)
        "TensorStd",           // -> Std (records)
        "TensorUnstack",       // inverse of TensorStack, composed

        // Diff variants (recorded through base ops)
        "TensorMeanDiff",      // composed
        "TensorStackDiff",     // composed
        "TensorIndexSelectDiff", // composed

        // Fused ops (record as single composite entries)
        "FusedConv2D",         // composed from Conv2D + BatchNorm + activation
        "FusedConv3D",         // composed
        "FusedConvTranspose2D", // composed
        "FusedBatchNorm",      // composed from BatchNorm + activation

        // Spectral filter ops (composed from FFT2D + inline complex multiply + IFFT2DReal)
        "NativeSpectralFilter",       // FFT2D → multiply → IFFT2DReal
        "NativeSpectralFilterBatch",  // batched across [B, C, H, W] → IFFT2DReal
    };

    /// <summary>
    /// Returns true if the op name is classified in any category.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool IsClassified(string opName)
    {
        return DifferentiableOps.Contains(opName)
            || NonDifferentiableOps.Contains(opName)
            || DelegatorOps.Contains(opName);
    }
}
