using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Sweeps every op classified <c>DifferentiableOps</c> and checks its recorded gradient against
/// central finite differences.
/// </summary>
/// <remarks>
/// <para>
/// The existing coverage checks that ops are CLASSIFIED (TapeCompletenessTests) and that GPU
/// matches CPU (GpuCpuAutoDifferentialTests). Neither checks that a gradient claimed to exist is
/// actually CORRECT. That gap is not hypothetical: Spectrogram recorded a tape node whose backward
/// delegated to ISTFT, a synthesis operator, and returned gradients ~1/nFft off with varying sign;
/// MelSpectrogram was classified non-differentiable and returned none at all; three GPU audio
/// overrides returned results without recording. Every one of those passed the existing suites.
/// </para>
/// <para>
/// This sweep is deliberately reported rather than asserted per-op: it invokes ops reflectively,
/// so it cannot construct valid arguments for all of them. Ops it cannot drive are listed as
/// SKIPPED with the reason, so the coverage gap is visible instead of silent. Only genuine
/// disagreements fail the test.
/// </para>
/// </remarks>
public class DifferentiableOpsGradCheckSweep
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public DifferentiableOpsGradCheckSweep(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Ops excluded from the numeric check, with the reason. Anything here is a deliberate,
    /// documented exclusion — not a silent gap.
    /// </summary>
    private static readonly Dictionary<string, string> Exempt = new(StringComparer.Ordinal)
    {
        // Non-smooth at points a random probe can land on. Finite differences straddle the kink
        // and disagree with any one-sided subgradient choice.
        ["TensorAbs"] = "non-smooth at 0",
        ["TensorSign"] = "piecewise constant",
        ["TensorReLU"] = "kink at 0",
        ["TensorLeakyReLU"] = "kink at 0",
        ["TensorMaximum"] = "kink where operands tie",
        ["TensorMinimum"] = "kink where operands tie",
        ["TensorClamp"] = "kink at the clamp bounds",
        ["TensorFloor"] = "piecewise constant",
        ["TensorCeiling"] = "piecewise constant",
        ["TensorRound"] = "piecewise constant",
        ["TensorTruncate"] = "piecewise constant",
        ["TensorHardTanh"] = "kink at the saturation bounds",
        ["TensorHardSigmoid"] = "kink at the saturation bounds",
        ["TensorReLU6"] = "kink at 0 and 6",

        // Stochastic — the forward differs between the taped call and each probe call.
        ["TensorDropout"] = "stochastic forward",
        ["TensorRandomLike"] = "stochastic forward",

        // Reductions over indices: the gradient is a selection, and a probe can move which
        // index wins.
        ["TensorMax"] = "argmax can switch between probes",
        ["TensorMin"] = "argmin can switch between probes",

        // Iterative algorithms rather than single ops.
        ["GriffinLim"] = "iterative phase reconstruction, not a single op",
    };

    /// <summary>
    /// Per-op argument tables: the semantically valid shapes and couplings each op actually
    /// requires.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the real fix for the coverage gap. Reflective synthesis cannot infer that
    /// <c>TensorMatMul</c> needs the last dim of <c>a</c> to equal the first of <c>b</c>, that
    /// convolutions want NCHW plus a matching kernel, or that a spectrum's bin count is tied to a
    /// transform length — so 79 ops were skipped with "threw ArgumentException" and another 31 for
    /// parameter types no heuristic can invent. Name-based heuristics helped with scalars but
    /// cannot express shape relationships.
    /// </para>
    /// <para>
    /// Entries win over the heuristic path. Ops with neither a working entry nor a working
    /// heuristic are reported individually as NEEDS TABLE ENTRY, so the remaining gap is an
    /// explicit, shrinkable list rather than a silent skip count.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<string, Func<Random, object[]>> OpCases = new(StringComparer.Ordinal)
    {
        // --- matmul family: inner dimensions must agree ---
        ["TensorMatMul"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 2], r)],
        ["TensorMatMulTransposed"] = r => [SafeTensor([2, 3], r), SafeTensor([2, 3], r)],
        ["BatchMatMul"] = r => [SafeTensor([2, 2, 3], r), SafeTensor([2, 3, 2], r)],
        ["TensorOuterProduct"] = r => [SafeTensor([3], r), SafeTensor([4], r)],
        ["TensorOuter"] = r => [SafeTensor([3], r), SafeTensor([4], r)],
        ["TensorVecDot"] = r => [SafeTensor([4], r), SafeTensor([4], r)],
        ["TensorInner"] = r => [SafeTensor([4], r), SafeTensor([4], r)],
        ["TensorKron"] = r => [SafeTensor([2, 2], r), SafeTensor([2, 2], r)],
        ["TensorTrace"] = r => [SafeTensor([3, 3], r)],
        ["TensorCosineSimilarity"] = r => [SafeTensor([2, 4], r), SafeTensor([2, 4], r), 1, 1e-8],

        // --- elementwise binaries needing matched shapes ---
        ["TensorAddMany"] = r => [new[] { SafeTensor([4], r), SafeTensor([4], r), SafeTensor([4], r) }],
        ["TensorMultiplyMany"] = r => [new[] { SafeTensor([4], r), SafeTensor([4], r) }],
        // Ldexp's second operand is a Tensor<int> EXPONENT, not a float tensor — the previous entry
        // passed two SafeTensors and threw.
        ["TensorLdexp"] = r => [SafeTensor([4], r), IdxTensor([4], 3, r)],

        // --- detection: IoU family takes [N,4] and [M,4] boxes in (x1,y1,x2,y2) order ---
        ["BoxIou"] = r => [Boxes(2, r), Boxes(3, r)],
        ["GeneralizedBoxIou"] = r => [Boxes(2, r), Boxes(3, r)],
        ["DistanceBoxIou"] = r => [Boxes(2, r), Boxes(3, r)],
        ["CompleteBoxIou"] = r => [Boxes(2, r), Boxes(3, r)],

        // --- RoI ops: boxes are [K,5] = (batchIdx, x1, y1, x2, y2) ---
        ["RoIAlign"] = r => [SafeTensor([1, 2, 4, 4], r), RoiBoxes(2), 2, 2, 1.0f, 2, false],
        ["RoIPool"] = r => [SafeTensor([1, 2, 4, 4], r), RoiBoxes(2), 2, 2, 1.0f],
        // Position-sensitive variants require C == outH * outW * outChannels (2*2*2 = 8).
        ["PsRoIAlign"] = r => [SafeTensor([1, 8, 4, 4], r), RoiBoxes(2), 2, 2, 2, 1.0f, 2],
        ["PsRoIPool"] = r => [SafeTensor([1, 8, 4, 4], r), RoiBoxes(2), 2, 2, 2, 1.0f],

        // --- fused linear: out = act(input @ weight + bias), so weight is [in, out] ---
        ["FusedLinearGELU"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r)],
        ["FusedLinearReLU"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r)],
        ["FusedLinearSigmoid"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r)],
        ["FusedLinearSwish"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r)],
        ["FusedLinearTanh"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4], r)],
        ["FusedLinearCrossEntropyWithLogits"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r),
                                                      SafeTensor([4], r), IdxTensor([2], 4, r)],

        // --- misc ---
        ["TensorAddMM|T,T,T"] = r => [SafeTensor([2, 2], r), SafeTensor([2, 3], r), SafeTensor([3, 2], r)],
        ["TensorAddMM|T,T,T,double,double"] = r => [SafeTensor([2, 2], r), SafeTensor([2, 3], r),
                                                    SafeTensor([3, 2], r), 1.0, 1.0],
        ["RBFKernel"] = r => [SafeTensor([2, 3], r), SafeTensor([4, 3], r), SafeTensor([4], r)],
        // Octonions are 8-dimensional, so the feature axis must be a multiple of 8.
        // RoPE: [B, H, S, D] with cos/sin tables of [S, D/2] for the interleaved variant.
        ["ApplyRoPEInterleaved"] = r => [SafeTensor([1, 1, 4, 4], r), SafeTensor([4, 2], r), SafeTensor([4, 2], r), 0],
        ["PadNd"] = r => [SafeTensor([1, 1, 3, 3], r), new[] { 1, 1, 1, 1 }, PadMode.Constant, 0.0],
        ["Interpolate"] = r => [SafeTensor([1, 2, 2, 2], r), new[] { 4, 4 }, InterpolateMode.Bilinear, false],
        // Dropout at rate 0 is deterministic, so it CAN be gradient-checked (unlike TensorDropout,
        // which is exempted as stochastic). training:true keeps the real code path.
        ["Dropout"] = r => [SafeTensor([2, 3], r), 0.0, true, null!],
        ["TensorScatterReduce"] = r => [SafeTensor([3, 2], r), 0, IdxTensor([3, 2], 3, r),
                                        SafeTensor([3, 2], r), ScatterReduceMode.Sum, true],
        // CTC needs real log-probabilities and NON-BLANK targets (0 is the blank label).
        ["CTCLoss"] = r => [LogProbs(4, 1, 3, r), CtcTargets(2, 3, r), new[] { 4 }, new[] { 2 }, 0],
        ["TensorCTCLoss"] = r => [LogProbs(4, 1, 3, r), CtcTargets(2, 3, r), new[] { 4 }, new[] { 2 }, 0],

        // --- normalization: last-dim normalized shapes ---
        // These carry `out` parameters, which still need an args slot for Invoke: LayerNorm's
        // signature is (input, gamma, beta, epsilon, out mean, out variance) = 6, and RMSNorm's is
        // (input, gamma, epsilon, out rms) = 4. The previous 4- and 3-argument entries threw
        // TargetParameterCountException, so both ops were silently unchecked.
        ["LayerNorm"] = r => [SafeTensor([2, 4], r), SafeTensor([4], r), SafeTensor([4], r), 1e-5, null!, null!],
        ["RMSNorm"] = r => [SafeTensor([2, 4], r), SafeTensor([4], r), 1e-6, null!],

        // --- normalization over NCHW, all with trailing out-params ---
        ["BatchNorm"] = r => [SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r), SafeTensor([3], r), 1e-5, null!, null!],
        ["InstanceNorm"] = r => [SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r), SafeTensor([3], r), 1e-5, null!, null!],
        ["GroupNorm"] = r => [SafeTensor([2, 4, 2, 2], r), 2, SafeTensor([4], r), SafeTensor([4], r), 1e-5, null!, null!],
        // BatchNormAffine takes mean/variance as INPUTS (no out-params). SafeTensor is already
        // strictly positive ([0.35, 0.95]), so it is a valid variance.
        ["BatchNormAffine"] = r => [SafeTensor([2, 3, 2, 2], r), SafeTensor([3], r), SafeTensor([3], r),
                                    SafeTensor([3], r), SafeTensor([3], r), 1e-5],

        // --- convolutions: NCHW/NCL/NCDHW input with a matching OIHW-style kernel ---
        // Conv2D/Conv3D/AvgPool2D/AvgPool3D each have two same-arity overloads differing only in
        // int vs int[], so these are keyed by parameter fingerprint.
        ["Conv1D|T,T,int,int,int"] = r => [SafeTensor([1, 2, 6], r), SafeTensor([3, 2, 3], r), 1, 0, 1],
        ["Conv2D|T,T,int,int,int"] = r => [SafeTensor([1, 2, 5, 5], r), SafeTensor([3, 2, 3, 3], r), 1, 0, 1],
        ["Conv2D|T,T,int[],int[],int[]"] = r => [SafeTensor([1, 2, 5, 5], r), SafeTensor([3, 2, 3, 3], r),
                                                 new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }],
        ["Conv3D|T,T,int,int,int"] = r => [SafeTensor([1, 2, 4, 4, 4], r), SafeTensor([2, 2, 2, 2, 2], r), 1, 0, 1],
        ["Conv3D|T,T,int[],int[],int[]"] = r => [SafeTensor([1, 2, 4, 4, 4], r), SafeTensor([2, 2, 2, 2, 2], r),
                                                 new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }],
        ["ConvTranspose2D"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([2, 3, 3, 3], r),
                                    new[] { 1, 1 }, new[] { 0, 0 }, new[] { 0, 0 }],
        ["ConvTranspose3D"] = r => [SafeTensor([1, 2, 3, 3, 3], r), SafeTensor([2, 2, 2, 2, 2], r),
                                    new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 }],
        // Depthwise: one kernel per input channel (groups == channels).
        ["DepthwiseConv1D"] = r => [SafeTensor([1, 3, 6], r), SafeTensor([3, 1, 3], r), 1, 0],
        ["DepthwiseConv2D"] = r => [SafeTensor([1, 3, 5, 5], r), SafeTensor([3, 1, 3, 3], r),
                                    new[] { 1, 1 }, new[] { 0, 0 }],

        // --- pooling ---
        ["MaxPool2D|T,int,int,int"] = r => [SafeTensor([1, 2, 4, 4], r), 2, 2, 0],
        ["AvgPool2D|T,int,int,int"] = r => [SafeTensor([1, 2, 4, 4], r), 2, 2, 0],
        ["AvgPool2D|T,int[],int[]"] = r => [SafeTensor([1, 2, 4, 4], r), new[] { 2, 2 }, new[] { 2, 2 }],
        ["AvgPool3D|T,int,int,int"] = r => [SafeTensor([1, 2, 4, 4, 4], r), 2, 2, 0],
        ["AvgPool3D|T,int[],int[],int[]"] = r => [SafeTensor([1, 2, 4, 4, 4], r),
                                                  new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }],
        ["AdaptiveAvgPool2D"] = r => [SafeTensor([1, 2, 4, 4], r), 2, 2],

        // --- resampling / spatial rearrangement ---
        ["PixelShuffle"] = r => [SafeTensor([1, 4, 2, 2], r), 2],   // C must be divisible by r^2
        ["Upsample3D"] = r => [SafeTensor([1, 2, 2, 2, 2], r), 2, 2, 2],
        ["TensorUpsampleBilinear"] = r => [SafeTensor([1, 2, 2, 2], r), new[] { 4, 4 }],
        ["Crop"] = r => [SafeTensor([1, 2, 4, 4], r), 1, 1, 2, 2],
        ["Unfold"] = r => [SafeTensor([1, 2, 4, 4], r), new[] { 2, 2 }, new[] { 1, 1 }, new[] { 0, 0 }],
        // Fold is Unfold's inverse: input is the [N, C*prod(kernel), L] column matrix.
        ["Fold"] = r => [SafeTensor([1, 2 * 2 * 2, 9], r), new[] { 4, 4 }, new[] { 2, 2 },
                         new[] { 1, 1 }, new[] { 0, 0 }],

        // --- sampling grids: grid is [N, H, W, 2] for 2-D, theta is [N, 2, 3] ---
        ["GridSample|T,T"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([1, 3, 3, 2], r)],
        ["AffineGrid"] = r => [SafeTensor([1, 2, 3], r), 3, 3],
        ["AffineGrid3D"] = r => [SafeTensor([1, 3, 4], r), 2, 2, 2, false],

        // --- shape ops: the target shape must be consistent with the input ---
        ["Reshape"] = r => [SafeTensor([2, 3], r), new[] { 3, 2 }],
        ["TensorSqueeze"] = r => [SafeTensor([1, 4], r), 0],
        ["TensorTile"] = r => [SafeTensor([2, 2], r), new[] { 2, 1 }],
        ["TensorConcatenate"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["Concat"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["TensorStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }, 0],
        ["TensorHStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorVStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorColumnStack"] = r => [new[] { SafeTensor([3], r), SafeTensor([3], r) }],
        ["TensorRowStack"] = r => [new[] { SafeTensor([3], r), SafeTensor([3], r) }],
        ["TensorRot90"] = r => [SafeTensor([3, 3], r), 1, new[] { 0, 1 }],
        ["TensorSlice"] = r => [SafeTensor([4, 4], r), new[] { 0, 0 }, new[] { 2, 2 }],
        ["TensorBlockDiag"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        ["TensorCartesianProd"] = r => [new[] { SafeTensor([2], r), SafeTensor([2], r) }],

        // --- spectral: bin count is tied to the transform length ---
        ["IRFFT"] = r => [SafeTensor([2 * (8 / 2 + 1)], r), 8],
        ["Spectrogram"] = r => [SafeTensor([64], r), 16, 4, 16, HannWindowFor(16)],

        // --- reductions with explicit axes. ReduceMax is overloaded 3-param / 4-param
        //     (the latter with `out int[] maxIndices`), hence the arity-keyed pair. ---
        ["ReduceMax/3"] = r => [SafeTensor([2, 3], r), new[] { 1 }, false],
        ["ReduceMax/4"] = r => [SafeTensor([2, 3], r), new[] { 1 }, false, null!],
        ["ReduceMaxWithTensorIndices"] = r => [SafeTensor([2, 3], r), new[] { 1 }, false, null!],

        // --- gather / scatter / index: the index tensor must stay in range for its axis ---
        ["Gather"] = r => [SafeTensor([2, 3], r), IdxTensor([2, 2], 3, r), 1],
        // Scatter / ScatterAdd's index+value form: `indices` is 1-D along the scatter axis (length
        // <= that axis) and `values` is the input shape with the axis replaced by indices.Length —
        // the loop walks `idx < indices.Length` against innerSize, so a full-size index grid overruns
        // the values array.
        ["Scatter"] = r => [SafeTensor([2, 3], r), IdxTensor([2], 3, r), SafeTensor([2, 2], r), 1],
        ["ScatterAdd|T,Ti,T,int"] = r => [SafeTensor([2, 3], r), IdxTensor([2], 3, r), SafeTensor([2, 2], r), 1],
        // Segment-style scatter family: source rows are grouped by indices along dim.
        ["ScatterAdd|T,Ti,int,int?"] = r => [SafeTensor([4, 2], r), IdxTensor([4], 3, r), 0, 3],
        ["ScatterSoftmax"] = r => [SafeTensor([4, 2], r), IdxTensor([4], 3, r), 0, 3],
        ["ScatterMax"] = r => [SafeTensor([4, 2], r), IdxTensor([4], 3, r), null!, 0, 3],
        ["ScatterMean"] = r => [SafeTensor([4, 2], r), IdxTensor([4], 3, r), null!, 0, 3],
        ["TensorIndexAdd"] = r => [SafeTensor([3, 2], r), 0, IdxRange(3), SafeTensor([3, 2], r)],
        ["TensorIndexCopy"] = r => [SafeTensor([3, 2], r), 0, IdxRange(3), SafeTensor([3, 2], r)],
        ["TensorIndexFill"] = r => [SafeTensor([3, 2], r), 0, IdxRange(2), 0.5],
        ["TensorIndexSelect"] = r => [SafeTensor([3, 2], r), IdxRange(2), 0],
        ["TensorTake"] = r => [SafeTensor([2, 3], r), IdxRange(4)],
        ["TensorTakeAlongDim"] = r => [SafeTensor([2, 3], r), IdxTensor([2, 3], 3, r), 1],
        ["TensorPut"] = r => [SafeTensor([2, 3], r), IdxRange(3), SafeTensor([3], r)],
        ["TensorSelectScatter"] = r => [SafeTensor([3, 2], r), SafeTensor([2], r), 0, 1],
        ["TensorSliceScatter"] = r => [SafeTensor([4, 2], r), SafeTensor([2, 2], r), 0, 1, 2],
        ["Embedding"] = r => [IdxTensor([2, 3], 5, r), SafeTensor([5, 4], r)],

        // --- masked ops. Three TensorMaskedFill overloads share arity 3 and differ only in the
        //     mask type, so all three are keyed by fingerprint. ---
        ["TensorMaskedFill|T,Tb,double"] = r => [SafeTensor([2, 3], r), BoolMask([2, 3]), 0.25],
        ["TensorMaskedFill|T,TBit,double"] = r => [SafeTensor([2, 3], r), BitMask([2, 3]), 0.25],
        ["TensorMaskedFill|T,bool[],double"] = r => [SafeTensor([2, 3], r), AltBools(6), 0.25],

        // --- TensorWhere: the MASK is the leading parameter, so all four overloads differ in slot 0 ---
        ["TensorWhere|Tb,T,T"] = r => [BoolMask([2, 3]), SafeTensor([2, 3], r), SafeTensor([2, 3], r)],
        ["TensorWhere|TBit,T,T"] = r => [BitMask([2, 3]), SafeTensor([2, 3], r), SafeTensor([2, 3], r)],
        ["TensorWhere|bool[],T,T"] = r => [AltBools(6), SafeTensor([2, 3], r), SafeTensor([2, 3], r)],

        // --- pooling variants returning indices. The index buffer is an `out` parameter, so it only
        //     needs a slot; MaxPool2DWithIndices' is a 5-DIMENSIONAL int array that no heuristic
        //     could ever synthesize. ---
        ["MaxPool2DWithIndices"] = r => [SafeTensor([1, 2, 4, 4], r), new[] { 2, 2 }, new[] { 2, 2 }, null!],
        ["MaxPool2DWithTensorIndices"] = r => [SafeTensor([1, 2, 4, 4], r), new[] { 2, 2 }, new[] { 2, 2 }, null!],
        ["MaxPool3DWithIndices"] = r => [SafeTensor([1, 2, 4, 4, 4], r), new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, null!],
        ["MaxPool3DWithTensorIndices"] = r => [SafeTensor([1, 2, 4, 4, 4], r), new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, null!],

        // --- GridSample's mode/padding overload ---
        ["GridSample|T,T,GridSampleMode,GridSamplePadding,bool"] = r =>
            [SafeTensor([1, 2, 4, 4], r), SafeTensor([1, 3, 3, 2], r),
             GridSampleMode.Bilinear, GridSamplePadding.Zeros, false],

        // --- fused linear cross-entropy: one overload takes class INDICES, the other a float
        //     target distribution over classes ---
        ["FusedLinearCrossEntropyWithLogits|T,T,T,Ti"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r),
                                                               SafeTensor([4], r), IdxTensor([2], 4, r)],
        ["FusedLinearCrossEntropyWithLogits|T,T,T,T"] = r => [SafeTensor([2, 3], r), SafeTensor([3, 4], r),
                                                              SafeTensor([4], r), RowNormalized([2, 4], r)],

        // --- ops whose shapes are dictated by their own validation ---
        // Octonions: input [B, F, 8] and weight [O, F, 8] — rank-3 with last dim 8 and matching F.
        ["OctonionMatMulTensor"] = r => [SafeTensor([1, 2, 8], r), SafeTensor([3, 2, 8], r)],
        // Locally-connected: rank-6 weights lead with the OUTPUT SPATIAL dims —
        // [outH, outW, outC, inC, kh, kw] — which the op's own error message confirms
        // ("Calculated output dimensions (2x2) do not match weights dimensions (3x2)"). A 4x4 input
        // with a 3x3 kernel at stride 1 gives outH = outW = 2.
        ["LocallyConnectedConv2D"] = r => [SafeTensor([1, 2, 4, 4], r), SafeTensor([2, 2, 3, 2, 3, 3], r),
                                           null!, new[] { 1, 1 }],
        // Trilinear: grid is [D, H, W, C] and positions are [N, 3].
        ["TensorTrilinearInterpolate"] = r => [SafeTensor([2, 2, 2, 1], r), TrilinearPositions()],
        // IndexPut takes ONE index tensor PER DIMENSION (indices.Length == rank), so a rank-2 target
        // needs two, and they jointly address source.Length points.
        ["TensorIndexPut"] = r => [SafeTensor([3, 2], r), new[] { IdxRange(2), IdxRange(2) },
                                   SafeTensor([2], r), false],

        // --- fused sequence-scan kernels: rank-3 [batch, seqLen, modelDim] with per-head gates
        //     shaped [batch, seqLen, numHeads]; modelDim must divide by numHeads ---
        ["Rwkv4WkvForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                    SafeTensor([4], r), SafeTensor([4], r)],
        // RWKV-7's decayLogit and iclRate are per-CHANNEL ([B, L, modelDim]), not per-head — unlike
        // the GLA / GatedDeltaNet / xLSTM gates, which are [B, L, numHeads].
        ["Rwkv7SequenceForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                         SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), 2],
        ["GlaScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                   SafeTensor([1, 3, 2], r), 2],
        ["GatedDeltaNetScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                             SafeTensor([1, 3, 2], r), SafeTensor([1, 3, 2], r), 2],
        ["XLstmScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                     SafeTensor([1, 3, 2], r), SafeTensor([1, 3, 2], r), SafeTensor([1, 3, 2], r), 2],
        // RgLru: [batch, seqLen, recDim] with a per-recDim decay vector.
        ["RgLruScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                     SafeTensor([4], r)],
        // Mamba S6: x/delta [B, L, innerDim]; aLog [innerDim, stateDim]; b/c [B, L, stateDim];
        // d [innerDim].
        ["MambaSelectiveScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([4, 2], r),
                                              SafeTensor([1, 3, 2], r), SafeTensor([1, 3, 2], r), SafeTensor([4], r)],
        ["ComplexDiagonalSsmScanForward"] = r => [SafeTensor([1, 3, 2, 2], r), SafeTensor([2, 3], r), SafeTensor([2, 3], r),
                                                   SafeTensor([2, 3, 2], r), SafeTensor([2, 3, 2], r),
                                                   SafeTensor([2, 2, 3], r), SafeTensor([2, 2, 3], r), SafeTensor([2, 2], r)],
        ["MesaScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 4], r),
                                     SafeTensor([2, 2, 2], r), 0.7, 2],
        ["RoutedDiagonalSsmScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 2], r),
                                                  SafeTensor([2, 3], r), SafeTensor([2, 3, 4], r),
                                                  SafeTensor([2, 4, 3], r), SafeTensor([2, 4], r)],
        // Mamba-2 SSD is per-HEAD where Mamba S6 is per-channel: delta is [B, L, numHeads], and aLog
        // and dParam are per-head SCALAR vectors of length numHeads rather than S6's
        // [innerDim, stateDim] matrix and [innerDim] vector. That is the "scalar-decay"
        // simplification SSD makes to obtain its matmul-based form.
        ["Mamba2SsdScanForward"] = r => [SafeTensor([1, 3, 4], r), SafeTensor([1, 3, 2], r), SafeTensor([2], r),
                                         SafeTensor([1, 3, 2], r), SafeTensor([1, 3, 2], r), SafeTensor([2], r), 2],
        ["TensorMaskedScatter"] = r => [SafeTensor([2, 3], r), BitMask([2, 3]), SafeTensor([2, 3], r)],
        ["TensorMaskedSelect"] = r => [SafeTensor([2, 3], r), BitMask([2, 3])],

        // --- variadic shape ops needing consistent member shapes ---
        ["TensorDStack"] = r => [new[] { SafeTensor([2, 2], r), SafeTensor([2, 2], r) }],
        // MultiDot chains matmuls, so adjacent inner dimensions must agree: (2x3)(3x4)(4x2).
        ["TensorMultiDot"] = r => [new[] { SafeTensor([2, 3], r), SafeTensor([3, 4], r), SafeTensor([4, 2], r) }],
    };

    /// <summary>
    /// Compact parameter-type fingerprint used to key table entries to a SPECIFIC overload,
    /// e.g. <c>T,T,int[],int[],int[]</c>. <c>T</c> denotes <c>Tensor&lt;double&gt;</c>.
    /// </summary>
    private static string ParamFingerprint(MethodInfo m)
        => string.Join(",", m.GetParameters().Select(p => TypeToken(p.ParameterType)));

    /// <summary>
    /// Stable short token per parameter type. Generic tensors MUST be distinguished by their element
    /// type — <c>Tensor&lt;bool&gt;</c> and <c>Tensor&lt;Bit&gt;</c> both render as "Tensor`1" under
    /// <c>Type.Name</c>, which would collide and make the three TensorMaskedFill overloads
    /// indistinguishable.
    /// </summary>
    private static string TypeToken(Type t)
    {
        if (t.IsByRef) t = t.GetElementType()!;
        if (t.IsArray && t.GetElementType() is { } el && el != typeof(int) && el != typeof(bool))
            return TypeToken(el) + "[]";
        if (t == typeof(Tensor<double>)) return "T";
        if (t == typeof(Tensor<int>)) return "Ti";
        if (t == typeof(Tensor<bool>)) return "Tb";
        if (t == typeof(Tensor<Bit>)) return "TBit";
        if (t == typeof(int)) return "int";
        if (t == typeof(int[])) return "int[]";
        if (t == typeof(bool)) return "bool";
        if (t == typeof(bool[])) return "bool[]";
        if (t == typeof(double)) return "double";
        if (t == typeof(int?)) return "int?";
        if (t == typeof(double?)) return "double?";
        if (t.IsGenericType) return t.Name + "<" + string.Join(",", t.GetGenericArguments().Select(TypeToken)) + ">";
        return t.Name;
    }

    /// <summary>Hann window of exactly nFft samples, matching CpuEngine's own definition.</summary>
    private static Tensor<double> HannWindowFor(int nFft)
    {
        var w = new Tensor<double>([nFft]);
        for (int i = 0; i < nFft; i++)
            w[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Math.Max(1, nFft - 1));
        return w;
    }

    private static Tensor<double> SafeTensor(int[] shape, Random rng)
    {
        // Values in [0.35, 0.95]: away from 0 (log/sqrt/reciprocal domains, |.| kink) and away
        // from 1 (acos/atanh edges), all strictly positive so domain-restricted ops are valid.
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = 0.35 + rng.NextDouble() * 0.6;
        return t;
    }

    /// <summary>
    /// Boxes as <c>[n, 4]</c> in (x1, y1, x2, y2) order with x2 &gt; x1 and y2 &gt; y1.
    /// </summary>
    /// <remarks>
    /// SafeTensor's uniform [0.35, 0.95] values cannot be used for boxes: they would frequently give
    /// x2 &lt; x1, producing degenerate/negative areas where the IoU formulae are not differentiable
    /// (and often not even meaningful). Boxes are spread apart so intersections are non-degenerate and
    /// finite differences do not straddle a kink.
    /// </remarks>
    private static Tensor<double> Boxes(int n, Random rng, double spread = 3.0)
    {
        var t = new Tensor<double>([n, 4]);
        for (int i = 0; i < n; i++)
        {
            double x1 = i * spread + 0.25 + rng.NextDouble() * 0.2;
            double y1 = i * spread + 0.35 + rng.NextDouble() * 0.2;
            t[i * 4 + 0] = x1;
            t[i * 4 + 1] = y1;
            t[i * 4 + 2] = x1 + 1.5 + rng.NextDouble() * 0.3;   // strictly > x1
            t[i * 4 + 3] = y1 + 1.6 + rng.NextDouble() * 0.3;   // strictly > y1
        }
        return t;
    }

    /// <summary>
    /// Genuine log-probabilities of shape <c>[T, N, C]</c>: log-softmax over the class axis, so every
    /// value is negative and each timestep's classes sum to 1 in probability space.
    /// </summary>
    /// <remarks>
    /// CTC consumes LOG-probabilities. Feeding it SafeTensor's positive [0.35, 0.95] values puts it
    /// outside its domain entirely, which is not a fair test of its gradient.
    /// </remarks>
    private static Tensor<double> LogProbs(int timeSteps, int batch, int classes, Random rng)
    {
        var t = new Tensor<double>([timeSteps, batch, classes]);
        for (int i = 0; i < timeSteps * batch; i++)
        {
            var logits = new double[classes];
            double max = double.NegativeInfinity;
            for (int c = 0; c < classes; c++) { logits[c] = -1.0 + rng.NextDouble() * 2.0; max = Math.Max(max, logits[c]); }
            double sumExp = 0;
            for (int c = 0; c < classes; c++) sumExp += Math.Exp(logits[c] - max);
            double lse = max + Math.Log(sumExp);
            for (int c = 0; c < classes; c++) t[i * classes + c] = logits[c] - lse;
        }
        return t;
    }

    /// <summary>
    /// CTC target labels in <c>[1, classes)</c> — index 0 is the BLANK and is not a legal target.
    /// </summary>
    private static Tensor<int> CtcTargets(int length, int classes, Random rng)
    {
        var t = new Tensor<int>([length]);
        for (int i = 0; i < length; i++) t[i] = 1 + rng.Next(classes - 1);
        return t;
    }

    /// <summary>
    /// Sample positions strictly INSIDE a 2x2x2 grid, so trilinear interpolation stays in its smooth
    /// interior rather than clamping at a boundary (where the gradient is one-sided).
    /// </summary>
    private static Tensor<double> TrilinearPositions()
    {
        var t = new Tensor<double>([1, 3]);
        t[0] = 0.4; t[1] = 0.5; t[2] = 0.6;
        return t;
    }

    /// <summary>RoI boxes as <c>[k, 5]</c> = (batchIndex, x1, y1, x2, y2), torchvision's layout.</summary>
    private static Tensor<double> RoiBoxes(int k)
    {
        var t = new Tensor<double>([k, 5]);
        for (int i = 0; i < k; i++)
        {
            t[i * 5 + 0] = 0;      // single-image batch
            t[i * 5 + 1] = 0.5;
            t[i * 5 + 2] = 0.5;
            t[i * 5 + 3] = 2.5;
            t[i * 5 + 4] = 2.5;
        }
        return t;
    }

    /// <summary>Index tensor with every value in [0, maxExclusive) — valid for gather/scatter axes.</summary>
    private static Tensor<int> IdxTensor(int[] shape, int maxExclusive, Random rng)
    {
        var t = new Tensor<int>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.Next(maxExclusive);
        return t;
    }

    /// <summary>Index tensor covering 0..n-1 exactly once, for ops requiring a permutation-like map.</summary>
    private static Tensor<int> IdxRange(int n)
    {
        var t = new Tensor<int>([n]);
        for (int i = 0; i < n; i++) t[i] = i;
        return t;
    }

    /// <summary>Alternating Bit mask — guarantees both branches are exercised and is deterministic.</summary>
    private static Tensor<Bit> BitMask(int[] shape)
    {
        var t = new Tensor<Bit>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i % 2 == 0;   // implicit bool -> Bit
        return t;
    }

    /// <summary>Alternating raw bool[] mask, for the overloads taking a plain array.</summary>
    private static bool[] AltBools(int n)
    {
        var m = new bool[n];
        for (int i = 0; i < n; i++) m[i] = i % 2 == 0;
        return m;
    }

    /// <summary>
    /// Rows that sum to 1 — a valid target DISTRIBUTION for the soft-label cross-entropy overload.
    /// </summary>
    private static Tensor<double> RowNormalized(int[] shape, Random rng)
    {
        var t = new Tensor<double>(shape);
        int cols = shape[shape.Length - 1];
        int rows = t.Length / cols;
        for (int r0 = 0; r0 < rows; r0++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++) { t[r0 * cols + c] = 0.1 + rng.NextDouble(); sum += t[r0 * cols + c]; }
            for (int c = 0; c < cols; c++) t[r0 * cols + c] /= sum;
        }
        return t;
    }

    /// <summary>Alternating bool mask, for the Tensor&lt;bool&gt; overloads.</summary>
    private static Tensor<bool> BoolMask(int[] shape)
    {
        var t = new Tensor<bool>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i % 2 == 0;
        return t;
    }

    private static bool IsTensorDouble(Type t) =>
        t == typeof(Tensor<double>) ||
        (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Tensor<>));

    /// <summary>
    /// Best-effort argument construction. Returns null when a parameter type is not something
    /// this harness can synthesize, which is reported as a skip.
    /// </summary>
    private static object[]? BuildArgs(MethodInfo m, int[] shape, Random rng, out int firstTensorIdx)
    {
        firstTensorIdx = -1;
        var ps = m.GetParameters();
        var args = new object[ps.Length];

        for (int i = 0; i < ps.Length; i++)
        {
            var pt = ps[i].ParameterType;

            if (pt.IsByRef) return null;                     // out/ref ops are covered elsewhere

            if (IsTensorDouble(pt))
            {
                args[i] = SafeTensor(shape, rng);
                if (firstTensorIdx < 0) firstTensorIdx = i;
                continue;
            }

            // Name-aware synthesis. Blanket values are actively harmful for semantically loaded
            // parameters: 0.5 for an `epsilon` is not a stabilizer but a large additive constant
            // (it produced a bogus ReduceLogVariance mismatch and collapsed BinaryCrossEntropy's
            // [eps, 1-eps] clamp to a point, making its forward constant), and 1 for an
            // `outputLength` desynchronizes a transform from its own spectrum.
            var pname = (ps[i].Name ?? string.Empty).ToLowerInvariant();
            bool IsEpsilon() => pname.Contains("epsilon") || pname == "eps" || pname.Contains("tolerance");
            bool IsLength() => pname.Contains("outputlength") || pname == "n" || pname == "nfft" || pname.Contains("signallength");
            bool IsAxis() => pname is "axis" or "dim" or "dimension";

            if (pt == typeof(double)) { args[i] = IsEpsilon() ? 1e-7 : 0.5; continue; }
            if (pt == typeof(float)) { args[i] = IsEpsilon() ? 1e-7f : 0.5f; continue; }
            if (pt == typeof(bool)) { args[i] = ps[i].HasDefaultValue && ps[i].DefaultValue is bool b ? b : false; continue; }
            if (pt == typeof(int))
            {
                if (IsLength()) { args[i] = shape[^1]; continue; }
                if (IsAxis()) { args[i] = shape.Length - 1; continue; }
                args[i] = ps[i].HasDefaultValue && ps[i].DefaultValue is int d ? d : 1;
                continue;
            }
            if (pt == typeof(int?) || pt == typeof(double?) || pt == typeof(float?))
            {
                var inner = Nullable.GetUnderlyingType(pt)!;
                if (IsEpsilon() && inner == typeof(double)) { args[i] = 1e-7; continue; }
                if (IsLength() && inner == typeof(int)) { args[i] = shape[^1]; continue; }
                args[i] = ps[i].HasDefaultValue ? ps[i].DefaultValue! : null!;
                continue;
            }
            if (pt == typeof(int[]))
            {
                // axes/dims default to the last axis, which is valid for any rank.
                args[i] = new[] { shape.Length - 1 };
                continue;
            }

            // A generic-T scalar (e.g. TensorBinaryCrossEntropy's `T epsilon`) — bind by name too.
            if (pt == typeof(double) || pt.IsGenericParameter)
            {
                args[i] = IsEpsilon() ? 1e-7 : 0.5;
                continue;
            }

            if (ps[i].HasDefaultValue) { args[i] = ps[i].DefaultValue!; continue; }

            return null;                                      // unsupported parameter type
        }

        return firstTensorIdx >= 0 ? args : null;
    }

    [Fact]
    public void EveryDifferentiableOp_GradientMatchesFiniteDifferences()
    {
        var engineType = typeof(IEngine);
        var candidates = engineType.GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => !m.IsSpecialName && m.IsGenericMethodDefinition)
            .Where(m => m.GetGenericArguments().Length == 1)
            .Where(m => IsTensorDouble(m.ReturnType))
            .ToList();

        var shapes = new[] { new[] { 6 }, new[] { 2, 3 } };
        var mismatches = new List<string>();
        var noGradient = new List<string>();
        var skipped = new List<string>();
        var checkedOk = new List<string>();
        var exempted = new List<string>();

        foreach (var def in candidates)
        {
            var name = def.Name;
            if (name.Contains('`')) name = name.Substring(0, name.IndexOf('`'));

            if (!OpRegistry.DifferentiableOps.Contains(name)) continue;
            if (Exempt.TryGetValue(name, out var why)) { exempted.Add($"{name} ({why})"); continue; }

            MethodInfo m;
            try { m = def.MakeGenericMethod(typeof(double)); }
            catch (Exception ex) { skipped.Add($"{name}: cannot bind <double> ({ex.GetType().Name})"); continue; }

            bool handled = false;
            string lastSkip = "no shape produced a valid invocation";

            // The per-op table wins over reflective synthesis: it is the only way to express shape
            // relationships (matmul inner dims, spectrum/transform-length coupling, NCHW layouts).
            // Overload-aware lookup, most specific first:
            //   1. "Conv2D|T,T,int[],int[],int[]"  — exact parameter-type fingerprint
            //   2. "ReduceMax/4"                   — arity
            //   3. "TensorMatMul"                  — bare name
            //
            // A name-only key cannot serve an overloaded op. ReduceMax has a 3-param form and a
            // 4-param form taking `out int[] maxIndices`; LayerNorm/BatchNorm/GroupNorm/InstanceNorm
            // carry `out` params that still need an Invoke slot each (their entries were short and
            // died with TargetParameterCountException, so those ops went silently unchecked); and
            // Conv2D, Conv3D and AvgPool2D/3D each have TWO overloads of the SAME arity that differ
            // only in int-vs-int[] parameters, which arity cannot distinguish.
            string fingerprint = $"{name}|{ParamFingerprint(m)}";
            bool hasTable = OpCases.TryGetValue(fingerprint, out var caseFactory)
                         || OpCases.TryGetValue($"{name}/{m.GetParameters().Length}", out caseFactory)
                         || OpCases.TryGetValue(name, out caseFactory);
            var shapesToTry = hasTable ? new[] { Array.Empty<int>() } : shapes;

            foreach (var shape in shapesToTry)
            {
                var rng = new Random(1234);
                object[]? args;
                if (hasTable)
                {
                    try { args = caseFactory!(rng); }
                    catch (Exception ex) { lastSkip = $"table entry threw {ex.GetType().Name}"; continue; }
                }
                else
                {
                    args = BuildArgs(m, shape, rng, out _);
                }
                if (args is null) { lastSkip = "unsupported parameter types (NEEDS TABLE ENTRY)"; continue; }

                // Sanity: does it even run untaped on this shape?
                try { _ = m.Invoke(_engine, CopyArgs(args)); }
                // Include the exception MESSAGE, not just its type: these ops validate their own shape
                // requirements and say precisely what they wanted, which is what a table entry needs.
                catch (Exception ex)
                {
                    lastSkip = $"{shape.Length}D threw {Inner(ex).GetType().Name}: {Inner(ex).Message}";
                    continue;
                }

                // Check EVERY tensor parameter, not just the first. Taking the first is wrong for
                // ops whose leading tensor is not a differentiable input — TensorWhere's leading
                // argument is the condition mask, which correctly receives no gradient. Only flag
                // an op when NO tensor input receives one.
                // FLATTEN Tensor<double>[] parameters. OfType<Tensor<double>>() sees only
                // directly-typed arguments, so every variadic op (TensorStack, Concat,
                // TensorHStack/VStack, TensorBlockDiag, TensorAddMany, …) reported
                // "no gradient for ANY of its 0 tensor input(s)" — a harness blind spot that read
                // exactly like a real missing backward and hid whether one existed. That is a false
                // accusation against a working op, not a caught regression.
                //
                // The IEnumerable case is deliberate and covers more than Tensor<double>[]: an op
                // declaring IReadOnlyList<Tensor<double>> would otherwise slip back into the blind
                // spot the array case was added to close.
                var tensorInputs = args.SelectMany(a => a switch
                {
                    Tensor<double> t => new[] { t },
                    Tensor<double>[] arr => arr,
                    IEnumerable<Tensor<double>> seq => seq.ToArray(),
                    _ => Array.Empty<Tensor<double>>(),
                }).ToArray();
                Tensor<double> input;
                Tensor<double> analytical;
                try
                {
                    using var tape = new GradientTape<double>();
                    var outT = (Tensor<double>)m.Invoke(_engine, CopyArgs(args))!;
                    var loss = _engine.ReduceSum(outT, null);
                    var grads = tape.ComputeGradients(loss, tensorInputs);

                    var got = tensorInputs.FirstOrDefault(t => grads.TryGetValue(t, out var gg) && gg is not null);
                    if (got is null)
                    {
                        noGradient.Add($"{name}: no gradient for ANY of its {tensorInputs.Length} tensor input(s)");
                        handled = true;
                        break;
                    }
                    input = got;
                    analytical = grads[got];
                }
                catch (Exception ex)
                {
                    lastSkip = $"backward threw {Inner(ex).GetType().Name}: {Inner(ex).Message}";
                    continue;
                }

                if (analytical.Length != input.Length)
                {
                    mismatches.Add($"{name}: gradient shape [{string.Join(",", analytical.Shape.ToArray())}] " +
                                   $"does not match input [{string.Join(",", input.Shape.ToArray())}] " +
                                   $"| args: {DescribeArgs(m, args)}");
                    handled = true;
                    break;
                }

                // Central finite differences on a few elements.
                const double eps = 1e-6;
                var bad = new List<string>();
                int probes = Math.Min(4, input.Length);
                for (int k = 0; k < probes; k++)
                {
                    double orig = input[k];
                    double lp, lm;
                    try
                    {
                        input[k] = orig + eps;
                        lp = _engine.TensorSum((Tensor<double>)m.Invoke(_engine, CopyArgs(args))!);
                        input[k] = orig - eps;
                        lm = _engine.TensorSum((Tensor<double>)m.Invoke(_engine, CopyArgs(args))!);
                    }
                    finally { input[k] = orig; }

                    double numerical = (lp - lm) / (2 * eps);
                    double a = analytical[k];
                    double denom = Math.Max(1.0, Math.Max(Math.Abs(a), Math.Abs(numerical)));
                    if (Math.Abs(a - numerical) / denom > 1e-4)
                        bad.Add($"[{k}] analytical {a:G6} vs numerical {numerical:G6}");
                }

                if (bad.Count > 0)
                    mismatches.Add($"{name} on {shape.Length}D: " + string.Join("; ", bad.Take(3)));
                else
                    checkedOk.Add(name);

                handled = true;
                break;
            }

            // Include the fingerprint so a table entry can be keyed to this EXACT overload without
            // guessing at the token spelling.
            if (!handled) skipped.Add($"{name}|{ParamFingerprint(m)}: {lastSkip}");
        }

        _out.WriteLine($"gradient-checked OK : {checkedOk.Count}");
        _out.WriteLine($"MISMATCH            : {mismatches.Count}");
        _out.WriteLine($"NO GRADIENT         : {noGradient.Count}");
        _out.WriteLine($"exempt (documented) : {exempted.Count}");
        _out.WriteLine($"skipped (harness)   : {skipped.Count}");
        _out.WriteLine("");
        foreach (var s in mismatches) _out.WriteLine("MISMATCH   " + s);
        foreach (var s in noGradient) _out.WriteLine("NO-GRAD    " + s);
        _out.WriteLine("");
        foreach (var s in skipped.OrderBy(x => x)) _out.WriteLine("skip  " + s);

        // A SKIP is an op the harness could not drive at all, so its gradient is simply UNVERIFIED.
        // That is materially different from a pass, and it is invisible in a green build unless it
        // is asserted on: this suite went from 114 skips to 0 by adding per-op argument tables, and
        // without a ratchet the next op whose parameters the synthesizer cannot handle would slip
        // back in silently and CI would stay green while coverage rotted.
        //
        // Measured 0 skips on BOTH target frameworks (net10.0 and net471, 251 ops verified on each),
        // so 0 is the honest floor rather than a number chosen to make the assertion pass.
        //
        // An op that genuinely cannot be gradient-checked has a supported escape hatch — add it to
        // the Exempt table above with the reason, which keeps it visible as a deliberate exclusion
        // instead of an accident. So the ratchet does not block legitimate work; it forces a new op
        // to be either COVERED or DOCUMENTED, never silently unchecked.
        if (mismatches.Count > 0 || noGradient.Count > 0 || skipped.Count > 0)
        {
            var problems = new List<string>();
            if (mismatches.Count > 0)
                problems.Add($"{mismatches.Count} op(s) disagree with finite differences");
            if (noGradient.Count > 0)
                problems.Add($"{noGradient.Count} record no gradient despite being classified differentiable");
            if (skipped.Count > 0)
                problems.Add($"{skipped.Count} could not be driven by the harness at all, leaving their " +
                             "gradients unverified (add a per-op entry to the argument table so the op can be " +
                             "invoked, or add it to the Exempt table with a documented reason)");

            Assert.Fail(
                string.Join("; ", problems) + ".\n" +
                string.Join("\n", mismatches
                    .Concat(noGradient)
                    .Concat(skipped.OrderBy(x => x).Select(s => "SKIPPED " + s))
                    .Take(40)));
        }
    }

    private static object[] CopyArgs(object[] args) => (object[])args.Clone();

    /// <summary>
    /// Renders the synthesized arguments so a reported finding is reproducible by hand. Without
    /// this, a mismatch says nothing about WHICH configuration produced it, and this sweep has
    /// already generated several findings that were artifacts of its own argument choices.
    /// </summary>
    private static string DescribeArgs(MethodInfo m, object[] args)
    {
        var ps = m.GetParameters();
        var parts = new List<string>(args.Length);
        for (int i = 0; i < args.Length; i++)
        {
            string v = args[i] switch
            {
                null => "null",
                Tensor<double> t => $"Tensor[{string.Join(",", t.Shape.ToArray())}]",
                int[] a => $"[{string.Join(",", a)}]",
                _ => Convert.ToString(args[i], System.Globalization.CultureInfo.InvariantCulture) ?? "?",
            };
            parts.Add($"{ps[i].Name}={v}");
        }
        return string.Join(", ", parts);
    }

    private static Exception Inner(Exception ex) => ex is TargetInvocationException { InnerException: { } inner } ? inner : ex;
}
