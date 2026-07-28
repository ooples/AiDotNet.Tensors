#if !NETFRAMEWORK

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Third parity phase: TAPE-DRIVEN gradients. Runs each op's forward under a live GradientTape on CPU and
/// on GPU, calls ComputeGradients, and compares the resulting input gradients.
///
/// WHAT THE OTHER PHASES ACTUALLY COVER — corrected 2026-07-20, having originally claimed otherwise here:
///   CheckForward  runs the forward op and compares outputs -> verifies the FORWARD kernel. This RUNS.
///   CheckBackward is DEAD CODE. It is fully written in OpParityHarness (lines ~171-195), but NO test method
///     invokes it, and it opens with `if (!op.HasBackward) Skip` while ZERO of the 475 registry cases pass a
///     runFloatGrad/runDoubleGrad delegate — so even if a test called it, every case would skip.
///
/// The consequence is the reason this file matters: GPU BACKWARD KERNELS HAVE HAD NO CPU-VS-GPU COVERAGE AT
/// ALL. This phase is not a third check refining two existing ones — it is the ONLY thing that has ever
/// compared a GPU backward against CPU. It covers them end-to-end without needing a per-case delegate,
/// which is precisely why CheckBackward never got populated: supplying gradients by hand for 475 cases is
/// the work nobody did.
///
/// Two distinct defect classes show up here, distinguishable by dumping the recorded tape on each engine:
///   SAME tape, different gradients  -> a GPU BACKWARD KERNEL is wrong (the backward receives IEngine and
///     runs its internal ops on it). Seen in SELU, MaskedScatter, IndexFill, IndexCopy.
///   DIFFERENT tape                  -> CPU decomposes while GPU fuses, and the fused backward disagrees.
///     Seen in the IoU family: CPU records 39 ops for IoULoss, GPU records 1 fused node.
/// Neither is a wiring defect, which is what this phase was originally built to find — the Scatter class of
/// bug is real, but it turned out to be the smaller half of what the phase surfaces.
///
/// That gap is not hypothetical. GPU Scatter (2026-07-20) passed both existing phases — ScatterBackward the
/// kernel is correct — while its recording produced d(values) exactly right and d(input) WRONG BY 3.14e-01,
/// because Scatter overwrites input at the scattered positions and the reused saved state did not reproduce
/// that mask. Models would have trained on silently wrong gradients with the whole suite green.
///
/// The same gap hid ~105 ops that bailed to CPU whenever a tape was active: a bail returns the CORRECT
/// forward from CPU, so forward parity passes while the GPU kernel never runs during training.
///
/// ROLLOUT IS BASELINE-AND-RATCHET, reusing the harness's existing KnownDivergence/GpuUnsafe convention
/// rather than a parallel ledger: ops listed in <see cref="TapeGradientBaseline"/> are recorded and skipped
/// instead of failing, so this can land on a suite that is not green. The list must only ever shrink, and
/// every entry carries a measured reason.
/// </summary>
public static class TapeGradientParityHarness
{
    private const string IncompleteContractLeafReason =
        "The identity-based complete-leaf audit measured an incomplete or ambiguous CPU/GPU contract "
        + "gradient set. The old content-intersection harness silently ignored this case; it needs a "
        + "dedicated recording/backward fix before it can participate in exact one-to-one parity.";

    /// <summary>
    /// Ops whose tape gradients are known-broken or known-unverifiable. Skipped, not failed.
    /// </summary>
    /// <remarks>
    /// Each entry needs a MEASURED reason, not a suspicion. Removing one means the op's tape gradients now
    /// match; adding one means a real defect was found and should be filed. This list must only shrink.
    /// </remarks>
    public static readonly IReadOnlyDictionary<string, string> TapeGradientBaseline =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["TensorMax"] = "The (tensor, scalar) overload has NO tape recording on either engine, so there "
                        + "is no gradient to compare. CpuEngine records only (tensor, tensor).",
            ["TensorMin"] = "As TensorMax.",
            ["FusedLinearCrossEntropyWithLogits"] = "The direct-GPU fused kernels do not yet expose a "
                        + "tape-compatible backward. The GPU override rejects tape-active execution explicitly "
                        + "instead of silently downloading logits through CpuEngine.",

            // Measured legacy debt exposed by replacing partial content intersection with complete contract
            // identity matching. These entries keep the rollout baseline-and-ratchet: every operator not named
            // here must provide exactly one gradient for every differentiable contract leaf on both engines.
            ["AffineGrid3D"] = IncompleteContractLeafReason,
            ["AmplitudeToDB"] = IncompleteContractLeafReason,
            ["BatchNormBackward"] = IncompleteContractLeafReason,
            ["BoxIou"] = IncompleteContractLeafReason,
            ["CompleteBoxIou"] = IncompleteContractLeafReason,
            ["ComputeDeltas"] = IncompleteContractLeafReason,
            ["Conv1D"] = IncompleteContractLeafReason,
            ["Conv1DBackwardInput"] = IncompleteContractLeafReason,
            ["Conv1DBackwardKernel"] = IncompleteContractLeafReason,
            ["CreateMelFilterbank"] = IncompleteContractLeafReason,
            ["CreateWindow"] = IncompleteContractLeafReason,
            ["CrossEntropyBackward"] = IncompleteContractLeafReason,
            ["DeformableConv2D"] = IncompleteContractLeafReason,
            ["DeformableConv2DGrouped"] = IncompleteContractLeafReason,
            ["DepthwiseConv1DBackwardInput"] = IncompleteContractLeafReason,
            ["DepthwiseConv1DBackwardKernel"] = IncompleteContractLeafReason,
            ["DepthwiseConv2D"] = IncompleteContractLeafReason,
            ["DistanceBoxIou"] = IncompleteContractLeafReason,
            ["EluBackward"] = IncompleteContractLeafReason,
            ["FlashAttentionBackward"] = IncompleteContractLeafReason,
            ["Fold"] = IncompleteContractLeafReason,
            ["FusedConv3D"] = IncompleteContractLeafReason,
            ["FusedConvTranspose2D"] = IncompleteContractLeafReason,
            ["FusedHierarchicalSoftmax"] = IncompleteContractLeafReason,
            ["FusedLinearBackward"] = IncompleteContractLeafReason,
            ["FusedLinearMaxout"] = IncompleteContractLeafReason,
            ["GeneralizedBoxIou"] = IncompleteContractLeafReason,
            ["GridSample"] = IncompleteContractLeafReason,
            ["GroupedQueryAttentionBackward"] = IncompleteContractLeafReason,
            ["GroupNormBackward"] = IncompleteContractLeafReason,
            ["GumbelSoftmaxBackward"] = IncompleteContractLeafReason,
            ["InstanceNormBackward"] = IncompleteContractLeafReason,
            ["Interpolate"] = IncompleteContractLeafReason,
            ["InterpolateByScale"] = IncompleteContractLeafReason,
            ["LayerNormBackward"] = IncompleteContractLeafReason,
            ["LayerNormBackwardGradBeta"] = IncompleteContractLeafReason,
            ["LayerNormBackwardGradGamma"] = IncompleteContractLeafReason,
            ["LocallyConnectedConv2D"] = IncompleteContractLeafReason,
            ["MaxPool2DBackwardWithTensorIndices"] = IncompleteContractLeafReason,
            ["MaxPool2DWithIndices"] = IncompleteContractLeafReason,
            ["MaxPool3DBackwardWithTensorIndices"] = IncompleteContractLeafReason,
            ["MlpForward"] = IncompleteContractLeafReason,
            ["MuLawDecoding"] = IncompleteContractLeafReason,
            ["MultiHeadAttentionForward"] = IncompleteContractLeafReason,
            ["NativeBispectrum"] = IncompleteContractLeafReason,
            ["NativeComplexAdd"] = IncompleteContractLeafReason,
            ["NativeComplexConjugate"] = IncompleteContractLeafReason,
            ["NativeComplexCrossSpectral"] = IncompleteContractLeafReason,
            ["NativeComplexFFTComplex"] = IncompleteContractLeafReason,
            ["NativeComplexIFFT"] = IncompleteContractLeafReason,
            ["NativeComplexIFFT2DReal"] = IncompleteContractLeafReason,
            ["NativeComplexIFFTNDReal"] = IncompleteContractLeafReason,
            ["NativeComplexIFFTReal"] = IncompleteContractLeafReason,
            ["NativeComplexMagnitude"] = IncompleteContractLeafReason,
            ["NativeComplexMagnitudeSquared"] = IncompleteContractLeafReason,
            ["NativeComplexMultiply"] = IncompleteContractLeafReason,
            ["NativeComplexPhase"] = IncompleteContractLeafReason,
            ["NativeComplexScale"] = IncompleteContractLeafReason,
            ["NativeComplexTopK"] = IncompleteContractLeafReason,
            ["NativeMagnitudeAndPhase"] = IncompleteContractLeafReason,
            ["NativeTrispectrum"] = IncompleteContractLeafReason,
            ["PsRoIAlign"] = IncompleteContractLeafReason,
            ["PsRoIPool"] = IncompleteContractLeafReason,
            ["ReduceMaxBackwardWithTensorIndices"] = IncompleteContractLeafReason,
            ["ReduceStd"] = IncompleteContractLeafReason,
            ["ReduceVarianceBackward"] = IncompleteContractLeafReason,
            ["Resample"] = IncompleteContractLeafReason,
            ["RMSNormBackward"] = IncompleteContractLeafReason,
            ["RoIAlign"] = IncompleteContractLeafReason,
            ["RoIPool"] = IncompleteContractLeafReason,
            ["Rwkv7SequenceForward"] = IncompleteContractLeafReason,
            ["ScaledDotProductAttentionBackward"] = IncompleteContractLeafReason,
            ["ScatterSoftmaxBackward"] = IncompleteContractLeafReason,
            ["SoftmaxBackward"] = IncompleteContractLeafReason,
            ["SparsemaxBackward"] = IncompleteContractLeafReason,
            ["SphericalSoftmaxBackward"] = IncompleteContractLeafReason,
            ["SpiralConvBackwardBias"] = IncompleteContractLeafReason,
            ["StdBackward"] = IncompleteContractLeafReason,
            ["TaylorSoftmaxBackward"] = IncompleteContractLeafReason,
            ["TensorAddMM"] = IncompleteContractLeafReason,
            ["TensorAddScaled"] = IncompleteContractLeafReason,
            ["TensorBCEWithLogitsLoss"] = IncompleteContractLeafReason,
            ["TensorBinaryCrossEntropy"] = IncompleteContractLeafReason,
            ["TensorCIoULoss"] = IncompleteContractLeafReason,
            ["TensorClampTensor"] = IncompleteContractLeafReason,
            ["TensorCopysign"] = IncompleteContractLeafReason,
            ["TensorCross"] = IncompleteContractLeafReason,
            ["TensorCrossEntropyLoss"] = IncompleteContractLeafReason,
            ["TensorDIoULoss"] = IncompleteContractLeafReason,
            ["TensorDot"] = IncompleteContractLeafReason,
            ["TensorGIoULoss"] = IncompleteContractLeafReason,
            ["TensorHuberLoss"] = IncompleteContractLeafReason,
            ["TensorIoULoss"] = IncompleteContractLeafReason,
            ["TensorIsClose"] = IncompleteContractLeafReason,
            ["TensorKLDivLoss"] = IncompleteContractLeafReason,
            ["TensorKron"] = IncompleteContractLeafReason,
            ["TensorL1Loss"] = IncompleteContractLeafReason,
            ["TensorLdexp"] = IncompleteContractLeafReason,
            ["TensorLerp"] = IncompleteContractLeafReason,
            ["TensorMaskedSelect"] = IncompleteContractLeafReason,
            ["TensorMoveDim"] = IncompleteContractLeafReason,
            ["TensorMSELoss"] = IncompleteContractLeafReason,
            ["TensorNLLLoss"] = IncompleteContractLeafReason,
            ["TensorPolygamma"] = IncompleteContractLeafReason,
            ["TensorRandomUniform"] = IncompleteContractLeafReason,
            ["TensorRound"] = IncompleteContractLeafReason,
            ["TensorSoftmaxBackward"] = IncompleteContractLeafReason,
            ["TensorSquashBackward"] = IncompleteContractLeafReason,
            ["TensorSwapAxes"] = IncompleteContractLeafReason,
            ["TensorTakeAlongDim"] = IncompleteContractLeafReason,
            ["TensorTensorSplit"] = IncompleteContractLeafReason,
            ["TensorUnfold"] = IncompleteContractLeafReason,
            ["TensorWhere"] = IncompleteContractLeafReason,
            ["TensorZeta"] = IncompleteContractLeafReason,
            ["Unfold"] = IncompleteContractLeafReason,
            ["VarBackward"] = IncompleteContractLeafReason,
            ["VolumeRendering"] = IncompleteContractLeafReason,
        };

    /// <summary>
    /// MINIMAL REPRODUCTION for the remaining TensorNormalize / TensorSquash failures (2026-07-20).
    /// </summary>
    /// <remarks>
    /// Reduced to three lines. Reproduce with:
    ///     var y    = engine.TensorBroadcastDivide(x, engine.TensorNorm(x, 1, keepDims: true));
    ///     var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);
    ///     compare d/dx across CPU and GPU.
    ///
    /// Measured worst absolute CPU-vs-GPU gradient difference:
    ///     TensorBroadcastDivide [4,8]/[4,1]   d/dx 1.490E-08, d/dn 0.000E+00   CLEARED
    ///     TensorNorm axis=1 keepDims=true     d/dx 1.490E-08                   CLEARED
    ///     TensorNorm axis=1 keepDims=false    d/dx 1.490E-08                   CLEARED
    ///     x * x  (two-path accumulation)      0.000E+00                        CLEARED
    ///     x + x  (two-path accumulation)      0.000E+00                        CLEARED
    ///     x / norm(x)                         3.056E-01                        FAILS
    ///
    /// The GPU result is a pure ramp with CONSTANT step 0.008305 against gradOutput's step of 0.013 —
    /// gradOutput scaled by a single constant rather than divided per row, with the denominator path's
    /// contribution absent. TensorNormalize and TensorSquash both show that signature and both route
    /// through this construction.
    ///
    /// So the defect is NOT in TensorNorm, NOT in TensorBroadcastDivide, and NOT in gradient accumulation
    /// generally — each is exact alone. It appears only when the divisor is COMPUTED FROM the same tensor
    /// as the numerator. Iterate on the three-line case, not the composite op.
    /// </remarks>
    private const string NormalizeRepro =
        "y = BroadcastDivide(x, Norm(x, 1, keepDims:true)); loss = ReduceSum(y*w); compare d/dx";

    private sealed class LeafGradientSnapshot
    {
        public int ContractIndex { get; }
        public int[] Shape { get; }
        public float[] Key { get; }
        public float[] Grad { get; }

        public LeafGradientSnapshot(int contractIndex, int[] shape, float[] key, float[] grad)
        {
            ContractIndex = contractIndex;
            Shape = shape;
            Key = key;
            Grad = grad;
        }
    }

    /// <summary>
    /// Ops whose FORWARD parity already fails. Their tape gradients differ only because the forward value
    /// they are built from differs, so reporting them here double-counts a defect that belongs to
    /// OpParityTests.Forward_CpuMatchesGpu.
    /// </summary>
    /// <remarks>
    /// This phase is only meaningful where the forward is CORRECT — that is its entire purpose: catching ops
    /// whose forward matches while the recorded gradient is wrong (the Scatter class). When the forward is
    /// already broken, a gradient divergence is a downstream symptom and fixing it here would be chasing the
    /// wrong layer.
    ///
    /// Measured 2026-07-20 by intersecting both suites: of 16 tape-gradient failures, these 3 also fail
    /// forward parity and the remaining 13 do not. TensorEq's forward fails at [0] a=0 b=1, matching its
    /// tape-gradient divergence exactly. Entries leave this list when their FORWARD is fixed, at which point
    /// the tape-gradient check becomes meaningful for them again.
    /// </remarks>
    public static readonly IReadOnlyDictionary<string, string> ForwardParityBroken =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
        };

    /// <summary>
    /// Discovers the differentiable LEAVES of a tape: tensors that appear as an INPUT to some recorded op
    /// but are never the OUTPUT of one. Those are exactly the sources ComputeGradients should be asked for.
    /// </summary>
    /// <remarks>
    /// This is what makes the phase work across the whole registry WITHOUT touching any of the 475 OpCase
    /// definitions. The earlier design required each case to declare its sources, because OpCase.RunFloat
    /// closes over its inputs and returns only the output — but the tape already knows them: every recorded
    /// entry carries its Output and Inputs, so the leaf set is derivable.
    ///
    /// ORDER IS BY FIRST APPEARANCE in the tape, which is deterministic for a given forward and therefore
    /// aligns between the CPU and GPU runs. That alignment is what lets gradients be compared PER SOURCE —
    /// the property that mattered for Scatter, where one operand was exact and another was corrupt.
    /// </remarks>
    private static List<Tensor<float>> DiscoverLeaves(GradientTape<float> tape)
    {
        var producedByAnOp = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        var seenInputs = new List<Tensor<float>>();
        var seenSet = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);

        for (int i = 0; i < tape.EntryCount; i++)
        {
            ref var e = ref tape.Entries[i];
            if (e.Output is not null) producedByAnOp.Add(e.Output);
            foreach (var inp in e.GetInputsArray())
                if (inp is not null && seenSet.Add(inp)) seenInputs.Add(inp);
        }

        // A leaf is an input that no recorded op produced.
        var leaves = new List<Tensor<float>>();
        foreach (var t in seenInputs)
            if (!producedByAnOp.Contains(t)) leaves.Add(t);
        return leaves;
    }

    /// <summary>
    /// Runs a tape case on the given engine and returns the concatenated gradients over its sources, in the
    /// case's declared order, plus the deferred GPU-&gt;host materialisation count for that run.
    /// </summary>
    private static (List<LeafGradientSnapshot> Leaves, long Materialisations, int LeafCount,
        bool UsesContractIdentity, int CapturedInputCount) TapeGradient(
        IEngine engine, Func<IEngine, Tensor<float>> runFloat)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        var capturedInputs = new List<Tensor<float>>();
        var sources = new List<Tensor<float>>();
        Dictionary<Tensor<float>, Tensor<float>>? grads = null;
        Tensor<float>? output = null;
        Tensor<float>? weight = null;
        Tensor<float>? weighted = null;
        Tensor<float>? loss = null;
        try
        {
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
            using var tape = new GradientTape<float>();

            try
            {
                using (OpInput.CaptureFloatInputs(capturedInputs))
                    output = runFloat(engine);

                // Non-uniform weighting: a constant upstream gradient can mask a wrong backward, because several
                // different index mappings then produce the same total.
                weight = new Tensor<float>(output.Shape.ToArray());
                for (int i = 0; i < weight.Length; i++) weight[i] = 0.19f + 0.013f * (i % 13);
                weighted = engine.TensorMultiply(output, weight);
                loss = engine.ReduceSum(weighted, null);

                sources = DiscoverLeaves(tape);
                bool usesContractIdentity = capturedInputs.Count > 0;
                if (sources.Count == 0)
                    return (new List<LeafGradientSnapshot>(),
                        AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount,
                        0, usesContractIdentity, capturedInputs.Count);

                grads = tape.ComputeGradients(loss, sources.ToArray());

                var capturedOrdinals = new Dictionary<Tensor<float>, int>(
                    ReferenceEqualityComparer<Tensor<float>>.Instance);
                for (int i = 0; i < capturedInputs.Count; i++)
                {
                    if (capturedOrdinals.ContainsKey(capturedInputs[i]))
                        throw new InvalidOperationException("The same OpInput tensor was captured more than once in one parity run.");
                    capturedOrdinals.Add(capturedInputs[i], i);
                }

                var leaves = new List<LeafGradientSnapshot>();
                foreach (var src in sources)
                {
                    int contractIndex = -1;
                    if (usesContractIdentity && !capturedOrdinals.TryGetValue(src, out contractIndex))
                        continue; // An implementation-only temporary, not an OpCase input contract.

                    if (!grads.TryGetValue(src, out var gradient) || gradient is null)
                        throw new InvalidOperationException(
                            $"GradientTape omitted an expected contract-leaf gradient with shape "
                            + $"[{string.Join(",", src.Shape._dims)}].");

                    var key = new float[src.Length];
                    for (int i = 0; i < src.Length; i++) key[i] = src[i];
                    var gradientValues = new float[gradient.Length];
                    for (int i = 0; i < gradient.Length; i++) gradientValues[i] = gradient[i];
                    leaves.Add(new LeafGradientSnapshot(
                        contractIndex, src.Shape.ToArray(), key, gradientValues));
                }

                long materialisations = AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount;
                return (leaves, materialisations, leaves.Count, usesContractIdentity, capturedInputs.Count);
            }
            finally
            {
                var disposed = new HashSet<Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
                void DisposeOnce(Tensor<float>? tensor)
                {
                    if (tensor is not null && disposed.Add(tensor)) tensor.Dispose();
                }

                if (grads is not null)
                    foreach (var gradient in grads.Values) DisposeOnce(gradient);
                DisposeOnce(loss);
                DisposeOnce(weighted);
                DisposeOnce(weight);
                DisposeOnce(output);
                foreach (var source in sources) DisposeOnce(source);
                foreach (var input in capturedInputs) DisposeOnce(input);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    /// <summary>
    /// Compares tape-driven gradients for one op across CPU and GPU.
    /// </summary>
    public static void CheckTapeGradient(OpCase op, OpParityFixture fx)
    {
        string opName = op.Name;
        string category = op.Category;

        // Registry names are PARAMETERISED — "TensorCross[4,3]", not "TensorCross" — so keying the baseline
        // on the bare op name matched NOTHING and the whole ledger was inert (a full run reported
        // "Skipped: 0" with baselined ops still executing). Match on the base name before '[' so an entry
        // covers every shape variant of that op, which is the granularity the debt is actually tracked at.
        int bracket = opName.IndexOf('[');
        string baseName = bracket >= 0 ? opName.Substring(0, bracket) : opName;
        if (ForwardParityBroken.TryGetValue(baseName, out var fwdReason))
        {
            fx.Record($"{opName}:tapegrad\t{category}\tFORWARD-BROKEN\t-\t-\t-");
            Skip.If(true, $"FORWARD PARITY ALREADY FAILS ({opName}): {fwdReason}");
            return;
        }

        if (TapeGradientBaseline.TryGetValue(baseName, out var baselineReason))
        {
            fx.Record($"{opName}:tapegrad\t{category}\tBASELINE\t-\t-\t-");
            Skip.If(true, $"TAPE-GRADIENT BASELINE ({opName}): {baselineReason}");
            return;
        }

        if (!fx.GpuReady)
        {
            if (fx.RequireGpu)
                throw new InvalidOperationException($"{opName}: GPU required but unavailable.", fx.GpuInitError);
            Skip.If(true, "No DirectGpu backend available.");
            return;
        }


        MaybeResetGpuEngineViaHarness(fx);
        var gpu = fx.Gpu;
        if (gpu is null) { Skip.If(true, "GPU engine unavailable after reset."); return; }

        var (cpuLeafList, _, cpuLeaves, cpuUsesContractIdentity, cpuCapturedInputs) =
            TapeGradient(fx.Cpu, op.RunFloat);
        var (gpuLeafList, materialisations, gpuLeaves, gpuUsesContractIdentity, gpuCapturedInputs) =
            TapeGradient(gpu, op.RunFloat);

        if (cpuLeaves == 0 && gpuLeaves == 0)
        {
            // The forward recorded no differentiable leaf. Legitimate for genuinely non-differentiable ops
            // (argmax, discrete indices, integer gather). Reported so it stays countable rather than
            // reading as a pass.
            fx.Record($"{opName}:tapegrad	{category}	NO-LEAF	-	-	-");
            Skip.If(true, $"{opName}: forward recorded no differentiable leaf.");
            return;
        }

        // OpInput capture supplies a stable contract ordinal across CPU and GPU runs. This excludes
        // implementation-only temporaries while still requiring every user-facing differentiable input to
        // have exactly one gradient on both engines. Content matching is only a fail-closed fallback for the
        // rare registry case that constructs inputs without OpInput.
        var pairs = new List<(float[] Cpu, float[] Gpu)>();
        Assert.True(cpuUsesContractIdentity == gpuUsesContractIdentity,
            $"{opName} tapegrad: only one engine captured OpInput contract identities.");

        if (cpuUsesContractIdentity)
        {
            Assert.True(cpuCapturedInputs == gpuCapturedInputs,
                $"{opName} tapegrad: the registry materialized a different number of inputs "
                + $"(CPU={cpuCapturedInputs}, GPU={gpuCapturedInputs}).");

            var cpuByIndex = cpuLeafList.ToDictionary(leaf => leaf.ContractIndex);
            var gpuByIndex = gpuLeafList.ToDictionary(leaf => leaf.ContractIndex);
            Assert.True(cpuByIndex.Count == gpuByIndex.Count,
                $"{opName} tapegrad: differentiable contract-leaf count differs "
                + $"(CPU={cpuByIndex.Count}, GPU={gpuByIndex.Count}).");

            foreach (var (contractIndex, cpuLeaf) in cpuByIndex)
            {
                Assert.True(gpuByIndex.TryGetValue(contractIndex, out var gpuLeaf),
                    $"{opName} tapegrad: GPU omitted gradient for contract input #{contractIndex}.");
                Assert.True(cpuLeaf.Shape.AsSpan().SequenceEqual(gpuLeaf!.Shape),
                    $"{opName} tapegrad: contract input #{contractIndex} has different CPU/GPU shapes.");
                Assert.True(cpuLeaf.Key.AsSpan().SequenceEqual(gpuLeaf.Key),
                    $"{opName} tapegrad: contract input #{contractIndex} has different CPU/GPU values.");
                pairs.Add((cpuLeaf.Grad, gpuLeaf.Grad));
            }

            Assert.True(gpuByIndex.Keys.All(cpuByIndex.ContainsKey),
                $"{opName} tapegrad: GPU produced a gradient for a contract input that CPU did not mark differentiable.");
        }
        else
        {
            // Cases that construct tensors without OpInput fall back to content keys. Fail closed when a key
            // is duplicated, and require every leaf on both sides to be consumed one-to-one.
            static bool SameKey(LeafGradientSnapshot left, LeafGradientSnapshot right)
                => left.Shape.AsSpan().SequenceEqual(right.Shape)
                    && left.Key.AsSpan().SequenceEqual(right.Key);

            static void RequireUniqueKeys(List<LeafGradientSnapshot> leaves, string side, string name)
            {
                for (int i = 0; i < leaves.Count; i++)
                    for (int j = i + 1; j < leaves.Count; j++)
                        Assert.False(SameKey(leaves[i], leaves[j]),
                            $"{name} tapegrad: {side} has duplicate-valued leaves, so content pairing is ambiguous.");
            }

            RequireUniqueKeys(cpuLeafList, "CPU", opName);
            RequireUniqueKeys(gpuLeafList, "GPU", opName);
            Assert.True(cpuLeafList.Count == gpuLeafList.Count,
                $"{opName} tapegrad: untracked leaf count differs "
                + $"(CPU={cpuLeafList.Count}, GPU={gpuLeafList.Count}).");

            var unmatchedGpu = new List<LeafGradientSnapshot>(gpuLeafList);
            foreach (var cpuLeaf in cpuLeafList)
            {
                int hit = unmatchedGpu.FindIndex(gpuLeaf => SameKey(cpuLeaf, gpuLeaf));
                Assert.True(hit >= 0,
                    $"{opName} tapegrad: GPU omitted a CPU leaf with shape [{string.Join(",", cpuLeaf.Shape)}].");
                pairs.Add((cpuLeaf.Grad, unmatchedGpu[hit].Grad));
                unmatchedGpu.RemoveAt(hit);
            }
            Assert.True(unmatchedGpu.Count == 0,
                $"{opName} tapegrad: GPU had {unmatchedGpu.Count} unmatched leaf gradients.");
        }

        Assert.True(pairs.Count > 0,
            $"{opName} tapegrad: no differentiable contract leaf could be paired between the engines "
            + $"(CPU {cpuLeaves} leaves, GPU {gpuLeaves}).");

        double maxAbs = 0, maxRel = 0;
        foreach (var (cg, gg) in pairs)
        {
            Assert.True(cg.Length == gg.Length,
                $"{opName} tapegrad: paired leaves have identical inputs but gradients of different length "
                + $"(CPU={cg.Length} GPU={gg.Length}) — the engines disagree about that input's gradient shape.");
            for (int i = 0; i < cg.Length; i++)
            {
                double d = Math.Abs(cg[i] - gg[i]);
                maxAbs = Math.Max(maxAbs, d);
                maxRel = Math.Max(maxRel, d / Math.Max(1.0, Math.Abs(cg[i])));
            }
        }

        fx.Record($"{opName}:tapegrad\t{category}\tOK\t{maxAbs:E3}\t{maxRel:E3}\t{materialisations}");

        // ANTI-VACUITY: residency, not divergence.
        //
        // The first version of this asserted maxAbs > 0 on the theory that two independent float32
        // implementations cannot agree to the last bit, with a hand-written whitelist for exceptions. Run
        // across the whole registry that produced 303 failures — because bit-identity is the NORM, not the
        // exception: any op whose backward does not consume forward-computed values (index, geometry, and
        // pure data-movement gradients) agrees exactly on both engines by construction.
        //
        // Divergence was only ever a PROXY for "did the device path run". MaterializeCount measures that
        // directly: a non-zero count means results were GPU-resident and had to be downloaded. It is the
        // right probe for every op, with no whitelist to maintain and no false positives.
        Assert.True(materialisations > 0,
            $"{opName} tapegrad: no deferred GPU->host materialisation, so nothing was ever GPU-resident "
            + "and the device path did not run. Either the forward bails under an active tape, or it has no "
            + "GPU override at all — in both cases the comparison would be CPU against CPU.");

        Assert.True(maxRel <= 1e-4,
            $"{opName} tapegrad: GPU gradient diverged from CPU (maxAbs={maxAbs:E3} maxRel={maxRel:E3}). "
            + "A gross mismatch means the recorded backward, its saved state, or the overload is wrong — "
            + "NOT float rounding. Scatter failed exactly this way at 3.14e-01 while both the forward and "
            + "the backward KERNEL were correct.");
    }

    /// <summary>Shares the forward phase's engine-reset cadence.</summary>
    /// <remarks>
    /// This was a deliberate no-op, on the reasoning that OpParityHarness owned the counter and calling it
    /// here would reset twice as often. That reasoning was wrong, and it cost real time: TensorSELU PASSES
    /// this phase in isolation and FAILS it deep in a 557-test run, which is precisely the failure mode
    /// OpParityHarness already documents — a driver that "accumulates internal state across a large number
    /// of kernel dispatches and, past a threshold, returns subtly wrong results for some later kernels
    /// (correct in isolation, wrong only deep into a long full-suite run)". The other phases defend against
    /// it by recreating the GPU engine every 64 ops; this phase adds ~557 more dispatches and did not.
    ///
    /// Sharing the SAME static counter is the point: the threshold is about dispatches per context, so all
    /// phases must count into one budget. That is the opposite of double-resetting.
    /// </remarks>
    private static void MaybeResetGpuEngineViaHarness(OpParityFixture fx)
        => OpParityHarness.MaybeResetGpuEngine(fx);

}
#endif
