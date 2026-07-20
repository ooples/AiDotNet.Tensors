using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Third parity phase: TAPE-DRIVEN gradients. Runs each op's forward under a live GradientTape on CPU and
/// on GPU, calls ComputeGradients, and compares the resulting input gradients.
///
/// WHY THE EXISTING TWO PHASES CANNOT CATCH THIS:
///   CheckForward  runs the forward op and compares outputs      -> verifies the FORWARD kernel.
///   CheckBackward calls the backward op DIRECTLY (RunFloatGrad) -> verifies the BACKWARD kernel.
/// Neither exercises the WIRING between them: whether the forward records a tape node at all, and whether
/// the saved state it records makes ComputeGradients produce the right gradient.
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
            ["Scatter"] = "GPU forward bails on an active tape because recording CpuEngine's ScatterBackward "
                        + "gave d(values) exact but d(input) wrong by 3.14e-01 — Scatter overwrites input at "
                        + "the scattered positions and the reused saved state does not reproduce that mask. "
                        + "Needs a GPU-side ScatterBackward.",
            ["Sparsemax"] = "GPU forward bails: the path throws InvalidOperationException because "
                        + "CudaBackend.Where looks up a \"where_select\" kernel that has no source and no "
                        + "registration on any backend.",
            ["TensorMax"] = "The (tensor, scalar) overload has NO tape recording on either engine, so there "
                        + "is no gradient to compare. CpuEngine records only (tensor, tensor).",
            ["TensorMin"] = "As TensorMax.",
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
        var producedByAnOp = new HashSet<Tensor<float>>(ReferenceEqualityComparer.Instance
            as IEqualityComparer<Tensor<float>> ?? EqualityComparer<Tensor<float>>.Default);
        var seenInputs = new List<Tensor<float>>();
        var seenSet = new HashSet<Tensor<float>>(ReferenceEqualityComparer.Instance
            as IEqualityComparer<Tensor<float>> ?? EqualityComparer<Tensor<float>>.Default);

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
    private static (float[] Grad, long Materialisations, int LeafCount) TapeGradient(
        IEngine engine, Func<IEngine, Tensor<float>> runFloat)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
            using var tape = new GradientTape<float>();

            var output = runFloat(engine);

            // Non-uniform weighting: a constant upstream gradient can mask a wrong backward, because several
            // different index mappings then produce the same total.
            var weight = new Tensor<float>(output.Shape.ToArray());
            for (int i = 0; i < weight.Length; i++) weight[i] = 0.19f + 0.013f * (i % 13);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, weight), null);

            var sources = DiscoverLeaves(tape);
            if (sources.Count == 0)
                return (Array.Empty<float>(), AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount, 0);

            var grads = tape.ComputeGradients(loss, sources.ToArray());

            var flat = new List<float>();
            foreach (var src in sources)
            {
                if (!grads.TryGetValue(src, out var g) || g is null) continue;
                for (int i = 0; i < g.Length; i++) flat.Add(g[i]);
            }
            return (flat.ToArray(), AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount, sources.Count);
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

        var (cpuGrad, _, cpuLeaves) = TapeGradient(fx.Cpu, op.RunFloat);
        var (gpuGrad, materialisations, gpuLeaves) = TapeGradient(gpu, op.RunFloat);

        if (cpuLeaves == 0 && gpuLeaves == 0)
        {
            // The forward recorded no differentiable leaf. Legitimate for genuinely non-differentiable ops
            // (argmax, discrete indices, integer gather). Reported so it stays countable rather than
            // reading as a pass.
            fx.Record($"{opName}:tapegrad	{category}	NO-LEAF	-	-	-");
            Skip.If(true, $"{opName}: forward recorded no differentiable leaf.");
            return;
        }

        Assert.True(cpuLeaves == gpuLeaves,
            $"{opName} tapegrad: CPU discovered {cpuLeaves} differentiable leaves, GPU discovered {gpuLeaves}. "
            + "The two engines recorded structurally different tapes for the same forward — that alone is a "
            + "defect, independent of gradient values.");


        Assert.True(cpuGrad.Length == gpuGrad.Length,
            $"{opName} tapegrad: gradient length differs CPU={cpuGrad.Length} GPU={gpuGrad.Length}. "
            + "One engine recorded a different set of differentiable inputs than the other.");

        double maxAbs = 0, maxRel = 0;
        for (int i = 0; i < cpuGrad.Length; i++)
        {
            double d = Math.Abs(cpuGrad[i] - gpuGrad[i]);
            maxAbs = Math.Max(maxAbs, d);
            maxRel = Math.Max(maxRel, d / Math.Max(1.0, Math.Abs(cpuGrad[i])));
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

    /// <summary>Mirrors the engine-reset cadence the forward/backward phases use.</summary>
    private static void MaybeResetGpuEngineViaHarness(OpParityFixture fx)
    {
        // Deliberately a no-op hook: OpParityHarness owns the reset counter, and duplicating it here would
        // reset twice as often and change the very GPU-pool state the reset exists to bound. Wired through
        // the existing phases' call instead when this phase is registered into the generated test set.
    }
}
