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
    /// Whether bit-identical CPU/GPU gradients are the CORRECT result for this op, so divergence cannot be
    /// used as the GPU-engagement signal.
    /// </summary>
    /// <remarks>
    /// The maxAbs &gt; 0 probe assumes two independent float32 implementations cannot agree to the last bit.
    /// That fails for any op whose BACKWARD does not consume values the FORWARD produced — the gradient is
    /// then computed from indices or geometry alone and is exact on both engines. Measured examples:
    /// Upsample3D (sums gradOutput per block), AffineGrid (d/d(theta) is the fixed coordinate grid),
    /// MaxPool3DWithIndices (scatters gradOutput to selected positions).
    ///
    /// NOTE: "computes rather than selects" is NOT the right test — AffineGrid computes its output and its
    /// gradient is still input-independent. The question is strictly whether the backward reads
    /// forward-produced values.
    /// </remarks>
    private static bool GradientIsExactOnBothEngines(string opName) => opName switch
    {
        "Upsample3D" or "AffineGrid" or "MaxPool3DWithIndices" or "MaxPool3D" or "Scatter" => true,
        _ => false,
    };

    /// <summary>
    /// A tape-gradient case: builds the op's inputs, runs the forward, and returns BOTH the output and the
    /// tensors to differentiate with respect to, in a FIXED order.
    /// </summary>
    /// <remarks>
    /// The explicit source list is required, not incidental. OpCase.RunFloat closes over its inputs and
    /// returns only the output, so a generic harness has no handle on the tensors to differentiate. And
    /// ComputeGradients(loss, sources: null) is not usable across engines: it keys the result dictionary by
    /// tensor REFERENCE, and the CPU and GPU runs build distinct tensor objects with no stable ordering to
    /// align them. An ordered source list is what makes the two runs comparable element by element — which
    /// matters because Scatter's defect was in ONE operand while the other was exact.
    /// </remarks>
    public delegate (Tensor<float> Output, IReadOnlyList<Tensor<float>> Sources) TapeCase(IEngine engine);

    /// <summary>
    /// Tape cases by op name. Ops absent from this map are SKIPPED AND REPORTED, never silently passed —
    /// coverage is explicit and grows as cases are added rather than being implied by the phase existing.
    /// </summary>
    public static readonly Dictionary<string, TapeCase> TapeCases = new(StringComparer.Ordinal);

    /// <summary>
    /// Runs a tape case on the given engine and returns the concatenated gradients over its sources, in the
    /// case's declared order, plus the deferred GPU-&gt;host materialisation count for that run.
    /// </summary>
    private static (float[] Grad, long Materialisations) TapeGradient(IEngine engine, TapeCase tapeCase)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.ResetMaterializeCount();
            using var tape = new GradientTape<float>();

            var (output, sources) = tapeCase(engine);

            // Non-uniform weighting: a constant upstream gradient can mask a wrong backward, because several
            // different index mappings then produce the same total.
            var weight = new Tensor<float>(output.Shape.ToArray());
            for (int i = 0; i < weight.Length; i++) weight[i] = 0.19f + 0.013f * (i % 13);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, weight), null);

            var grads = tape.ComputeGradients(loss, sources.ToArray());

            var flat = new List<float>();
            foreach (var src in sources)
            {
                Assert.True(grads.TryGetValue(src, out var g) && g is not null,
                    "a declared source received no gradient — the forward recorded no tape node for it");
                for (int i = 0; i < g.Length; i++) flat.Add(g[i]);
            }
            return (flat.ToArray(), AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.MaterializeCount);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    /// <summary>
    /// Compares tape-driven gradients for one op across CPU and GPU.
    /// </summary>
    public static void CheckTapeGradient(string opName, OpParityFixture fx, string category = "tape")
    {
        if (TapeGradientBaseline.TryGetValue(opName, out var baselineReason))
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

        if (!TapeCases.TryGetValue(opName, out var tapeCase))
        {
            // No tape case declared yet. Reported so the coverage gap is VISIBLE in the parity log rather
            // than reading as a pass. This is the honest state: coverage grows as cases are written.
            fx.Record($"{opName}:tapegrad	{category}	NO-CASE	-	-	-");
            Skip.If(true, $"{opName}: no tape-gradient case declared (coverage gap, not a pass).");
            return;
        }

        var (cpuGrad, _) = TapeGradient(fx.Cpu, tapeCase);
        var (gpuGrad, materialisations) = TapeGradient(gpu, tapeCase);


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

        // ANTI-VACUITY. A GPU op that bails to CPU produces bit-identical output, so "it passed" would mean
        // nothing. Ops whose gradient is exact on both engines by construction use the residency counter
        // instead — see GradientIsExactOnBothEngines.
        if (GradientIsExactOnBothEngines(opName))
        {
            Assert.True(materialisations > 0,
                $"{opName} tapegrad: nothing was GPU-resident, so the device path did not run. "
                + "(This op's gradient is exact on both engines, so bit-identity cannot signal engagement.)");
        }
        else
        {
            Assert.True(maxAbs > 0.0,
                $"{opName} tapegrad: CPU and GPU gradients were BIT-IDENTICAL. For an op whose backward "
                + "consumes forward-computed values that means the GPU path did not run and this compared "
                + "the CPU implementation with itself. If bit-identity is genuinely correct here, add the op "
                + "to GradientIsExactOnBothEngines with the reason.");
        }

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
