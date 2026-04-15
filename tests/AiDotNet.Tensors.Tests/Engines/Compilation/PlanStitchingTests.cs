using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Acceptance tests for issue #170 — plan stitching via
/// <see cref="ICompiledPlan{T}.Then"/>. Validates that two compiled
/// inference plans can be composed into a single replay-equivalent plan
/// without per-execute tensor materialization at the boundary.
/// </summary>
public class PlanStitchingTests
{
    private static readonly Random Rng = new(unchecked((int)0xC0FFEE));

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Builds a "MatMul-then-Sigmoid" inference plan against the given input
    /// tensor reference. Two ops are enough to be representative (the
    /// boundary copy spans plan A's last step and plan B's first step).
    /// Sigmoid keeps the test arithmetic numerically tame and shape-preserving.
    /// </summary>
    private static ICompiledPlan<float> CompileMatMulRelu(
        IEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var product = engine.TensorMatMul(input, weight);
            engine.Sigmoid(product);
            plan = scope.CompileInference<float>();
        }
        return plan;
    }

    private static void RandomizeInPlace(Tensor<float> t, int seed)
    {
        var rng = new Random(seed);
        var data = t.GetDataArray();
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    private static float[] Snapshot(Tensor<float> t) => t.AsSpan().ToArray();

    // ── Acceptance criterion #1: bitwise equivalence ─────────────────────────
    [Fact]
    public void Then_Execute_BitwiseIdenticalToSequentialExecution_Across100RandomInputs()
    {
        var engine = new CpuEngine();

        // A: [4,3] @ [3,5] → relu → [4,5]
        var inputA  = Tensor<float>.CreateRandom([4, 3]);
        var weightA = Tensor<float>.CreateRandom([3, 5]);

        // B: [4,5] @ [5,2] → relu → [4,2]
        var inputB  = Tensor<float>.CreateRandom([4, 5]);
        var weightB = Tensor<float>.CreateRandom([5, 2]);

        var planA = CompileMatMulRelu(engine, inputA, weightA);
        var planB = CompileMatMulRelu(engine, inputB, weightB);

        // Build the stitched plan. After this call, planA.Then(planB)'s
        // execute should be observationally equivalent to running planA,
        // then copying planA's output into planB's input, then running planB.
        using var stitched = planA.Then(planB);

        for (int trial = 0; trial < 100; trial++)
        {
            // Randomize ONLY the leaf input — weights and biases stay fixed
            // so both paths see the same parameters. Same seed each trial so
            // stitched and sequential paths use identical data.
            RandomizeInPlace(inputA, trial);

            // Sequential: A → manual boundary copy → B
            var aOut = planA.Execute();
            aOut.AsSpan().CopyTo(inputB.AsWritableSpan());
            var bOutSequential = planB.Execute();
            float[] sequential = Snapshot(bOutSequential);

            // Re-randomize because Execute() mutated buffers; reset and re-run via stitched
            RandomizeInPlace(inputA, trial);
            var bOutStitched = stitched.Execute();
            float[] stitchedResult = Snapshot(bOutStitched);

            Assert.Equal(sequential.Length, stitchedResult.Length);
            for (int i = 0; i < sequential.Length; i++)
            {
                // Bitwise comparison: both paths execute the same scalar ops
                // in the same order on the same data, so floats must be ==.
                Assert.Equal(sequential[i], stitchedResult[i]);
            }
        }

        planA.Dispose();
        planB.Dispose();
    }

    // ── Acceptance criterion #2: zero intermediate materialization ──────────
    [Fact]
    public void Then_StitchedPlan_HasExactlyOneBoundaryStep_NoNewTensorMaterialization()
    {
        // The structural acceptance criterion: a stitched plan's step count
        // is exactly A.StepCount + 1 (the boundary copy) + B.StepCount.
        // No "materialize intermediate tensor" step is inserted because both
        // boundary buffers were pre-allocated at compile time. The boundary
        // step is an in-place memcpy between existing storage.
        var engine = new CpuEngine();

        var inA = Tensor<float>.CreateRandom([2, 3]);
        var wA  = Tensor<float>.CreateRandom([3, 4]);

        var inB = Tensor<float>.CreateRandom([2, 4]);
        var wB  = Tensor<float>.CreateRandom([4, 1]);

        var planA = CompileMatMulRelu(engine, inA, wA);
        var planB = CompileMatMulRelu(engine, inB, wB);

        using var stitched = planA.Then(planB);

        Assert.Equal(planA.StepCount + 1 + planB.StepCount, stitched.StepCount);

        planA.Dispose();
        planB.Dispose();
    }

    // ── Acceptance criterion #3: stitch-time validation ─────────────────────
    [Fact]
    public void Then_WithMismatchedShapes_ThrowsAtStitchTime_NotExecuteTime()
    {
        var engine = new CpuEngine();

        // A's output is [4,5], but B's input is [4,7] — incompatible.
        var planA = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([4, 3]),
            Tensor<float>.CreateRandom([3, 5]));
        var planB = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([4, 7]),
            Tensor<float>.CreateRandom([7, 2]));

        var ex = Assert.Throws<ArgumentException>(() => planA.Then(planB));
        // Error message must name the shapes for debuggability.
        Assert.Contains("[4, 5]", ex.Message);
        Assert.Contains("[4, 7]", ex.Message);

        planA.Dispose();
        planB.Dispose();
    }

    // ── Argument validation ─────────────────────────────────────────────────
    [Fact]
    public void Then_WithNullNext_ThrowsArgumentNullException()
    {
        var engine = new CpuEngine();
        var plan = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 2]),
            Tensor<float>.CreateRandom([2, 2]));

        Assert.Throws<ArgumentNullException>(() => plan.Then(null!));
        plan.Dispose();
    }

    [Fact]
    public void Then_WithNonBuiltInImplementation_ThrowsNotSupportedException()
    {
        // Stitching needs to splice each plan's internal step array, which
        // is implementation-specific. A foreign ICompiledPlan<T> can't be
        // stitched generically — the API rejects it cleanly rather than
        // guessing at a fallback path.
        var engine = new CpuEngine();
        var plan = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 2]),
            Tensor<float>.CreateRandom([2, 2]));

        var stub = new StubCompiledPlan();
        Assert.Throws<NotSupportedException>(() => plan.Then(stub));
        plan.Dispose();
    }

    // ── Associativity ───────────────────────────────────────────────────────
    [Fact]
    public void Then_IsAssociative_LeftFoldEquivalentToRightFold()
    {
        // (a.Then(b)).Then(c) and a.Then(b.Then(c)) must produce
        // structurally-equivalent stitched plans with the same step count
        // and the same final output values for the same inputs.
        var engine = new CpuEngine();

        var inA = Tensor<float>.CreateRandom([2, 3]);
        var planA = CompileMatMulRelu(engine, inA,
            Tensor<float>.CreateRandom([3, 4]));
        var planB = CompileMatMulRelu(engine, Tensor<float>.CreateRandom([2, 4]),
            Tensor<float>.CreateRandom([4, 5]));
        var planC = CompileMatMulRelu(engine, Tensor<float>.CreateRandom([2, 5]),
            Tensor<float>.CreateRandom([5, 6]));

        // Step counts of each side: same total (boundary steps are 2 either way).
        using var leftFold  = (planA.Then(planB)).Then(planC);
        using var rightFold = planA.Then(planB.Then(planC));

        Assert.Equal(leftFold.StepCount, rightFold.StepCount);

        // Numeric equivalence: same input → same output, bitwise.
        RandomizeInPlace(inA, seed: 7);
        var leftOut  = Snapshot(leftFold.Execute());
        RandomizeInPlace(inA, seed: 7);
        var rightOut = Snapshot(rightFold.Execute());

        Assert.Equal(leftOut.Length, rightOut.Length);
        for (int i = 0; i < leftOut.Length; i++)
            Assert.Equal(leftOut[i], rightOut[i]);

        planA.Dispose();
        planB.Dispose();
        planC.Dispose();
    }

    // ── Re-entrancy: stitched plans can themselves be stitched ──────────────
    [Fact]
    public void Then_IsReEntrant_StitchedPlanCanBeStitchedAgain()
    {
        var engine = new CpuEngine();

        var planA = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 3]),
            Tensor<float>.CreateRandom([3, 4]));
        var planB = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 4]),
            Tensor<float>.CreateRandom([4, 5]));
        var planC = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 5]),
            Tensor<float>.CreateRandom([5, 6]));

        var ab  = planA.Then(planB);
        using var abc = ab.Then(planC); // re-entrant: ab is itself a stitched plan

        // Should not throw; output shape comes from C.
        var output = abc.Execute();
        Assert.Equal(2, output._shape[0]);
        Assert.Equal(6, output._shape[1]);

        ab.Dispose();
        planA.Dispose();
        planB.Dispose();
        planC.Dispose();
    }

    // ── Ownership: stitched plan does NOT dispose its constituents ──────────
    [Fact]
    public void Then_StitchedPlan_Dispose_DoesNotDisposeOriginalPlans()
    {
        // The xmldoc contract: callers retain ownership of `this` and `next`.
        // Disposing the stitched plan must NOT invalidate the originals — a
        // single plan can participate in multiple stitched pipelines, so
        // shared ownership of step arrays is fine but pinned-handle disposal
        // must stay with the originals.
        var engine = new CpuEngine();

        var inA = Tensor<float>.CreateRandom([2, 3]);
        var planA = CompileMatMulRelu(engine, inA,
            Tensor<float>.CreateRandom([3, 4]));
        var planB = CompileMatMulRelu(engine,
            Tensor<float>.CreateRandom([2, 4]),
            Tensor<float>.CreateRandom([4, 1]));

        var stitched = planA.Then(planB);

        RandomizeInPlace(inA, seed: 123);
        var stitchedOutput = Snapshot(stitched.Execute());

        // Dispose the stitched plan FIRST.
        stitched.Dispose();

        // Originals must still be usable.
        RandomizeInPlace(inA, seed: 123);
        var aOutAfter = Snapshot(planA.Execute()); // Should not throw ObjectDisposed
        Assert.NotEmpty(aOutAfter);

        var bOutAfter = Snapshot(planB.Execute()); // Should not throw
        Assert.NotEmpty(bOutAfter);

        planA.Dispose();
        planB.Dispose();
    }

    // ── Empty-plan rejection ────────────────────────────────────────────────
    // Note: constructing an empty plan via the public API isn't directly
    // possible (CompileInference always produces ≥1 step). The internal guard
    // against empty plans exists to defend against future code paths that
    // might construct empty plans (e.g., dead-code-elimination passes); it's
    // structurally hard to reach from a black-box test, so we exercise it
    // only via the negative test above (mismatched shapes also flow through
    // the empty-plan validation).

    // ── Stub for NotSupportedException test ─────────────────────────────────
    private sealed class StubCompiledPlan : ICompiledPlan<float>
    {
        public Tensor<float> Execute() => new Tensor<float>(new[] { 1 });
        public bool IsValid(int[] inputShape) => true;
        public int StepCount => 0;
        public ICompiledPlan<float> Then(ICompiledPlan<float> next) => this;
        public void Dispose() { }
    }
}
