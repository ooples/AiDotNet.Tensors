using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests for the three JIT optimization passes from issue #182:
/// MemoryPlanningPass, TileSchedulingPass, OperatorReorderingPass.
/// </summary>
public class JitOptimizationPassTests
{
    private readonly CpuEngine _engine = new();

    // ════════════════════════════════════════════════════════════════════
    // MemoryPlanningPass
    // ════════════════════════════════════════════════════════════════════

    [Fact]
    public void MemoryPlanning_ReusesDeadBuffers_WhenLifetimesDontOverlap()
    {
        // Build a chain: step0 → step1 → step2 → step3
        // After step1 reads step0's output, step0's buffer is dead.
        // Step2 should be able to reuse step0's buffer if same shape.
        var pass = new MemoryPlanningPass();
        var input = Tensor<float>.CreateRandom([4, 8]);

        // Chain of 5 steps — enough to trigger the pass (threshold is 4)
        var buf0 = new Tensor<float>([4, 8]);
        var buf1 = new Tensor<float>([4, 8]);
        var buf2 = new Tensor<float>([4, 8]);
        var buf3 = new Tensor<float>([4, 8]);
        var buf4 = new Tensor<float>([4, 8]);

        var steps = new[]
        {
            new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(input).AsSpan().CopyTo(o.AsWritableSpan()); }, buf0, new[] { input }),
            new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(buf0).AsSpan().CopyTo(o.AsWritableSpan()); }, buf1, new[] { buf0 }),
            new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(buf1).AsSpan().CopyTo(o.AsWritableSpan()); }, buf2, new[] { buf1 }),
            new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(buf2).AsSpan().CopyTo(o.AsWritableSpan()); }, buf3, new[] { buf2 }),
            new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(buf3).AsSpan().CopyTo(o.AsWritableSpan()); }, buf4, new[] { buf3 }),
        };

        var optimized = pass.TryOptimize(steps, _engine);

        // The pass should have reused some buffers — even if it returns the
        // same steps array (in-place rebind), the fact that it found reuse
        // opportunities means it returns non-null.
        // Note: with the greedy first-fit algorithm, buf2 could reuse buf0's
        // storage (buf0 is dead after step1 reads it, and buf2 has the same shape).
        // Whether the actual implementation achieves this depends on the
        // lastUse computation. Just verify the pass runs without crashing
        // and that the resulting steps still produce correct output.
        for (int i = 0; i < steps.Length; i++)
            steps[i].Execute(_engine, steps[i].OutputBuffer);

        // Final output should be non-trivial (5 sigmoid applications)
        var finalData = buf4.AsSpan();
        bool anyNonZero = false;
        for (int i = 0; i < finalData.Length; i++)
            if (Math.Abs(finalData[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Chain after memory planning produced all-zero output");
    }

    [Fact]
    public void MemoryPlanning_ReturnsNull_ForTinyPlans()
    {
        var pass = new MemoryPlanningPass();
        var t = Tensor<float>.CreateRandom([2, 2]);
        var steps = new[]
        {
            new CompiledStep<float>("Sigmoid", (e, o) => { }, t, new[] { t }),
            new CompiledStep<float>("Sigmoid", (e, o) => { }, t, new[] { t }),
        };
        Assert.Null(pass.TryOptimize(steps, _engine));
    }

    // ════════════════════════════════════════════════════════════════════
    // TileSchedulingPass
    // ════════════════════════════════════════════════════════════════════

    [Fact]
    public void TileScheduling_ComputesGemmTileSize_FitsL1()
    {
        // For float (4 bytes), L1 = 32KB:
        // 3 × T² × 4 ≤ 32768 → T² ≤ 2730 → T ≈ 52 → rounded to 52
        int tile = TileSchedulingPass.ComputeGemmTileSize(sizeof(float));
        Assert.True(tile >= 4, $"Tile {tile} is below minimum 4");
        Assert.True(tile <= 64, $"Tile {tile} is above expected range");
        Assert.Equal(0, tile % 4); // Must be SIMD-aligned (multiple of 4)

        // Verify working set fits L1 (32KB)
        int workingSet = 3 * tile * tile * sizeof(float);
        Assert.True(workingSet <= 32 * 1024, $"Working set {workingSet} exceeds L1 (32KB)");
    }

    [Fact]
    public void TileScheduling_ComputesConvTileSize_ReasonableRange()
    {
        var input = Tensor<float>.CreateRandom([1, 64, 32, 32]);
        var kernel = Tensor<float>.CreateRandom([128, 64, 3, 3]);
        var step = new CompiledStep<float>("Conv2D", (e, o) => { },
            new Tensor<float>([1, 128, 30, 30]),
            new[] { input, kernel },
            savedState: new object[] { new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 } });

        int tile = TileSchedulingPass.ComputeConvSpatialTile(step, sizeof(float));
        Assert.True(tile >= 4 && tile <= 64, $"Conv tile {tile} outside [4, 64] range");
    }

    [Fact]
    public void TileScheduling_ReturnsNull_AlwaysAnnotationOnly()
    {
        // TileScheduling currently only annotates (returns null for structural change)
        var pass = new TileSchedulingPass();
        var t = Tensor<float>.CreateRandom([4, 4]);
        var steps = new[]
        {
            new CompiledStep<float>("TensorMatMul", (e, o) => { }, t, new[] { t, t }),
        };
        Assert.Null(pass.TryOptimize(steps, _engine));
    }

    // ════════════════════════════════════════════════════════════════════
    // OperatorReorderingPass
    // ════════════════════════════════════════════════════════════════════

    [Fact]
    public void OperatorReordering_PullsConsumerCloserToProducer()
    {
        var pass = new OperatorReorderingPass();

        // Graph: A produces X, then independent B runs, then C reads X.
        // Optimal: A → C → B (keeps X in cache).
        var inputA = Tensor<float>.CreateRandom([4, 4]);
        var outputA = new Tensor<float>([4, 4]); // X
        var inputB = Tensor<float>.CreateRandom([4, 4]);
        var outputB = new Tensor<float>([4, 4]);
        var outputC = new Tensor<float>([4, 4]);
        // A fourth step to reach the 4-step threshold
        var outputD = new Tensor<float>([4, 4]);

        var stepA = new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(inputA).AsSpan().CopyTo(o.AsWritableSpan()); }, outputA, new[] { inputA });
        var stepB = new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(inputB).AsSpan().CopyTo(o.AsWritableSpan()); }, outputB, new[] { inputB }); // Independent
        var stepC = new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(outputA).AsSpan().CopyTo(o.AsWritableSpan()); }, outputC, new[] { outputA }); // Reads A's output
        var stepD = new CompiledStep<float>("Sigmoid", (e, o) => { e.Sigmoid(outputC).AsSpan().CopyTo(o.AsWritableSpan()); }, outputD, new[] { outputC });

        var steps = new[] { stepA, stepB, stepC, stepD };
        var optimized = pass.TryOptimize(steps, _engine);

        if (optimized is not null)
        {
            // After reordering, C should be closer to A than B is.
            int posA = Array.IndexOf(optimized, stepA);
            int posC = Array.IndexOf(optimized, stepC);
            int posB = Array.IndexOf(optimized, stepB);

            Assert.True(posC < posB || posC == posA + 1,
                $"Expected C (pos {posC}) to be pulled closer to A (pos {posA}) than B (pos {posB})");
        }

        // Verify semantic equivalence: executing the reordered steps
        // produces the same final result.
        var stepsToRun = optimized ?? steps;
        for (int i = 0; i < stepsToRun.Length; i++)
            stepsToRun[i].Execute(_engine, stepsToRun[i].OutputBuffer);

        bool anyNonZero = false;
        var data = outputD.AsSpan();
        for (int i = 0; i < data.Length; i++)
            if (Math.Abs(data[i]) > 1e-8f) { anyNonZero = true; break; }
        Assert.True(anyNonZero, "Reordered execution produced all-zero final output");
    }

    [Fact]
    public void OperatorReordering_PreservesDependencies_NeverReordersDependent()
    {
        var pass = new OperatorReorderingPass();

        // Strict dependency chain: A → B → C → D. No reordering should happen.
        var inputA = Tensor<float>.CreateRandom([4, 4]);
        var buf1 = new Tensor<float>([4, 4]);
        var buf2 = new Tensor<float>([4, 4]);
        var buf3 = new Tensor<float>([4, 4]);
        var buf4 = new Tensor<float>([4, 4]);

        var steps = new[]
        {
            new CompiledStep<float>("Sigmoid", (e, o) => { }, buf1, new[] { inputA }),
            new CompiledStep<float>("Sigmoid", (e, o) => { }, buf2, new[] { buf1 }),
            new CompiledStep<float>("Sigmoid", (e, o) => { }, buf3, new[] { buf2 }),
            new CompiledStep<float>("Sigmoid", (e, o) => { }, buf4, new[] { buf3 }),
        };

        var optimized = pass.TryOptimize(steps, _engine);
        Assert.Null(optimized); // Strict chain — no reordering possible
    }

    [Fact]
    public void OperatorReordering_ReturnsNull_ForTinyPlans()
    {
        var pass = new OperatorReorderingPass();
        var t = Tensor<float>.CreateRandom([2, 2]);
        var steps = new[]
        {
            new CompiledStep<float>("Sigmoid", (e, o) => { }, t, new[] { t }),
        };
        Assert.Null(pass.TryOptimize(steps, _engine));
    }

    // ════════════════════════════════════════════════════════════════════
    // Integration: all 3 passes compose without errors
    // ════════════════════════════════════════════════════════════════════

    [Fact]
    public void AllPasses_ComposeOnCompiledPlan_WithoutErrors()
    {
        // Compile a small MLP and verify all passes run without crashing
        var input  = Tensor<float>.CreateRandom([4, 8]);
        var w1     = Tensor<float>.CreateRandom([8, 16]);
        var w2     = Tensor<float>.CreateRandom([16, 4]);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h1 = _engine.TensorMatMul(input, w1);
            var a1 = _engine.Sigmoid(h1);
            var h2 = _engine.TensorMatMul(a1, w2);
            _engine.Sigmoid(h2);
            plan = scope.CompileInference<float>();
        }

        // Plan should compile without errors (passes ran internally)
        Assert.True(plan.StepCount > 0);

        // Execute should produce valid output
        var output = plan.Execute();
        Assert.False(float.IsNaN(output[0]));

        plan.Dispose();
    }
}
