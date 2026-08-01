using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Records PER-STEP timing for a compiled training plan to characterise an intermittent stall.
/// </summary>
/// <remarks>
/// <para>
/// MLP_Training_CompiledVsEager reports a mean over 200 steps. Across five runs the compiled path
/// measured 0.799, 3.730, 0.731, 0.733 and 0.806 ms/step against a stable ~0.97 ms eager — so it is
/// normally 1.15-1.37x FASTER, but one run in five blew up to 4.7x its own median and dragged the
/// mean below parity. A mean cannot distinguish "one catastrophic step" from "a sustained slow
/// phase", and those have completely different causes, so this records every step individually.
/// </para>
/// <para>
/// Reports only — the point is to characterise the stall, not to add another red test.
/// </para>
/// </remarks>
public class CompiledStepStallProbe
{
    private readonly ITestOutputHelper _out;
    public CompiledStepStallProbe(ITestOutputHelper output) => _out = output;

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int len = shape.Aggregate(1, (a, b) => a * b);
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    [Fact]
    [Trait("Category", "Performance")]
    public void CompiledTrainingStep_PerStepDistribution()
    {
        var engine = new CpuEngine();
        int batchSize = 32, inputDim = 128, hiddenDim = 64, outputDim = 10;
        const int warmup = 20, measure = 200;
        float lr = 0.01f;

        var input = CreateRandom([batchSize, inputDim], 1);
        var target = CreateRandom([batchSize, outputDim], 2);
        var w1 = CreateRandom([inputDim, hiddenDim], 3);
        var w2 = CreateRandom([hiddenDim, outputDim], 4);

        using var scope = GraphMode.Enable();
        var h = engine.ReLU(engine.TensorMatMul(input, w1));
        var pred = engine.TensorMatMul(h, w2);
        var diff = engine.TensorSubtract(pred, target);
        _ = engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
        var plan = scope.CompileTraining(new[] { w1, w2 });

        void OneStep()
        {
            plan.Step();
            var grads = plan.Gradients;
            if (grads[0] is not null)
                engine.TensorSubtractInPlace(w1, engine.TensorMultiplyScalar(grads[0], lr));
            if (grads[1] is not null)
                engine.TensorSubtractInPlace(w2, engine.TensorMultiplyScalar(grads[1], lr));
        }

        for (int i = 0; i < warmup; i++) OneStep();

        var us = new double[measure];
        var gen0 = new int[measure];
        var gen1 = new int[measure];
        var gen2 = new int[measure];

        for (int i = 0; i < measure; i++)
        {
            int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
            var sw = Stopwatch.StartNew();
            OneStep();
            sw.Stop();
            us[i] = sw.Elapsed.TotalMilliseconds * 1000.0;
            gen0[i] = GC.CollectionCount(0) - g0;
            gen1[i] = GC.CollectionCount(1) - g1;
            gen2[i] = GC.CollectionCount(2) - g2;
        }

        var sorted = (double[])us.Clone();
        Array.Sort(sorted);
        double median = sorted[measure / 2];
        double mean = us.Average();

        _out.WriteLine($"compiled step, {measure} steps after {warmup} warmup (microseconds)");
        _out.WriteLine($"  min={sorted[0]:F1}  p50={median:F1}  p90={sorted[(int)(measure * 0.90)]:F1}  " +
                       $"p99={sorted[(int)(measure * 0.99)]:F1}  max={sorted[measure - 1]:F1}");
        _out.WriteLine($"  mean={mean:F1}   mean/median={mean / median:F2}x");

        // Which steps are outliers, and did a GC land on them?
        var slow = Enumerable.Range(0, measure).Where(i => us[i] > 3 * median).ToArray();
        _out.WriteLine($"  steps over 3x median: {slow.Length}");
        foreach (int i in slow.Take(15))
            _out.WriteLine($"    step {i,3}: {us[i],9:F1} us  ({us[i] / median:F1}x median)  gc gen0/1/2 = {gen0[i]}/{gen1[i]}/{gen2[i]}");

        // Is the excess concentrated in a few steps, or spread across many?
        double excess = us.Where(v => v > 3 * median).Sum() - slow.Length * median;
        _out.WriteLine($"  excess time in those steps: {excess:F0} us of {us.Sum():F0} us total ({100 * excess / us.Sum():F1}%)");

        // Sustained-phase check: compare the first and last quarters.
        double firstQuarter = us.Take(measure / 4).Average();
        double lastQuarter = us.Skip(3 * measure / 4).Average();
        _out.WriteLine($"  first quarter mean={firstQuarter:F1} us, last quarter mean={lastQuarter:F1} us");

        // Allocation per step drives how OFTEN a Gen0 lands in the window. Compare both paths:
        // whichever allocates more collects more often, and the pause has to be paid by whoever
        // is being timed when it fires.
#if NET5_0_OR_GREATER
        // Eager cannot be measured here: the enclosing GraphMode scope is still open, so a tape
        // step would record into the graph instead. Eager numbers come from the real benchmark.
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) OneStep();
        long compiledPerStep = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        _out.WriteLine($"  alloc/step whole OneStep      = {compiledPerStep:N0} B");

        // Split: plan.Step() versus the optimizer tail that lives OUTSIDE the plan.
        before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) plan.Step();
        long stepOnly = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        _out.WriteLine($"  alloc/step plan.Step() only   = {stepOnly:N0} B");

        before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++)
        {
            var g = plan.Gradients;
            if (g[0] is not null) engine.TensorSubtractInPlace(w1, engine.TensorMultiplyScalar(g[0], lr));
            if (g[1] is not null) engine.TensorSubtractInPlace(w2, engine.TensorMultiplyScalar(g[1], lr));
        }
        long optOnly = (GC.GetAllocatedBytesForCurrentThread() - before) / 50;
        _out.WriteLine($"  alloc/step optimizer tail     = {optOnly:N0} B");

        // For scale: the two parameter tensors the optimizer tail materializes each step.
        _out.WriteLine($"  (w1 {w1.Length * 4:N0} B + w2 {w2.Length * 4:N0} B = {(w1.Length + w2.Length) * 4:N0} B of that is the " +
                       "TensorMultiplyScalar temporaries)");

        // Attribute plan.Step()'s allocation per op, using the plan's own StepProbe hook, which
        // fires at every forward and backward boundary. Warm first so JIT of the probe path
        // itself is not counted.
        var marks = new System.Collections.Generic.List<(string Label, long Bytes)>();
        long last = 0;
        CompiledTrainingPlan<float>.StepProbe = label =>
        {
            long now = GC.GetAllocatedBytesForCurrentThread();
            if (last != 0) marks.Add((label, now - last));
            last = now;
        };
        try
        {
            last = GC.GetAllocatedBytesForCurrentThread();
            plan.Step();                       // warm the probe path
            marks.Clear();
            last = GC.GetAllocatedBytesForCurrentThread();
            plan.Step();                       // the measured step
        }
        finally { CompiledTrainingPlan<float>.StepProbe = null; }

        // The specialized (allocation-free) MatMul backward is gated on native BLAS. If that is
        // false here, every matmul backward falls to the generic dictionary path that rents.
        _out.WriteLine($"  BlasProvider.IsAvailable = {AiDotNet.Tensors.Helpers.BlasProvider.IsAvailable}");

        var bwdNames = CompiledTrainingPlan<float>.ProfBackwardStepNames;
        _out.WriteLine($"  --- backward step names ({bwdNames.Length}) ---");
        for (int i = 0; i < bwdNames.Length; i++) _out.WriteLine($"    BWD-{i}: {bwdNames[i]}");

        _out.WriteLine($"  --- per-op allocation inside plan.Step() ({marks.Count} boundaries) ---");
        foreach (var (label, bytes) in marks.Where(m => m.Bytes > 512).OrderByDescending(m => m.Bytes).Take(20))
            _out.WriteLine($"    {bytes,10:N0} B  {label}");
        _out.WriteLine($"    {marks.Sum(m => m.Bytes),10:N0} B  TOTAL across boundaries");
#endif
    }
}
