using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Elementwise fusion in compiled training plans: connected chains of registered pointwise ops (add, subtract,
/// multiply, divide, negate, ReLU, tanh, sigmoid) over same-shape float buffers replay as ONE tiled action. The fused
/// plan must produce the same bits as the unfused plan (every member still writes its own buffer, through the same
/// kernel), gradients must match the eager tape, and the range kernels must reproduce a whole-buffer call exactly.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class PointwiseFusionTests
{
    private static Tensor<float> Rnd(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }

    /// <summary>An LSTM cell written from primitives, the way a layer unrolls it: one input projection for all four
    /// gates, reshaped to [batch, 4, hidden] and sliced per gate (TensorSliceAxis), plus a recurrent MatMul per gate;
    /// then i, f, o = sigmoid, g = tanh, c = f*c0 + i*g, h = o*tanh(c) -- elementwise ops and slices that depend on each
    /// other across the four gate chains, so grouping them needs the graph, not a fixed pattern. The loss subtracts
    /// and divides so those kernels are exercised too.</summary>
    private static Tensor<float> Cell(IEngine e, Tensor<float> x, Tensor<float> h0, Tensor<float> c0, Tensor<float>[] p)
    {
        int batch = x._shape[0], hidden = h0._shape[1];
        var projected = e.Reshape(e.TensorMatMul(x, p[0]), new[] { batch, 4, hidden });
        Tensor<float> Pre(int k) => e.TensorAdd(e.TensorSliceAxis(projected, 1, k), e.TensorMatMul(h0, p[1 + k]));
        var i = e.Sigmoid(Pre(0));
        var f = e.Sigmoid(Pre(1));
        var g = e.Tanh(Pre(2));
        var o = e.Sigmoid(Pre(3));
        var c = e.TensorAdd(e.TensorMultiply(f, c0), e.TensorMultiply(i, g));
        var h = e.TensorMultiply(o, e.Tanh(c));
        var r = e.ReLU(e.TensorSubtract(h, e.TensorNegate(c)));
        var q = e.TensorDivide(r, e.TensorAdd(e.TensorMultiply(c, c), TwoPlusSquare(e, c0)));
        return e.ReduceSum(e.TensorMultiply(q, h), null);
    }

    // A strictly positive divisor term that is not a parameter path: 2 + c0*c0 >= 2.
    private static Tensor<float> TwoPlusSquare(IEngine e, Tensor<float> c0)
    {
        var two = new Tensor<float>(c0._shape);
        for (int k = 0; k < two.Length; k++) two[k] = 2f;
        return e.TensorAdd(e.TensorMultiply(c0, c0), two);
    }

    private static Tensor<float>[] Params(int inFeatures, int hidden) => new[]
    {
        Rnd(new[] { inFeatures, 4 * hidden }, 20, 0.3f),
        Rnd(new[] { hidden, hidden }, 21, 0.3f), Rnd(new[] { hidden, hidden }, 22, 0.3f),
        Rnd(new[] { hidden, hidden }, 23, 0.3f), Rnd(new[] { hidden, hidden }, 24, 0.3f),
    };

    private static (float[][] Grads, float Loss, string[] Names) RunPlan(int batch, int hidden, int inFeatures, bool fusion, int steps,
        Action<float[][], int>? perStep = null)
    {
        bool prior = PointwiseKernelRegistry.Enabled;
        PointwiseKernelRegistry.Enabled = fusion;
        try
        {
            var engine = new CpuEngine();
            var x = Rnd(new[] { batch, inFeatures }, 11);
            var h0 = Rnd(new[] { batch, hidden }, 12);
            var c0 = Rnd(new[] { batch, hidden }, 13);
            var ps = Params(inFeatures, hidden);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Cell(engine, x, h0, c0, ps);
                plan = scope.CompileTraining(ps);
            }
            try
            {
                var concrete = Assert.IsType<CompiledTrainingPlan<float>>(plan);
                float[][] grads = Array.Empty<float[]>();
                float loss = 0;
                for (int s = 0; s < steps; s++)
                {
                    loss = plan.Step().AsSpan()[0];
                    grads = plan.Gradients.Select(g => g.AsSpan().ToArray()).ToArray();
                    perStep?.Invoke(grads, s);
                }
                return (grads, loss, concrete.ForwardActionNames.ToArray());
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            PointwiseKernelRegistry.Enabled = prior;
        }
    }

    [Theory]
    [InlineData(37, 40, 9)]    // 1480 elements: one tile with a non-multiple-of-8 tail
    [InlineData(70, 64, 16)]   // 4480 elements: two tiles
    [InlineData(130, 256, 8)]  // 33280 elements: split into chunks over the pool
    public void FusedPlan_IsBitIdenticalToUnfused_AndMatchesTape(int batch, int hidden, int inFeatures)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var unfused = RunPlan(batch, hidden, inFeatures, fusion: false, steps: 1);
            float[][]? first = null;
            var fused = RunPlan(batch, hidden, inFeatures, fusion: true, steps: 4, (g, s) =>
            {
                if (first is null) { first = g; return; }
                for (int p = 0; p < g.Length; p++)
                    for (int i = 0; i < g[p].Length; i++)
                        Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(first[p][i]) == TestHelpers.MathCompat.SingleToInt32Bits(g[p][i]),
                            $"step {s} parameter {p} element {i} drifted from step 0: {first[p][i]:R} -> {g[p][i]:R}");
            });

            Assert.Contains(fused.Names, n => n.StartsWith("fused:pointwise[", StringComparison.Ordinal) && n.Contains("TensorSliceAxis"));
            Assert.DoesNotContain(fused.Names, n => n == "generic:TensorSliceAxis");
            Assert.DoesNotContain(unfused.Names, n => n.StartsWith("fused:pointwise[", StringComparison.Ordinal));
            Assert.True(fused.Names.Length < unfused.Names.Length,
                $"fusion did not reduce the forward action count ({unfused.Names.Length} -> {fused.Names.Length})");

            Assert.Equal(TestHelpers.MathCompat.SingleToInt32Bits(unfused.Loss), TestHelpers.MathCompat.SingleToInt32Bits(fused.Loss));
            for (int p = 0; p < unfused.Grads.Length; p++)
                for (int i = 0; i < unfused.Grads[p].Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(unfused.Grads[p][i]) == TestHelpers.MathCompat.SingleToInt32Bits(fused.Grads[p][i]),
                        $"parameter {p} element {i}: unfused {unfused.Grads[p][i]:R} fused {fused.Grads[p][i]:R}");

            // And both agree with the eager tape.
            var engine = new CpuEngine();
            var x = Rnd(new[] { batch, inFeatures }, 11);
            var h0 = Rnd(new[] { batch, hidden }, 12);
            var c0 = Rnd(new[] { batch, hidden }, 13);
            var ps = Params(inFeatures, hidden);
            using var tape = new GradientTape<float>();
            var loss = Cell(engine, x, h0, c0, ps);
            var tg = tape.ComputeGradients(loss, ps);
            for (int p = 0; p < ps.Length; p++)
            {
                var expected = tg[ps[p]].GetFlattenedData();
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(fused.Grads[p][i] - expected[i]) <= 1e-4f * (1f + Math.Abs(expected[i])),
                        $"parameter {p} element {i}: tape {expected[i]:R} fused plan {fused.Grads[p][i]:R}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Theory]
    [InlineData("TensorAdd", 2)]
    [InlineData("TensorSubtract", 2)]
    [InlineData("TensorMultiply", 2)]
    [InlineData("TensorDivide", 2)]
    [InlineData("TensorNegate", 1)]
    [InlineData("ReLU", 1)]
    [InlineData("Tanh", 1)]
    [InlineData("Sigmoid", 1)]
    public unsafe void RangeKernel_OverAnySplit_MatchesWholeBufferCall(string opName, int arity)
    {
        var op = (OpType)Enum.Parse(typeof(OpType), opName);
        foreach (int length in new[] { 5, 50, 1000, 4103, 9000 })
        {
            var a = Rnd(new[] { length }, 1, 6f).AsSpan().ToArray();
            var b = Rnd(new[] { length }, 2, 6f).AsSpan().ToArray();
            for (int i = 0; i < b.Length; i++) if (Math.Abs(b[i]) < 0.25f) b[i] = 0.25f; // a divisor away from zero
            var whole = new float[length];
            var split = new float[length];
            var shape = new[] { length };
            var kernel = PointwiseKernelRegistry.TryGet(op, shape, arity == 2 ? new[] { shape, shape } : new[] { shape }, null);
            Assert.NotNull(kernel);
            fixed (float* pa = a, pb = b, pw = whole, ps = split)
            {
                kernel!.Kernel(pa, arity == 2 ? pb : null, pw, 0, length);
                // Ranges start on multiples of 64 (the fused group's tile and chunk alignment).
                for (int start = 0; start < length; start += 192)
                    kernel.Kernel(pa, arity == 2 ? pb : null, ps, start, Math.Min(192, length - start));
            }
            for (int i = 0; i < length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(whole[i]) == TestHelpers.MathCompat.SingleToInt32Bits(split[i]),
                    $"{op} length {length} element {i}: whole {whole[i]:R} split {split[i]:R}");
        }
    }

    [Fact]
    public unsafe void SliceAxisRangeKernel_OverAnySplit_MatchesEngineSlice()
    {
        var engine = new CpuEngine();
        var source = Rnd(new[] { 7, 5, 6, 9 }, 3);
        for (int axis = 0; axis < 4; axis++)
            for (int index = 0; index < source._shape[axis]; index += 2)
            {
                var expected = engine.TensorSliceAxis(source, axis, index);
                var kernel = PointwiseKernelRegistry.TryGet(OpTypeParser.Parse("TensorSliceAxis"), expected._shape,
                    new[] { source._shape }, new object[] { axis, index });
                Assert.NotNull(kernel);
                Assert.False(kernel!.Elementwise);
                var src = source.AsSpan().ToArray();
                var dst = new float[expected.Length];
                fixed (float* ps = src, pd = dst)
                    for (int start = 0; start < dst.Length; start += 64)
                        kernel.Kernel(ps, null, pd, start, Math.Min(64, dst.Length - start));
                var want = expected.AsSpan().ToArray();
                for (int i = 0; i < want.Length; i++)
                    Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(want[i]) == TestHelpers.MathCompat.SingleToInt32Bits(dst[i]),
                        $"axis {axis} index {index} element {i}: engine {want[i]:R} range kernel {dst[i]:R}");
            }
        // Shapes or saved state that are not a slice of this input get no kernel.
        Assert.Null(PointwiseKernelRegistry.TryGet(OpTypeParser.Parse("TensorSliceAxis"), new[] { 7, 6, 9 },
            new[] { source._shape }, new object[] { 1, 5 }));
        Assert.Null(PointwiseKernelRegistry.TryGet(OpTypeParser.Parse("TensorSliceAxis"), new[] { 7, 6, 8 },
            new[] { source._shape }, new object[] { 1, 0 }));
    }
}