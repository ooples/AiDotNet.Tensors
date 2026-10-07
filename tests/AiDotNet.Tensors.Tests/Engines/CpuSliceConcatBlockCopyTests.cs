using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// CpuEngine slice and concatenate copy contiguous runs instead of walking element indices, the compiled replay
/// writes a concatenation straight into its output buffer, and a slice backward adds whole rows into its region.
/// Every result here must equal an independent element-by-element reference bit for bit: these paths only move
/// values (or add each element once), so any difference is a wrong offset, not rounding.
/// </summary>
[Collection("CompilationGlobalState")]
public class CpuSliceConcatBlockCopyTests
{
    private static Tensor<float> Iota(params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = i * 0.5f - 7.25f;
        return t;
    }

    private static int[] Strides(int[] shape)
    {
        var s = new int[shape.Length];
        int stride = 1;
        for (int d = shape.Length - 1; d >= 0; d--) { s[d] = stride; stride *= shape[d]; }
        return s;
    }

    private static float[] ReferenceSlice(Tensor<float> x, int[] start, int[] length)
    {
        var data = x.ToArray();
        var strides = Strides(x._shape);
        int total = 1;
        foreach (var l in length) total *= l;
        var result = new float[total];
        for (int flat = 0; flat < total; flat++)
        {
            int remaining = flat, src = 0;
            for (int d = length.Length - 1; d >= 0; d--)
            {
                int idx = remaining % length[d];
                remaining /= length[d];
                src += (start[d] + idx) * strides[d];
            }
            result[flat] = data[src];
        }
        return result;
    }

    private static int Bits(float v) => System.BitConverter.ToInt32(System.BitConverter.GetBytes(v), 0);

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (Bits(expected[i]) != Bits(actual[i]))
                Assert.Fail($"{what}: element {i} expected {expected[i]:R} got {actual[i]:R}");
    }

    public static TheoryData<int[], int[], int[]> SliceBoxes => new()
    {
        { new[] { 10 }, new[] { 3 }, new[] { 5 } },                              // rank 1
        { new[] { 4, 6 }, new[] { 1, 2 }, new[] { 2, 3 } },                      // partial last axis
        { new[] { 4, 6 }, new[] { 1, 0 }, new[] { 3, 6 } },                      // full last axis: one run
        { new[] { 8, 32, 32 }, new[] { 0, 5, 0 }, new[] { 8, 1, 32 } },          // LSTM per-timestep slice
        { new[] { 8, 32, 64 }, new[] { 0, 7, 0 }, new[] { 8, 9, 64 } },          // concat-backward block
        { new[] { 3, 4, 5, 6 }, new[] { 1, 1, 2, 0 }, new[] { 2, 3, 3, 6 } },    // two outer axes, merged tail
        { new[] { 3, 4, 5, 6 }, new[] { 2, 3, 4, 5 }, new[] { 1, 1, 1, 1 } },    // single element
        { new[] { 3, 4, 5, 6 }, new[] { 0, 0, 0, 0 }, new[] { 3, 4, 5, 6 } },    // whole tensor
        { new[] { 600, 4, 700 }, new[] { 10, 1, 3 }, new[] { 500, 2, 600 } },    // large enough to run in parallel
    };

    [Theory]
    [MemberData(nameof(SliceBoxes))]
    public void TensorSlice_MatchesElementWalk(int[] shape, int[] start, int[] length)
    {
        var engine = new CpuEngine();
        var x = Iota(shape);
        var sliced = engine.TensorSlice(x, start, length);
        Assert.Equal(length, sliced._shape);
        AssertBitEqual(ReferenceSlice(x, start, length), sliced.ToArray(), "slice");
    }

    public static TheoryData<int[][], int> ConcatCases => new()
    {
        { new[] { new[] { 4, 1, 3 }, new[] { 4, 1, 3 }, new[] { 4, 1, 3 } }, 1 },   // LSTM time-axis concat
        { new[] { new[] { 2, 3, 5 }, new[] { 2, 4, 5 } }, 1 },                       // ragged along the axis
        { new[] { new[] { 2, 3, 5 }, new[] { 2, 3, 2 } }, 2 },                       // last axis
        { new[] { new[] { 2, 3, 5 }, new[] { 2, 3, 2 } }, -1 },                      // negative axis
        { new[] { new[] { 2, 3, 4, 2 }, new[] { 2, 1, 4, 2 }, new[] { 2, 2, 4, 2 } }, 1 },
        { new[] { new[] { 3, 2 }, new[] { 1, 2 } }, 0 },                             // first axis
    };

    [Theory]
    [MemberData(nameof(ConcatCases))]
    public void TensorConcatenate_MatchesReferenceWalk(int[][] shapes, int axis)
    {
        var engine = new CpuEngine();
        var inputs = new Tensor<float>[shapes.Length];
        for (int i = 0; i < shapes.Length; i++)
        {
            inputs[i] = Iota(shapes[i]);
            var span = inputs[i].AsWritableSpan();
            for (int j = 0; j < span.Length; j++) span[j] += 100f * i;   // distinct values per input
        }
        int normalized = axis < 0 ? shapes[0].Length + axis : axis;
        var expected = Tensor<float>.Concatenate(inputs, normalized);   // independent element-wise walk
        var actual = engine.TensorConcatenate(inputs, axis);
        Assert.Equal(expected._shape, actual._shape);
        AssertBitEqual(expected.ToArray(), actual.ToArray(), "concat");

        var output = new Tensor<float>(expected._shape);
        Assert.True(CpuEngine.TryConcatenateInto(inputs, axis, output));
        AssertBitEqual(expected.ToArray(), output.ToArray(), "concat into");
    }

    [Fact]
    public void TensorConcatenate_MismatchedNonAxisExtent_Throws()
    {
        var engine = new CpuEngine();
        var a = Iota(2, 3, 4);
        var b = Iota(2, 3, 5);
        Assert.Throws<System.ArgumentException>(() => engine.TensorConcatenate(new[] { a, b }, 1));
        Assert.False(CpuEngine.TryConcatenateInto(new[] { a, b }, 1, new Tensor<float>(new[] { 2, 6, 4 })));
    }

    [Fact]
    public void TryConcatenateInto_WrongOutputShape_ReturnsFalseAndLeavesOutput()
    {
        var output = new Tensor<float>(new[] { 2, 5, 4 });
        output.Fill(9f);
        Assert.False(CpuEngine.TryConcatenateInto(new[] { Iota(2, 3, 4), Iota(2, 3, 4) }, 1, output));
        foreach (var v in output.ToArray()) Assert.Equal(9f, v);
    }

    /// <summary>The per-timestep LSTM pattern: slice every step of a sequence along axis 1, transform it, and
    /// concatenate the steps back along axis 1. Compiled replay (direct concat write) and the backward (region add
    /// of each slice gradient, slice-run copies of the concat gradient) must match the eager tape, every step.</summary>
    [Fact]
    public void SliceAxisThenConcat_CompiledMatchesEagerTape()
    {
        const int B = 3, T = 5, F = 4;
        var engine = new CpuEngine();
        var x = Iota(B, T, F);
        var w = Iota(F, F);
        var wSpan = w.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = 0.05f * ((i * 7) % 11 - 5);

        Tensor<float> Forward(Tensor<float> input, Tensor<float> weight)
        {
            var steps = new Tensor<float>[T];
            for (int t = 0; t < T; t++)
            {
                // Two slices per step, so every timestep region of dX receives two contributions (a region
                // add, not just a copy into a cleared buffer).
                var xt = engine.TensorAdd(engine.TensorSliceAxis(input, 1, t),
                    engine.TensorSliceAxis(input, 1, (t + 1) % T));                     // [B, F]
                var h = engine.Tanh(engine.TensorMatMul(xt, weight));               // [B, F]
                steps[t] = engine.Reshape(h, new[] { B, 1, F });
            }
            var seq = engine.TensorConcatenate(steps, 1);                           // [B, T, F]
            return engine.ReduceSum(engine.TensorMultiply(seq, seq), null);
        }

        Tensor<float> eagerGradW, eagerGradX;
        float eagerLoss;
        using (var tape = new GradientTape<float>())
        {
            var loss = Forward(x, w);
            eagerLoss = loss[0];
            var grads = tape.ComputeGradients(loss, sources: new[] { w, x });
            eagerGradW = grads[w];
            eagerGradX = grads[x];
        }

        var wC = new Tensor<float>(w._shape);
        w.AsSpan().CopyTo(wC.AsWritableSpan());
        var xC = new Tensor<float>(x._shape);
        x.AsSpan().CopyTo(xC.AsWritableSpan());
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Forward(xC, wC);
            plan = scope.CompileTraining(new[] { wC, xC });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 3; step++)
            {
                var loss = plan.Step();
                Assert.True(System.Math.Abs(loss[0] - eagerLoss) <= 1e-5f * (1f + System.Math.Abs(eagerLoss)),
                    $"step {step}: compiled loss {loss[0]:R} vs eager {eagerLoss:R}");
                AssertClose(eagerGradW, wC.Grad, $"dW step {step}");
                AssertClose(eagerGradX, xC.Grad, $"dX step {step}");
            }
        }
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float>? actual, string what)
    {
        Assert.True(actual is not null, $"{what}: gradient is null");
        if (actual is null) return;
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (System.Math.Abs(e - a) > 1e-5f * (1f + System.Math.Abs(e)))
                Assert.Fail($"{what}: element {i} eager={e:R} compiled={a:R}");
        }
    }
}