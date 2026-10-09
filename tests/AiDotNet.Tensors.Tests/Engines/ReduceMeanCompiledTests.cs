using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The compiled float single-axis ReduceMean node: forward means and the broadcast dY / axisSize gradient, on every
/// step of a compiled plan, against a double-precision transcription.
/// </summary>
public class ReduceMeanCompiledTests
{
    public static TheoryData<int[], int, bool> Cases => new()
    {
        { new[] { 64, 32, 64 }, 1, false },   // sequence pooling
        { new[] { 3, 5, 7 }, 0, true },       // first axis, keepDims
        { new[] { 3, 5, 7 }, 2, false },      // last axis (inner = 1)
        { new[] { 4, 6, 9 }, -2, true },      // negative axis
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task CompiledPlan_MatchesReference_OnEveryStep(int[] shape, int axis, bool keepDims)
    {
        await Task.Yield();
        var rng = new Random(3);
        int n = 1; foreach (var d in shape) n *= d;
        var xa = new float[n];
        for (int i = 0; i < n; i++) xa[i] = (float)(rng.NextDouble() * 2 - 1);
        var x = new Tensor<float>(xa, shape);
        int ax = axis < 0 ? shape.Length + axis : axis;
        int outer = 1, inner = 1, size = shape[ax];
        for (int d = 0; d < ax; d++) outer *= shape[d];
        for (int d = ax + 1; d < shape.Length; d++) inner *= shape[d];
        var ga = new float[outer * inner];
        for (int i = 0; i < ga.Length; i++) ga[i] = (float)(rng.NextDouble() * 2 - 1);

        var expectedY = new double[outer * inner];
        var expectedDx = new double[n];
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++)
            {
                double s = 0;
                for (int a = 0; a < size; a++) s += xa[(o * size + a) * inner + i];
                expectedY[o * inner + i] = s / size;
                for (int a = 0; a < size; a++) expectedDx[(o * size + a) * inner + i] = ga[o * inner + i] / (double)size;
            }

        var engine = new CpuEngine();
        ICompiledTrainingPlan<float> plan;
        Tensor<float> y;
        using (var scope = GraphMode.Enable())
        {
            y = engine.ReduceMean(x, new[] { axis }, keepDims);
            var g = new Tensor<float>(ga, y.Shape.ToArray());
            engine.ReduceSum(engine.TensorMultiply(y, g), null);
            plan = scope.CompileTraining(new[] { x });
        }
        using (plan)
        {
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                AssertClose(expectedY, y.ToArray(), $"y step {step}");
                AssertClose(expectedDx, plan.Gradients[0].ToArray(), $"dX step {step}");
            }
        }
    }

    private static void AssertClose(double[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0;
        for (int i = 0; i < expected.Length; i++) maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]));
        Assert.True(maxErr <= 1e-5, $"{what}: max |error| {maxErr:G4}");
    }
}
