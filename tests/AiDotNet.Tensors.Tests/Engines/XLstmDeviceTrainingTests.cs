using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("CompilationGlobalState")]
public class XLstmDeviceTrainingTests
{
    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void OpenCl_NarrowEveryAxisAndItsGradientStayOnDevice(int axis)
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var x = new Tensor<float>(new[] { 2, 3, 4 });
        for (int i = 0; i < x.Length; i++) x[i] = i;
        using var tape = new GradientTape<float>();
        GpuLaunchProbe.Reset();
        var resident = engine.TensorAddScalar(x, 1f);
        var slice = engine.TensorNarrow(resident, axis, 1, 1);
        var squeezed = engine.TensorSqueeze(slice, axis);
        var result = engine.TensorAdd(squeezed, squeezed);
        var gradients = tape.ComputeGradients(engine.ReduceSum(result, null), new[] { x });
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        int stride = axis == 0 ? 12 : axis == 1 ? 4 : 1;
        for (int i = 0; i < x.Length; i++) Assert.Equal((i / stride) % x.Shape[axis] == 1 ? 2f : 0f, gradients[x][i]);
    }
    [Fact]
    public void OpenCl_MaximumPreservesSavedInputsAndStridedArithmeticResidency()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var a = new Tensor<float>(new[] { 2, 3 }); var b = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < a.Length; i++) { a[i] = i - 2; b[i] = .5f; }
        using var tape = new GradientTape<float>();
        GpuLaunchProbe.Reset();
        var maximum = engine.TensorMax(a, b);
        var transposed = engine.TensorTranspose(maximum);
        var sum = engine.TensorAdd(transposed, transposed);
        var loss = engine.ReduceSum(sum, null);
        a[0] = 9; // Saved max mask must still reflect the pre-mutation value.
        var gradients = tape.ComputeGradients(loss, new[] { a, b });
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal(i - 2 > .5f ? 2 : 0, gradients[a][i]);
            Assert.Equal(i - 2 > .5f ? 0 : 2, gradients[b][i]);
        }
    }
    private static Tensor<float>[] Inputs(int width, float scale)
    {
        return Enumerable.Range(0, 6).Select(j =>
        {
            var x = new Tensor<float>(new[] { 2, 4, j < 3 ? width * 2 : 2 });
            for (int i = 0; i < x.Length; i++) x[i] = j < 3 ? scale * MathF.Sin(i * .7f + j) : .6f + .3f * MathF.Sin(i + j);
            return x;
        }).ToArray();
    }
    private static Tensor<float> Loss(CpuEngine e, Tensor<float>[] x)
    {
        var y = e.XLstmScanForward(x[0], x[1], x[2], x[3], x[4], x[5], 2);
        return e.ReduceSum(e.TensorMultiply(y, y), null);
    }
    private static void Near(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        for (int i = 0; i < actual.Length; i++)
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 4e-4f * (1 + Math.Abs(expected[i])));
    }
    [Theory]
    [InlineData(3, .2f)] [InlineData(3, 2f)] [InlineData(65, .2f)]
    public void OpenCl_ForwardAndAllSixGradientsStayOnDevice(int width, float scale)
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var x = Inputs(width, scale);
        Tensor<float> expected; Dictionary<Tensor<float>, Tensor<float>> reference;
        using (var tape = new GradientTape<float>())
        { expected = Loss(new CpuEngine(), x); reference = tape.ComputeGradients(expected, x); }
        GpuLaunchProbe.Reset();
        Tensor<float> actual; Dictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new GradientTape<float>())
        { actual = Loss(engine, x); gradients = tape.ComputeGradients(actual, x); }
        Assert.True(GpuLaunchProbe.Count >= 3);
        Assert.Equal(0, GpuLaunchProbe.Readbacks);
        Near(expected, actual);
        foreach (var input in x) Near(reference[input], gradients[input]);
    }
    [Fact]
    public void OpenCl_CompiledTrainingRefreshesInputs()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable, backend.InitializationException?.ToString());
        using var direct = new DirectGpuEngine(backend); using var engine = new DirectGpuTensorEngine(direct);
        var x = Inputs(3, 2);
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable()) { Loss(engine, x); plan = scope.CompileTraining(x); }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0);
            for (int step = 0; step < 3; step++)
            {
                x[0][0] = .4f + step;
                Tensor<float> expected; Dictionary<Tensor<float>, Tensor<float>> reference;
                using (var tape = new GradientTape<float>())
                { expected = Loss(new CpuEngine(), x); reference = tape.ComputeGradients(expected, x); }
                var actual = plan.Step();
                Near(expected, actual);
                foreach (var input in x) Near(reference[input], input.Grad!);
            }
        }
    }
}
