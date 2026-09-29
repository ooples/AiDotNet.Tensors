using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The last host crossings of a resident training step were reductions: ReduceMean under a tape deferred to the CPU
/// (downloading its input), sums over leading and trailing axes (a conv bias gradient) permuted with per-call uploaded
/// tables, and every backward seed was a host tensor. Each now stays on the device; these check the values, shapes and
/// gradients against the CPU engine, and that a warm reduction under a tape moves nothing across the boundary.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuReductionResidencyTests
{
    private static readonly int[] Shape = { 3, 4, 5, 6 };
    private const float Tolerance = 1e-5f;
    private const int WarmupSteps = 2;

    public enum Reduction { Mean, Sum }

    public static TheoryData<Reduction, int[], bool> Cases()
    {
        var data = new TheoryData<Reduction, int[], bool>();
        var axisSets = new[]
        {
            new[] { 0, 1, 2, 3 }, // full
            new[] { 2, 3 },       // trailing block
            new[] { 0, 2, 3 },    // leading and trailing: a conv bias gradient
            new[] { 0 },          // leading only: a linear bias gradient
            new[] { 1 },          // a middle axis: the general permute path
        };
        foreach (var op in new[] { Reduction.Mean, Reduction.Sum })
            foreach (var axes in axisSets)
                foreach (bool keepDims in new[] { false, true })
                    data.Add(op, axes, keepDims);
        return data;
    }

    private static Tensor<float> Input()
    {
        var rng = new Random(31);
        var data = new float[Shape.Aggregate(1, (a, d) => a * d)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        return new Tensor<float>(data, Shape);
    }

    private static Tensor<float> Reduce(IEngine engine, Reduction op, Tensor<float> x, int[] axes, bool keepDims)
        => op == Reduction.Mean ? engine.ReduceMean(x, axes, keepDims) : engine.ReduceSum(x, axes, keepDims);

    private static (float[] Value, int[] OutShape, float[] Grad) Run(IEngine engine, Tensor<float> x, Reduction op, int[] axes, bool keepDims)
    {
        using var tape = new GradientTape<float>();
        var y = Reduce(engine, op, x, axes, keepDims);
        // A non-uniform weight on the output, so the gradient checks each output's broadcast back.
        var weights = new Tensor<float>(Enumerable.Range(1, y.Length).Select(i => (float)i).ToArray(), y.Shape.ToArray());
        var loss = engine.ReduceSum(engine.TensorMultiply(y, weights), null);
        var grads = tape.ComputeGradients(loss, new[] { x });
        return (y.ToArray(), y.Shape.ToArray(), grads[x].ToArray());
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void DeviceReduction_MatchesTheCpuEngine_AndStaysOnTheDevice(Reduction op, int[] axes, bool keepDims)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        var cpu = Run(new CpuEngine(), Input(), op, axes, keepDims);
        var x = gpu.UploadToGpu(Input(), GpuTensorRole.General);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            for (int i = 0; i < WarmupSteps; i++) Run(gpu, x, op, axes, keepDims);

            // The forward, loss and backward are measured; the output weights are uploaded before, and the results are
            // read back for the comparison after.
            int outLength = Reduce(gpu, op, x, axes, keepDims).Length;
            var outShape = Reduce(gpu, op, x, axes, keepDims).Shape.ToArray();
            var weights = gpu.UploadToGpu(
                new Tensor<float>(Enumerable.Range(1, outLength).Select(i => (float)i).ToArray(), outShape), GpuTensorRole.General);
            GpuResidencyScope scope;
            using (scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                using var tape = new GradientTape<float>();
                var y = Reduce(gpu, op, x, axes, keepDims);
                var loss = gpu.ReduceSum(gpu.TensorMultiply(y, weights), null);
                tape.ComputeGradients(loss, new[] { x });
            }

            bool middleAxis = axes.SequenceEqual(new[] { 1 });
            if (!middleAxis)
                Assert.True(scope.Uploads + scope.Downloads == 0,
                    $"{op} over [{string.Join(",", axes)}] crossed the boundary: " +
                    string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}")));

            var gpuRun = Run(gpu, x, op, axes, keepDims);
            Assert.Equal(cpu.OutShape, gpuRun.OutShape);
            for (int i = 0; i < cpu.Value.Length; i++)
                Assert.True(Math.Abs(cpu.Value[i] - gpuRun.Value[i]) <= Tolerance, $"value[{i}]: GPU {gpuRun.Value[i]}, CPU {cpu.Value[i]}");
            for (int i = 0; i < cpu.Grad.Length; i++)
                Assert.True(Math.Abs(cpu.Grad[i] - gpuRun.Grad[i]) <= Tolerance * Math.Max(1f, Math.Abs(cpu.Grad[i])),
                    $"grad[{i}]: GPU {gpuRun.Grad[i]}, CPU {cpu.Grad[i]}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
