// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// TensorGatherClassValues is the class-index core of cross-entropy: out[r] = values[r, class[r]], 0 for an ignored
/// class. A loss that one-hot encodes class targets on the host at trace time trains a compiled plan on the FIRST
/// batch's targets forever; this op reads the targets live on every replay and on the device.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ClassGatherTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ClassGatherTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    // Rows: in range, last class, -1 sentinel, == C, non-integer (rounds to 1), NaN.
    private static readonly float[] Classes = { 0f, 3f, -1f, 4f, 1.2f, float.NaN };

    private static Tensor<float> Values()
    {
        var v = new Tensor<float>(new[] { 2, 3, 4 });
        for (int i = 0; i < v.Length; i++) v[i] = (i % 7) * 0.25f - 0.6f;
        return v;
    }

    private static Tensor<float> ClassTensor(float[] classes)
    {
        var t = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < classes.Length; i++) t[i] = classes[i];
        return t;
    }

    private static (float[] value, float[] grad) Run(IEngine e, Tensor<float> values, Tensor<float> classes, out long readback, out string sites)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = e;
        try
        {
            using var tape = new GradientTape<float>();
            var input = e.TensorTanh(values);                   // device-resident on the GPU engine
            var weights = new Tensor<float>(new[] { 2, 3 });
            for (int i = 0; i < weights.Length; i++) weights[i] = i + 1;
            GpuLaunchProbe.Reset();
            var picked = e.TensorGatherClassValues(input, classes);
            var loss = e.ReduceSum(e.TensorMultiply(picked, weights), new[] { 0, 1 }, keepDims: false);
            var grad = tape.ComputeGradients(loss, new[] { values })[values];
            readback = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
            Assert.Equal(new[] { 2, 3 }, picked.Shape.ToArray());
            return (picked.ToArray(), grad.ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [Fact]
    public void Cpu_GathersTheClass_AndIgnoredRowsGetZeroValueAndGradient()
    {
        var values = Values();
        var (picked, grad) = Run(new CpuEngine(), values, ClassTensor(Classes), out _, out _);
        int[] expectedClass = { 0, 3, -1, -1, 1, -1 };
        for (int r = 0; r < 6; r++)
        {
            float expected = expectedClass[r] < 0 ? 0f : (float)Math.Tanh(values[r * 4 + expectedClass[r]]);
            Assert.True(Math.Abs(picked[r] - expected) < 1e-6f, $"row {r}: {picked[r]} vs {expected}");
            for (int c = 0; c < 4; c++)
            {
                float t = (float)Math.Tanh(values[r * 4 + c]);
                float expectedGrad = c == expectedClass[r] ? (r + 1) * (1 - t * t) : 0f;
                Assert.True(Math.Abs(grad[r * 4 + c] - expectedGrad) < 1e-5f, $"d/dv[{r},{c}] {grad[r * 4 + c]} vs {expectedGrad}");
            }
        }
    }

    [Fact]
    public void Cpu_RejectsAClassCountThatDoesNotMatchTheRows()
    {
        var e = new CpuEngine();
        Assert.Throws<ArgumentException>(() => e.TensorGatherClassValues(Values(), new Tensor<float>(new[] { 5 })));
    }

    [SkippableFact]
    public void Gpu_MatchesCpu_WithoutReadingTheValuesBack()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var values = Values();
        var (cpuPicked, cpuGrad) = Run(new CpuEngine(), values, ClassTensor(Classes), out _, out _);
        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        float[] gpuPicked, gpuGrad;
        long readback;
        string sites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            (gpuPicked, gpuGrad) = Run(_fixture.Engine!, values, ClassTensor(Classes), out readback, out sites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readback <= 64, $"the class gather forward+backward read back {readback} bytes: {sites}");
        for (int i = 0; i < cpuPicked.Length; i++)
            Assert.True(Math.Abs(cpuPicked[i] - gpuPicked[i]) < 1e-6f, $"picked[{i}] cpu {cpuPicked[i]} gpu {gpuPicked[i]}");
        for (int i = 0; i < cpuGrad.Length; i++)
            Assert.True(Math.Abs(cpuGrad[i] - gpuGrad[i]) < 1e-5f, $"d/dv[{i}] cpu {cpuGrad[i]} gpu {gpuGrad[i]}");
    }

    [Fact]
    public void CompiledPlan_ReadsTheCurrentClasses_ForwardAndBackward_Cpu() => CompiledPlanReadsLiveClasses(new CpuEngine());

    [SkippableFact]
    public void CompiledPlan_ReadsTheCurrentClasses_ForwardAndBackward_Gpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        CompiledPlanReadsLiveClasses(_fixture.Engine!);
    }

    /// <summary>
    /// loss = sum_r (x·W)[r, class[r]] with SGD at lr 1: W' = W - xᵀ·onehot(class). The classes change IN PLACE
    /// between steps; step 2's loss and W update must use the new classes (a trace-time snapshot uses the old ones).
    /// </summary>
    private static void CompiledPlanReadsLiveClasses(IEngine engine)
    {
        const int n = 3, dIn = 2, c = 4;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var x = new Tensor<float>(new[] { n, dIn });
            for (int i = 0; i < x.Length; i++) x[i] = 0.5f + i;
            var w = new Tensor<float>(new[] { dIn, c });
            for (int i = 0; i < w.Length; i++) w[i] = 0.1f * (i - 3);
            var classes = new Tensor<float>(new[] { n });
            float[] first = { 0, 2, 3 }, second = { 1, -1, 0 };
            for (int i = 0; i < n; i++) classes[i] = first[i];
            var w0 = w.ToArray();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var logits = engine.TensorMatMul(x, w);
                engine.ReduceSum(engine.TensorGatherClassValues(logits, classes), null);
                plan = scope.CompileTraining(new[] { w });
            }

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1.0f);
                float loss1 = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(loss1 - Loss(x, w0, first)) < 1e-4f, $"step 1 loss {loss1} vs {Loss(x, w0, first)}");
                var w1 = SgdStep(x, w0, first);

                for (int i = 0; i < n; i++) classes[i] = second[i];   // the next batch's targets, in place
                float loss2 = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(loss2 - Loss(x, w1, second)) < 1e-4f,
                    $"step 2 loss {loss2}: expected {Loss(x, w1, second)} for the current classes, {Loss(x, w1, first)} would mean frozen targets");
                var w2 = SgdStep(x, w1, second);
                var got = w.ToArray();
                for (int i = 0; i < got.Length; i++)
                    Assert.True(Math.Abs(got[i] - w2[i]) < 1e-4f, $"W[{i}] after step 2: {got[i]} vs {w2[i]} (backward scattered against stale classes?)");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }

        float Loss(Tensor<float> xs, float[] ws, float[] cls)
        {
            float total = 0;
            for (int r = 0; r < n; r++)
            {
                int k = (int)cls[r];
                if (k < 0) continue;
                for (int j = 0; j < dIn; j++) total += xs[r * dIn + j] * ws[j * c + k];
            }
            return total;
        }

        float[] SgdStep(Tensor<float> xs, float[] ws, float[] cls)
        {
            var next = (float[])ws.Clone();
            for (int r = 0; r < n; r++)
            {
                int k = (int)cls[r];
                if (k < 0) continue;
                for (int j = 0; j < dIn; j++) next[j * c + k] -= xs[r * dIn + j];
            }
            return next;
        }
    }
}
