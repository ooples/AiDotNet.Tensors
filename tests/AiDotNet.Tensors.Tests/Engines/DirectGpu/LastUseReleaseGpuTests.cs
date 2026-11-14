using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On a tape that opts in (GradientTapeOptions.ReleaseDeviceActivationsAtLastUse), device intermediates are freed at
/// their last backward use (BackwardStorageReleasePlan), a graph's first backward run probes which of them code reads
/// afterwards (AutoRetainProbe), and a read the probe could not see throws an error that names the producing op and is
/// itself recorded. A default tape never invalidates a tensor its caller holds.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class LastUseReleaseGpuTests
{
    // Odd shapes: the probe's keys are structural, so they must not collide with another test's graph.
    private static readonly int[] Shape = { 3, 7, 5 };

    private static Tensor<float> Input(int seed) =>
        new(Enumerable.Range(0, 105).Select(i => (float)Math.Sin(i * 0.37 + seed)).ToArray(), Shape);

    private static float[] Sigmoid(float[] x) => x.Select(v => 1f / (1f + MathF.Exp(-v))).ToArray();

    private static readonly GradientTapeOptions OptIn = new() { ReleaseDeviceActivationsAtLastUse = true };

    // y = sigmoid(x); loss = sum(y * w): y is an intermediate whose last backward use is the multiply's backward.
    private static Tensor<float> Step(IEngine engine, Tensor<float> x, Tensor<float> w, bool readAfterBackward,
        out float[]? read, bool optIn = true)
    {
        using var tape = new GradientTape<float>(optIn ? OptIn : null);
        var y = engine.Sigmoid(x);
        var loss = engine.ReduceSum(engine.TensorMultiply(y, w), null);
        tape.ComputeGradients(loss, new[] { x });
        read = readAfterBackward ? y.ToArray() : null;
        return y;
    }

    [SkippableFact]
    public void AnIntermediateReadAfterBackward_OnTheGraphsFirstRun_IsKeptOnEveryLaterRun()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(1);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(2), GpuTensorRole.General);
        var expected = Sigmoid(host.ToArray());

        // The first run is the probe: y is quarantined, not freed, so the read succeeds and marks y's producer.
        for (int run = 0; run < 3; run++)
        {
            Step(gpu, x, w, readAfterBackward: true, out var read);
            Assert.Equal(expected, read!, new FloatTolerance(1e-5f));
        }
        Assert.True(AutoRetainProbe.RetainedCount >= 1, "the probe did not retain the intermediate it saw read");
    }

    [SkippableFact]
    public void AnIntermediateNothingReads_IsFreedAfterTheProbe_AndALateReadNamesItsProducer()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(3);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(4), GpuTensorRole.General);

        Step(gpu, x, w, readAfterBackward: false, out _);   // probe run: nothing is read
        var y = Step(gpu, x, w, readAfterBackward: false, out _);   // freed at its last backward use

        var error = Assert.Throws<InvalidOperationException>(() => y.ToArray());
        Assert.Contains("[3, 7, 5]", error.Message);
        Assert.Contains("GradientTape.Retain", error.Message);
        Assert.Contains("forward op", error.Message);

        // The failed read was recorded: the next run keeps y.
        Step(gpu, x, w, readAfterBackward: true, out var read);
        Assert.Equal(Sigmoid(host.ToArray()), read!, new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void Retain_KeepsAnIntermediate_WithoutAProbe()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(5);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(6), GpuTensorRole.General);
        Step(gpu, x, w, readAfterBackward: false, out _);   // spend the probe

        using var tape = new GradientTape<float>(OptIn);
        var y = gpu.Sigmoid(x);
        tape.Retain(y);
        var loss = gpu.ReduceSum(gpu.TensorMultiply(y, w), null);
        tape.ComputeGradients(loss, new[] { x });
        Assert.Equal(Sigmoid(host.ToArray()), y.ToArray(), new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void AReshapeViewOfALeaf_NeverFreesTheLeafsStorage()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(7);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        for (int run = 0; run < 3; run++)   // past the probe run, where releases are real
        {
            using var tape = new GradientTape<float>(OptIn);
            var v = gpu.Reshape(x, new[] { 21, 5 });   // a recorded op whose output aliases the leaf's storage
            var loss = gpu.ReduceSum(gpu.TensorMultiply(v, v), null);
            var grads = tape.ComputeGradients(loss, new[] { x });
            Assert.Equal(host.ToArray().Select(a => 2f * a).ToArray(), grads[x].ToArray(), new FloatTolerance(1e-5f));
        }
        Assert.Equal(host.ToArray(), x.ToArray());
    }

    // loss = sum(tanh(x) * w) on another shape: a different graph from Step's.
    private static void OtherGraphStep(IEngine engine, Tensor<float> x, Tensor<float> w, bool optIn)
    {
        using var tape = new GradientTape<float>(optIn ? OptIn : null);
        var loss = engine.ReduceSum(engine.TensorMultiply(engine.Tanh(x), w), null);
        tape.ComputeGradients(loss, new[] { x });
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void AnOutputHeldAcrossAnotherGraphsBackward_StaysReadable(bool optIn)
    {
        // Keep one model's output, run a different model's backward, then compare: the first output must still be
        // readable (PyTorch keeps it while it is referenced). A probe window used to close at ANY backward run.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(9);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(10), GpuTensorRole.General);
        var x2 = gpu.UploadToGpu(new Tensor<float>(Enumerable.Range(0, 44).Select(i => (float)Math.Cos(i * 0.3)).ToArray(), new[] { 4, 11 }), GpuTensorRole.General);
        var w2 = gpu.UploadToGpu(new Tensor<float>(Enumerable.Range(0, 44).Select(i => (float)Math.Sin(i * 0.2)).ToArray(), new[] { 4, 11 }), GpuTensorRole.General);

        var y = Step(gpu, x, w, readAfterBackward: false, out _, optIn);
        OtherGraphStep(gpu, x2, w2, optIn);
        OtherGraphStep(gpu, x2, w2, optIn);
        Assert.Equal(Sigmoid(host.ToArray()), y.ToArray(), new FloatTolerance(1e-5f));
    }

    [SkippableFact]
    public void ADefaultTape_NeverFreesAnIntermediateItsCallerHolds()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");
        AutoRetainProbe.ResetForTests();
        var host = Input(11);
        var x = gpu.UploadToGpu(host, GpuTensorRole.General);
        var w = gpu.UploadToGpu(Input(12), GpuTensorRole.General);
        Step(gpu, x, w, readAfterBackward: false, out _, optIn: false);
        var y = Step(gpu, x, w, readAfterBackward: false, out _, optIn: false);   // past where an opted-in tape frees
        Step(gpu, x, w, readAfterBackward: false, out _, optIn: false);
        Assert.Equal(Sigmoid(host.ToArray()), y.ToArray(), new FloatTolerance(1e-5f));
        Assert.Equal(0, AutoRetainProbe.OpenWindowCount);   // and a default tape never probes
    }

    private sealed class FloatTolerance : System.Collections.Generic.IEqualityComparer<float>
    {
        private readonly float _tolerance;
        public FloatTolerance(float tolerance) => _tolerance = tolerance;
        public bool Equals(float a, float b) => Math.Abs(a - b) <= _tolerance * Math.Max(1f, Math.Abs(a));
        public int GetHashCode(float value) => 0;
    }
}
