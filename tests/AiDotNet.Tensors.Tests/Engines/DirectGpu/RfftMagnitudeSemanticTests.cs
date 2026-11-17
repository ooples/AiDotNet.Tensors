using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class RfftMagnitudeSemanticTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public RfftMagnitudeSemanticTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static void Verify(IEngine engine, bool compiled)
    {
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            const int rows = 3, n = 32, bins = n / 2 + 1;
            const double epsilon = .125;
            var input = new Tensor<float>(new[] { rows, n });
            var upstream = new Tensor<float>(Enumerable.Range(0, rows * bins).Select(i => (i % 7 - 3) * .125f).ToArray(), new[] { rows, bins });
            var secondUpstream = new Tensor<float>(Enumerable.Range(0, rows * bins).Select(i => (i % 5 - 2) * .0625f).ToArray(), new[] { rows, bins });
            Tensor<float>? retained = null;
            Tensor<float> Objective()
            {
                retained = engine.TensorRfftMagnitude(input, epsilon);
                return engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(retained, upstream)),
                    engine.ReduceSum(engine.TensorMultiply(engine.TensorRfftMagnitude(input, .5), secondUpstream)));
            }
            void Feed(int step)
            {
                for (int i = 0; i < input.Length; i++) input[i] = (i % 13 - 6) * .0625f + step * .03125f;
                if (engine.TryGet<DirectGpuTensorEngine>(out var gpu))
                { gpu.EnsureResidentInput(input); gpu.EnsureResidentInput(upstream); gpu.EnsureResidentInput(secondUpstream); }
            }
            Feed(0);
            ICompiledTrainingPlan<float>? plan = null;
            if (compiled)
            {
                using var graph = GraphMode.EnableTraining(new[] { input });
                var loss = Objective();
                Assert.NotNull(retained!.LazySource);
                Assert.Equal("TensorRfftMagnitude", retained.LazySource!.OpName);
                graph.RetainOutput(retained);
                plan = graph.CompileTraining(new[] { input }, loss);
                Assert.IsType<CompiledTrainingPlan<float>>(plan).DisableGraphStep();
            }
            using (plan)
                for (int step = 0; step < 3; step++)
                {
                    Feed(step);
                    using var tape = plan is null ? new GradientTape<float>() : null;
                    Tensor<float> gradient;
                    if (plan is null) gradient = tape!.ComputeGradients(Objective(), new[] { input })[input];
                    else
                    {
                        plan.Step(); gradient = plan.Gradients[0];
                        Assert.Contains("specialized:TensorRfftMagnitude", Assert.IsType<CompiledTrainingPlan<float>>(plan).BackwardStepKinds);
                    }
                    // Retained magnitude and accumulated input VJP are independent
                    // double-DFT oracles, checked after the entire backward completes.
                    var cpu = new CpuEngine();
                    var expectedMagnitude = cpu.TensorRfftMagnitude(input, epsilon).ToArray();
                    var expected = new Tensor<float>(new[] { rows, n });
                    cpu.TensorRfftMagnitudeBackwardInto(upstream, input, expected, epsilon);
                    cpu.TensorRfftMagnitudeBackwardInto(secondUpstream, input, expected, .5, accumulate: true);
                    Close(expectedMagnitude, retained!.ToArray()); Close(expected.ToArray(), gradient.ToArray());
                }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Cpu_semantic_replay_retains_outputs_and_accumulates_shared_input_gradients(bool compiled)
        => Verify(new CpuEngine(), compiled);

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Native_semantic_replay_retains_outputs_and_accumulates_shared_input_gradients(bool compiled)
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        Verify(_fixture.Engine ?? throw new InvalidOperationException(), compiled);
    }

    [SkippableFact]
    public void Native_in_place_vjp_reads_complete_primals_before_writing_and_has_no_readback()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        var values = Enumerable.Range(0, 128).Select(i => (i % 11 - 5) * .125f).ToArray();
        var input = new Tensor<float>(values, new[] { 2, 64 });
        var upstream = new Tensor<float>(Enumerable.Range(0, 66).Select(i => (i % 7 - 3) * .0625f).ToArray(), new[] { 2, 33 });
        var expected = new Tensor<float>(new[] { 2, 64 });
        new CpuEngine().TensorRfftMagnitudeBackwardInto(upstream, input, expected, .25);
        gpu.EnsureResidentInput(input); gpu.EnsureResidentInput(upstream);
        long before = GpuLaunchProbe.ReadbackBytes;
        gpu.TensorRfftMagnitudeBackwardInto(upstream, input, input, .25);
        Assert.Equal(before, GpuLaunchProbe.ReadbackBytes);
        Close(expected.ToArray(), input.ToArray());
    }

    [SkippableFact]
    public void Native_warm_resident_calls_reuse_scratch_without_input_sized_managed_storage()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        const int rows = 32, n = 1024, bins = 513;
        var input = new Tensor<float>(Enumerable.Range(0, rows * n).Select(i => (i % 13 - 6) * .03125f).ToArray(), new[] { rows, n });
        var upstream = new Tensor<float>(Enumerable.Repeat(.125f, rows * bins).ToArray(), new[] { rows, bins });
        var output = new Tensor<float>(new[] { rows, bins }); var gradient = new Tensor<float>(new[] { rows, n });
        gpu.EnsureResidentInput(input); gpu.EnsureResidentInput(upstream);
        var pool = gpu.CreateScratchPool(); var oldPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                void Execute()
                {
                    gpu.SetCurrentScratchAction(0); gpu.TensorRfftMagnitudeInto(output, input);
                    gpu.SetCurrentScratchAction(1); gpu.TensorRfftMagnitudeBackwardInto(upstream, input, gradient);
                }
                Execute(); Execute();
                long bytes = pool.BufferBytes, allocated = GC.GetAllocatedBytesForCurrentThread(), reads = GpuLaunchProbe.ReadbackBytes;
                for (int i = 0; i < 3; i++) Execute();
                Assert.Equal(bytes, pool.BufferBytes); Assert.Equal(reads, GpuLaunchProbe.ReadbackBytes);
                long payload = GC.GetAllocatedBytesForCurrentThread() - allocated;
                Assert.True(payload < rows * n * sizeof(float), $"Warmed calls allocated {payload} managed bytes.");
            }
        }
        finally { DirectGpuTensorEngine.RestoreScratchPool(oldPool); pool.Release(); gpu.ResumeActivationEviction(); }
    }

    [SkippableFact]
    public void Native_unsupported_precision_fails_explicitly()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        Assert.Throws<NotSupportedException>(() => _fixture.Engine!.TensorRfftMagnitude(new Tensor<double>(new[] { 2, 8 })));
    }

    [SkippableFact]
    public void Cuda_warmed_real_fft_forward_and_vjp_capture_replay_changed_inputs()
    {
        Skip.IfNot(_fixture.IsAvailable, "No native GPU backend.");
        var gpu = _fixture.Engine ?? throw new InvalidOperationException();
        Skip.IfNot(gpu.Name.Contains("CUDA", StringComparison.OrdinalIgnoreCase), "CUDA capture test.");
        const int rows = 2, n = 64, bins = 33;
        var input = new Tensor<float>(Enumerable.Range(0, rows * n).Select(i => (i % 11 - 5) * .125f).ToArray(), new[] { rows, n });
        var upstream = new Tensor<float>(Enumerable.Repeat(.25f, rows * bins).ToArray(), new[] { rows, bins });
        var output = new Tensor<float>(new[] { rows, bins }); var gradient = new Tensor<float>(new[] { rows, n });
        gpu.EnsureResidentInput(input); gpu.EnsureResidentInput(upstream);
        var pool = gpu.CreateScratchPool(); var oldPool = DirectGpuTensorEngine.EnterScratchPool(pool);
        IntPtr graph = IntPtr.Zero;
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath())
            {
                void Execute()
                {
                    gpu.SetCurrentScratchAction(0); gpu.TensorRfftMagnitudeInto(output, input, .125);
                    gpu.SetCurrentScratchAction(1); gpu.TensorRfftMagnitudeBackwardInto(upstream, input, gradient, .125);
                }
                Execute(); Execute(); gpu.SynchronizeStream();
                graph = gpu.CaptureGpuGraph(Execute);
                Assert.NotEqual(IntPtr.Zero, graph);
                for (int step = 0; step < 3; step++)
                {
                    for (int i = 0; i < input.Length; i++) input[i] = (i % 13 - 6) * .0625f + step * .03125f;
                    gpu.RefreshResidentInputInPlace(input);
                    gpu.LaunchGpuGraph(graph); gpu.SynchronizeStream();
                    // Replay does not run managed binding callbacks. Mark each
                    // written destination device-current, as the plan replay does.
                    gpu.BindResidentBuffer(output, output._gpuBuffer!, output._gpuBackend!);
                    gpu.BindResidentBuffer(gradient, gradient._gpuBuffer!, gradient._gpuBackend!);
                    var expectedGradient = new Tensor<float>(new[] { rows, n });
                    var cpu = new CpuEngine();
                    Close(cpu.TensorRfftMagnitude(input, .125).ToArray(), output.ToArray());
                    cpu.TensorRfftMagnitudeBackwardInto(upstream, input, expectedGradient, .125);
                    Close(expectedGradient.ToArray(), gradient.ToArray());
                }
            }
        }
        finally
        {
            if (graph != IntPtr.Zero) gpu.DestroyGpuGraph(graph);
            DirectGpuTensorEngine.RestoreScratchPool(oldPool); pool.Release(); gpu.ResumeActivationEviction();
        }
    }

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 3e-4 * (1 + Math.Abs(expected[i])),
                $"[{i}] actual={actual[i]:G9} expected={expected[i]:G9}");
    }
}
