using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class InferenceWorkspaceDemandTests
{
    [Fact]
    public void OnlyAnIncompatibleDemandInsideItsInferenceBoundaryReturnsIdleNativeBacking()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var first = new NativeCell(32);
        var second = new NativeCell(64);
        broker.Return(backend, first);
        broker.Return(backend, second);
        InferenceWorkspaceDemandScope.BeforeAllocation(backend, 65); // No inference caller.
        Assert.Equal(0, first.Releases);
        using (InferenceWorkspaceDemandScope.Enter(engine))
        {
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 64); // An existing cell can fit.
            Assert.Equal(0, first.Releases);
            using (InferenceWorkspaceDemandScope.Enter(null))
                InferenceWorkspaceDemandScope.BeforeAllocation(backend, 65); // Differentiable/capture child masks caller.
            Assert.Equal(0, first.Releases);
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 65);
        }
        Assert.Equal(1, first.Releases);
        Assert.Equal(1, second.Releases);
        Assert.Equal(0, first.Disposes);
        Assert.Equal(0, second.Disposes);
        Assert.False(StepScratchPool.IsPooled(first));
        Assert.Equal(0, broker.RetainedBytes);
        Assert.Equal(384, broker.NativeReturnedForInferenceBytes);
        Assert.Equal(2, broker.NativeReturnedForInferenceCells);
        Assert.Null(broker.TryRent(backend, 32));
        engine.ClearActionScratchPool();
        Assert.Equal(1, first.Releases); // Clearing the terminal owner cannot release transferred cells twice.
    }

    [Fact]
    public void AnotherBackendOrThreadAndAnOutstandingLoanStayOutsideTheNativeReturn()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        var otherBackend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var broker = engine.CompiledWorkspaceBroker;
        var idle = new NativeCell(32);
        var active = new NativeCell(32);
        var other = new NativeCell(32);
        broker.Return(backend, active);
        Assert.Same(active, broker.TryRent(backend, 32));
        broker.Return(backend, idle);
        broker.Return(otherBackend, other);
        var thread = new Thread(() =>
        {
            using var scope = InferenceWorkspaceDemandScope.Enter(engine);
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 33);
        });
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(10)));
        Assert.Equal(0, idle.Releases);
        using (InferenceWorkspaceDemandScope.Enter(engine))
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 33);
        Assert.Equal(1, idle.Releases);
        Assert.Equal(0, active.Releases);
        Assert.Equal(0, other.Releases);
        Assert.Equal(128, broker.RetainedBytes);
        broker.Return(backend, active);
    }

    [Fact]
    public void ResidentExecutionAndAnActiveDifferentiationKeepIdleStorage()
    {
        var backend = MockDirectGpuBackend.Create(new MockBackendState());
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var cell = new NativeCell(16);
        engine.CompiledWorkspaceBroker.Return(backend, cell);
        using var boundary = InferenceWorkspaceDemandScope.Enter(engine);
        engine.SuspendActivationEviction();
        try
        {
            using (engine.EnterCompiledCapturePath())
                InferenceWorkspaceDemandScope.BeforeAllocation(backend, 17);
        }
        finally { engine.ResumeActivationEviction(); }
        Assert.Equal(0, cell.Releases);
        using (var tape = new GradientTape<float>())
            InferenceWorkspaceDemandScope.BeforeAllocation(backend, 17);
        Assert.Equal(0, cell.Releases);
        InferenceWorkspaceDemandScope.BeforeAllocation(backend, 17);
        Assert.Equal(1, cell.Releases);
    }

    private sealed class NativeCell : IGpuBuffer, IPoolableGpuBuffer
    {
        private static int s_next;
        public NativeCell(int size) { Size = size; Handle = (IntPtr)Interlocked.Increment(ref s_next); }
        public AiDotNet.Tensors.Engines.DirectGpu.GpuElementType ElementType => AiDotNet.Tensors.Engines.DirectGpu.GpuElementType.Float32;
        public int Size { get; }
        public int Capacity => Size;
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        public int Releases { get; private set; }
        public int Disposes { get; private set; }
        public void MarkRented(int size) => throw new InvalidOperationException("No host-pool rent in this fixture.");
        public void Release() { Releases++; Handle = IntPtr.Zero; }
        public void Dispose() { Disposes++; Handle = IntPtr.Zero; }
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TrainPredictTrainPreservesUnreadParameterGradientsAndRetainedViews(bool inputGradients)
    {
        using var gpu = CreateGpu();
        var previous = AiDotNetEngine.Current;
        bool? oldInputGradients = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        try
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = inputGradients;
            var expected = Trajectory(new CpuEngine(), inputGradients, out _);
            var actual = Trajectory(gpu, inputGradients, out long returned);
            Assert.True(returned > 0, "The inference request must actually return compiled native backing.");
            Assert.Equal(expected.Count, actual.Count);
            for (int sample = 0; sample < expected.Count; sample++)
            {
                Assert.Equal(expected[sample].Length, actual[sample].Length);
                for (int i = 0; i < expected[sample].Length; i++)
                    Assert.True(Math.Abs(expected[sample][i] - actual[sample][i]) <= 2e-6f + 3e-3f * Math.Abs(expected[sample][i]),
                        $"sample{sample}[{i}]: {expected[sample][i]} vs {actual[sample][i]}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = oldInputGradients;
        }
    }

    private static List<float[]> Trajectory(IEngine engine, bool inputGradients, out long returned)
    {
        AiDotNetEngine.Current = engine;
        var input = Values(8, 8, 1);
        var weight = Values(8, 8, 2);
        CompiledTrainingPlan<float> plan;
        using (var graph = GraphMode.EnableTraining(new[] { weight }))
        {
            var value = engine.TensorMatMul(input, weight);
            for (int i = 0; i < 4; i++) value = engine.Tanh(engine.TensorMultiplyScalar(value, 0.9f));
            GradientRecording.RetainOutput("public", engine.Reshape(engine.TensorNarrow(value, 0, 1, 2), new[] { 2, 8 }));
            var loss = engine.ReduceMean(engine.TensorMultiply(value, value), new[] { 0, 1 }, false);
            plan = Assert.IsType<CompiledTrainingPlan<float>>(graph.CompileTraining(new[] { weight }, loss));
        }
        using (plan)
        {
            plan.DisableGraphStep();
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.001);
            var samples = new List<float[]>();
            for (int iteration = 0; iteration < 3; iteration++)
            {
                input[0] = 0.1f + iteration * 0.01f;
                using var retained = RetainedOutputCapture<float>.Begin();
                var loss = plan.Step();
                retained.Complete();
                var view = retained.Get("public");
                var parameterGradient = plan.Gradients[0].Reshape(new[] { 8, 8 });
                var inputGradient = input.Grad;
                // Keep every public device result unread until a larger unrelated inference demand has retired idle backing.
                using (new NoGradScope<float>())
                using (InferenceWorkspaceDemandScope.Enter(engine))
                using (var prediction = engine.TensorAdd(new Tensor<float>(new[] { 4096 }), new Tensor<float>(new[] { 4096 })))
                {
                    Assert.Equal(4096, prediction.Length);
                }
                samples.Add(loss.ToArray());
                samples.Add(parameterGradient.ToArray());
                samples.Add(view.ToArray());
                samples.Add(weight.ToArray());
                if (inputGradients) samples.Add(Assert.IsType<Tensor<float>>(inputGradient).ToArray());
            }
            returned = engine is DirectGpuTensorEngine direct ? direct.CompiledWorkspaceBroker.NativeReturnedForInferenceBytes : 0;
            return samples;
        }
    }

    private static Tensor<float> Values(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var data = new float[rows * columns];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(random.NextDouble() - 0.5) * 0.1f;
        return new Tensor<float>(data, new[] { rows, columns });
    }

    private static DirectGpuTensorEngine CreateGpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or DllNotFoundException)
        { Skip.If(true, "CUDA unavailable: " + e.GetType().Name); throw; }
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend, "CUDA required.");
        return gpu;
    }
}
