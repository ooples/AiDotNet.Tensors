// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Releasing a GPU engine while other threads run ops on it: <see cref="DirectGpuTensorEngine.Dispose"/> waits for
/// their leases, new ops take the CPU paths, and every device result keeps its data. In this namespace so every GPU
/// lane runs it.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class GpuEngineReleaseTests
{
    private const int N = 48;

    private static Tensor<float> Matrix(int seed)
    {
        var rng = new Random(seed);
        var data = new float[N * N];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        return new Tensor<float>(data, new[] { N, N });
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])),
                $"{label}: element {i} expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>
    /// Four threads multiply and read results back while the engine is disposed under them. Disposal frees the backend
    /// only once their in-flight ops are done (the cudnnSetStream access violation was a free under a running op),
    /// later ops run on the CPU paths, and every result read back is the right one.
    /// </summary>
    [SkippableFact]
    public void Disposing_under_ops_on_other_threads_waits_for_them_and_every_result_is_right()
    {
        var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var a = Matrix(1);
        var b = Matrix(2);
        var expected = new CpuEngine().TensorMatMul(a, b).ToArray();

        using var stop = new CancellationTokenSource();
        int completed = 0;
        var workers = new Task[4];
        for (int w = 0; w < workers.Length; w++)
        {
            workers[w] = Task.Run(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    var product = gpu.TensorMatMul(a, b);
                    AssertClose(expected, product.ToArray(), "a product computed across the disposal");
                    Interlocked.Increment(ref completed);
                }
            });
        }

        SpinWait.SpinUntil(() => Volatile.Read(ref completed) >= 20, TimeSpan.FromSeconds(30));
        gpu.Dispose();
        Assert.True(gpu.IsDisposed);
        Assert.False(gpu.IsGpuAvailable);
        int atDisposal = Volatile.Read(ref completed);
        SpinWait.SpinUntil(() => Volatile.Read(ref completed) >= atDisposal + 20, TimeSpan.FromSeconds(30));
        stop.Cancel();
        Task.WaitAll(workers);
        Assert.True(Volatile.Read(ref completed) >= atDisposal + 20, "ops stopped completing after the disposal");
    }

    /// <summary>
    /// A result left on the device (never read) when the adopted engine is released is downloaded first: reading it
    /// afterwards gives its values, not a fault on a freed buffer.
    /// </summary>
    [SkippableFact]
    public void Releasing_the_adopted_engine_keeps_the_data_of_results_still_on_the_device()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);
            var a = Matrix(3);
            var b = Matrix(4);
            var expected = new CpuEngine().TensorMatMul(a, b).ToArray();
            var onDevice = gpu.TensorMatMul(a, b);

            Assert.True(AiDotNetEngine.ReleaseGpu());
            Assert.True(gpu.IsDisposed);
            Assert.IsType<CpuEngine>(AiDotNetEngine.Current);
            Assert.Equal(0, gpu.LiveOwnedResultsForTest);
            AssertClose(expected, onDevice.ToArray(), "a device result read after the release");
            Assert.False(AiDotNetEngine.ReleaseGpu());
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>
    /// Parameters a compiled training plan updated on the device keep the trained values through a release: the
    /// weights read afterwards are those of a CPU run of the same steps, not their initial values.
    /// </summary>
    [SkippableFact]
    public void Releasing_the_adopted_engine_keeps_the_trained_parameters()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            var expected = TrainThreeSteps(new CpuEngine());
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);
            var w = TrainThreeSteps(gpu, readBack: false);

            Assert.True(AiDotNetEngine.ReleaseGpu());
            Assert.True(gpu.IsDisposed);
            AssertClose(expected.ToArray(), w.ToArray(), "a trained weight read after the release");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    // Three SGD steps of a compiled plan for loss = sum((a @ w) * b), on the given engine; returns w.
    private static Tensor<float> TrainThreeSteps(IEngine engine, bool readBack = true)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var a = Matrix(11);
            var b = Matrix(12);
            var w = Matrix(13);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorMultiply(engine.TensorMatMul(a, w), b), null);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
                for (int step = 0; step < 3; step++) plan.Step();
            }
            if (readBack) _ = w.ToArray();
            return w;
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// A release that cannot drain in time frees nothing: the engine is parked again, still works, the next adoption
    /// reinstalls it, and it releases once the op holding it is done.
    /// </summary>
    [SkippableFact]
    public void A_release_that_times_out_frees_nothing_and_leaves_the_engine_usable()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);

            using var held = new ManualResetEventSlim();
            using var finish = new ManualResetEventSlim();
            var op = Task.Run(() =>
            {
                using var lease = gpu.EnterGpuOp();
                held.Set();
                finish.Wait();
            });
            held.Wait();

            Assert.Throws<TimeoutException>(() => AiDotNetEngine.ReleaseGpu(TimeSpan.FromMilliseconds(100)));
            Assert.False(gpu.IsDisposed);
            Assert.True(gpu.IsGpuAvailable);
            Assert.True(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false));
            Assert.Same(gpu, AiDotNetEngine.Current);
            var a = Matrix(5);
            AssertClose(new CpuEngine().TensorAdd(a, a).ToArray(), gpu.TensorAdd(a, a).ToArray(), "after the timeout");

            finish.Set();
            op.Wait();
            Assert.True(AiDotNetEngine.ReleaseGpu());
            Assert.True(gpu.IsDisposed);
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>An op cannot release the engine it runs on: it would wait for itself, or free under its own frame.</summary>
    [SkippableFact]
    public void Releasing_from_inside_an_op_on_the_engine_throws_and_frees_nothing()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);
            using (gpu.EnterGpuOp())
                Assert.Throws<InvalidOperationException>(() => AiDotNetEngine.ReleaseGpu());
            Assert.False(gpu.IsDisposed);
            Assert.True(AiDotNetEngine.ReleaseGpu());
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    /// <summary>
    /// A reset keeps the adopted engine but frees the device memory nothing uses (torch.cuda.empty_cache): results no
    /// object references any more, and the buffers the pool retained for reuse. Measured on a backend that reports
    /// free device memory (CUDA, HIP).
    /// </summary>
    [SkippableFact]
    public void A_reset_frees_the_device_memory_nothing_uses()
    {
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            Skip.IfNot(AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false), "No GPU was adopted");
            var gpu = Assert.IsType<DirectGpuTensorEngine>(AiDotNetEngine.Current);
            var pressure = gpu.QueryBackend as IDeviceMemoryPressure;
            Skip.If(pressure is null || !pressure.TryGetDeviceMemory(out _, out _),
                "Not applicable: this backend cannot report free device memory, so the release cannot be measured.");

            const int Elements = 8 << 20; // 32 MiB per result
            DropDeviceResults(gpu, Elements, count: 8);
            pressure!.TryGetDeviceMemory(out ulong before, out _);
            AiDotNetEngine.ResetToCpu();
            pressure.TryGetDeviceMemory(out ulong after, out _);

            Assert.False(gpu.IsDisposed);
            long freed = (long)after - (long)before;
            Assert.True(freed >= 4L * Elements * sizeof(float),
                $"the reset freed {freed >> 20} MiB of the {8L * Elements * sizeof(float) >> 20} MiB nothing used");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }

    // Results computed and dropped (in their own frame, so nothing on this one keeps them reachable).
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    internal static void DropDeviceResults(DirectGpuTensorEngine gpu, int elements, int count)
    {
        var x = new Tensor<float>(new float[elements], new[] { elements });
        for (int i = 0; i < count; i++)
        {
            var y = gpu.TensorAdd(x, x);
            GC.KeepAlive(y);
        }
    }

    /// <summary>
    /// Every op this sweep drives leases the backend it borrows (strict mode on this engine throws on a borrow without
    /// one): eager ops, a recorded backward, and a compiled training plan's steps.
    /// </summary>
    [SkippableFact]
    public void Every_op_in_the_sweep_borrows_the_backend_under_a_lease()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available");
        var previous = AiDotNetEngine.Current;
        gpu.StrictGpuLeases = true;
        try
        {
            AiDotNetEngine.Current = gpu;
            var a = Matrix(6);
            var b = Matrix(7);
            var image = new Tensor<float>(new[] { 2, 3, 8, 8 });
            for (int i = 0; i < image.Length; i++) image[i] = (i % 7) * 0.1f;
            var kernel = new Tensor<float>(new[] { 4, 3, 3, 3 });
            for (int i = 0; i < kernel.Length; i++) kernel[i] = (i % 5) * 0.05f;

            _ = gpu.TensorMatMul(a, b).ToArray();
            _ = gpu.TensorAdd(a, b).ToArray();
            _ = gpu.TensorMultiply(a, b).ToArray();
            _ = gpu.TensorSoftmax(a, 1).ToArray();
            _ = gpu.ReduceSum(a, null).ToArray();
            _ = gpu.Conv2D(image, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }).ToArray();
            _ = gpu.TensorSlice(a, new[] { 0, 4 }, new[] { N, 8 }).ToArray();
            _ = gpu.TensorConcatenate(new[] { a, b }, axis: 1).ToArray();

            using (var tape = new GradientTape<float>())
            {
                var loss = gpu.ReduceSum(gpu.TensorMultiply(gpu.TensorMatMul(a, b), a), null);
                _ = tape.ComputeGradients(loss, new[] { a, b })[a].ToArray();
            }

            ICompiledTrainingPlan<float> plan;
            var w = Matrix(8);
            using (var scope = GraphMode.Enable())
            {
                gpu.ReduceSum(gpu.TensorMultiply(gpu.TensorMatMul(a, w), b), null);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
                for (int step = 0; step < 3; step++) _ = plan.Step().ToArray();
            }
        }
        finally
        {
            gpu.StrictGpuLeases = false;
            AiDotNetEngine.Current = previous;
        }
    }
}
