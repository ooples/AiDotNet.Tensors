using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>The CPU engine's <see cref="IAsyncEngine"/>: the synchronous result, in a task that is already complete.</summary>
public sealed class CpuAsyncEngineTests
{
    private static Tensor<float> Values(int seed, params int[] shape)
    {
        int n = shape.Aggregate(1, (x, y) => x * y);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => 0.25f + MathF.Abs(MathF.Sin(seed + 0.37f * i))).ToArray(), shape);
    }

    [Fact]
    public void Every_operation_returns_the_synchronous_result_already_complete()
    {
        IAsyncEngine engine = new CpuEngine();
        var cpu = new CpuEngine();
        var a = Values(1, 3, 4);
        var b = Values(2, 3, 4);
        var m = Values(3, 4, 5);
        var cases = new (ValueTask<Tensor<float>> Async, Tensor<float> Sync)[]
        {
            (engine.TensorAddAsync(a, b), cpu.TensorAdd(a, b)),
            (engine.TensorSubtractAsync(a, b), cpu.TensorSubtract(a, b)),
            (engine.TensorMultiplyAsync(a, b), cpu.TensorMultiply(a, b)),
            (engine.TensorDivideAsync(a, b), cpu.TensorDivide(a, b)),
            (engine.TensorMatMulAsync(a, m), cpu.TensorMatMul(a, m)),
            (engine.TensorExpAsync(a), cpu.TensorExp(a)),
            (engine.TensorLogAsync(a), cpu.TensorLog(a)),
            (engine.TensorSqrtAsync(a), cpu.TensorSqrt(a)),
            (engine.TensorAbsAsync(a), cpu.TensorAbs(a)),
            (engine.TensorNegateAsync(a), cpu.TensorNegate(a)),
            (engine.ReLUAsync(a), cpu.ReLU(a)),
            (engine.SigmoidAsync(a), cpu.Sigmoid(a)),
            (engine.TanhAsync(a), cpu.Tanh(a)),
        };
        foreach (var (task, expected) in cases)
        {
            Assert.True(task.IsCompletedSuccessfully);
            Assert.Equal(expected.Shape.ToArray(), task.Result.Shape.ToArray());
            Assert.Equal(expected.ToArray(), task.Result.ToArray());
        }
    }

    [Fact]
    public async Task A_cancelled_token_stops_the_operation_before_it_runs()
    {
        IAsyncEngine engine = new CpuEngine();
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        var a = Values(1, 4);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TensorAddAsync(a, a, cancelled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TensorExpAsync(a, cancelled.Token).AsTask());
    }
}

#if NET7_0_OR_GREATER
/// <summary>
/// <see cref="AwaitableGpuTensorEngine"/> on the real WebGPU backend, executing on Dawn. A counting decorator proves
/// which operations reached the device, so a result that silently came from the CPU cannot pass as a GPU one.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class AwaitableGpuTensorEngineTests
{
    private static Tensor<float> Values(int seed, params int[] shape)
    {
        int n = shape.Aggregate(1, (x, y) => x * y);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => 0.25f + MathF.Abs(MathF.Sin(seed + 0.37f * i))).ToArray(), shape);
    }

    private static Tensor<float> Signed(int seed, params int[] shape)
    {
        int n = shape.Aggregate(1, (x, y) => x * y);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => 2f * MathF.Sin(seed + 0.61f * i)).ToArray(), shape);
    }

    private static async Task<(AwaitableGpuTensorEngine Engine, CountingAwaitableBackend Counter)> StartAsync()
    {
        // Sets up the Dawn host this process shares, or skips where Dawn is not installed.
        DirectGpu.DawnHostedWebGpu.BackendOrSkip();
        var backend = new AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuAwaitableBackend();
        Assert.True(await backend.InitializeAsync());
        var counter = new CountingAwaitableBackend(backend);
        return (new AwaitableGpuTensorEngine(counter, ownsBackend: true), counter);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float relative)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= relative * MathF.Max(1f, MathF.Abs(e[i])),
                $"element {i}: expected {e[i]:R}, got {a[i]:R}");
    }

    [SkippableFact]
    public async Task Elementwise_operations_run_on_the_device_and_match_the_cpu()
    {
        var (engine, counter) = await StartAsync();
        using var _ = engine;
        var cpu = new CpuEngine();
        var a = Values(1, 3, 5, 7);
        var b = Values(2, 3, 5, 7);
        var s = Signed(3, 3, 5, 7);
        var cases = new (string Name, Func<Task<Tensor<float>>> Async, Tensor<float> Expected)[]
        {
            ("add", () => engine.TensorAddAsync(a, b).AsTask(), cpu.TensorAdd(a, b)),
            ("subtract", () => engine.TensorSubtractAsync(a, b).AsTask(), cpu.TensorSubtract(a, b)),
            ("multiply", () => engine.TensorMultiplyAsync(a, b).AsTask(), cpu.TensorMultiply(a, b)),
            ("divide", () => engine.TensorDivideAsync(a, b).AsTask(), cpu.TensorDivide(a, b)),
            ("exp", () => engine.TensorExpAsync(s).AsTask(), cpu.TensorExp(s)),
            ("log", () => engine.TensorLogAsync(a).AsTask(), cpu.TensorLog(a)),
            ("sqrt", () => engine.TensorSqrtAsync(a).AsTask(), cpu.TensorSqrt(a)),
            ("abs", () => engine.TensorAbsAsync(s).AsTask(), cpu.TensorAbs(s)),
            ("negate", () => engine.TensorNegateAsync(s).AsTask(), cpu.TensorNegate(s)),
            ("relu", () => engine.ReLUAsync(s).AsTask(), cpu.ReLU(s)),
            ("sigmoid", () => engine.SigmoidAsync(s).AsTask(), cpu.Sigmoid(s)),
            ("tanh", () => engine.TanhAsync(s).AsTask(), cpu.Tanh(s)),
        };
        foreach (var (name, run, expected) in cases)
        {
            int before = counter.Kernels;
            var actual = await run();
            Assert.True(counter.Kernels == before + 1, $"{name} did not run on the device");
            AssertClose(expected, actual, 2e-6f);
        }
    }

    [SkippableFact]
    public async Task A_matrix_product_runs_on_the_device_and_matches_the_cpu()
    {
        var (engine, counter) = await StartAsync();
        using var _ = engine;
        var a = Signed(4, 17, 33);
        var b = Signed(5, 33, 9);
        var actual = await engine.TensorMatMulAsync(a, b);
        Assert.Equal(1, counter.Kernels);
        AssertClose(new CpuEngine().TensorMatMul(a, b), actual, 1e-5f);
    }

    [SkippableFact]
    public async Task Operations_the_kernels_do_not_cover_give_the_cpu_answer_without_touching_the_device()
    {
        var (engine, counter) = await StartAsync();
        using var _ = engine;
        var cpu = new CpuEngine();
        var a = Values(1, 3, 4);
        var row = Values(2, 1, 4);
        var doubles = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var batched = Values(3, 2, 3, 4);
        var right = Values(4, 4, 5);

        AssertClose(cpu.TensorAdd(a, row), await engine.TensorAddAsync(a, row), 0f);
        Assert.Equal(cpu.TensorExp(doubles).ToArray(), (await engine.TensorExpAsync(doubles)).ToArray());
        AssertClose(cpu.TensorMatMul(batched, right), await engine.TensorMatMulAsync(batched, right), 0f);
        Assert.Equal(0, counter.Kernels);
    }

    [SkippableFact]
    public async Task Under_a_gradient_tape_the_operation_is_recorded_on_the_calling_thread()
    {
        var (engine, counter) = await StartAsync();
        using var _ = engine;
        var a = Values(1, 2, 3);
        var b = Values(2, 2, 3);
        float[] expected, actual;
        var cpu = new CpuEngine();
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
        {
            var y = cpu.TensorMultiply(a, b);
            expected = tape.ComputeGradients(cpu.ReduceSum(y, null), new[] { a })[a].ToArray();
        }
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>())
        {
            var y = await engine.TensorMultiplyAsync(a, b);
            actual = tape.ComputeGradients(cpu.ReduceSum(y, null), new[] { a })[a].ToArray();
        }
        Assert.Equal(expected, actual);
        Assert.Equal(b.ToArray(), actual);
        Assert.Equal(0, counter.Kernels);
    }

    [SkippableFact]
    public async Task The_factory_picks_webgpu_when_the_platform_offers_it()
    {
        DirectGpu.DawnHostedWebGpu.BackendOrSkip();
        var engine = await AsyncEngine.CreateAsync();
        try
        {
            var gpu = Assert.IsType<AwaitableGpuTensorEngine>(engine);
            Assert.Equal("WebGPU", gpu.Backend.BackendName);
            var a = Values(1, 8);
            AssertClose(new CpuEngine().TensorAdd(a, a), await engine.TensorAddAsync(a, a), 0f);
        }
        finally
        {
            (engine as IDisposable)?.Dispose();
        }
    }
}

/// <summary>Counts the kernels an <see cref="AiDotNet.Tensors.Engines.DirectGpu.IAwaitableGpuBackend"/> is asked to run.</summary>
internal sealed class CountingAwaitableBackend : AiDotNet.Tensors.Engines.DirectGpu.IAwaitableGpuBackend
{
    private readonly AiDotNet.Tensors.Engines.DirectGpu.IAwaitableGpuBackend _inner;
    private int _kernels;

    public CountingAwaitableBackend(AiDotNet.Tensors.Engines.DirectGpu.IAwaitableGpuBackend inner) => _inner = inner;

    public int Kernels => Volatile.Read(ref _kernels);

    private Task Count(Task kernel)
    {
        Interlocked.Increment(ref _kernels);
        return kernel;
    }

    public bool IsAvailable => _inner.IsAvailable;
    public string BackendName => _inner.BackendName;
    public TensorDevice DeviceType => _inner.DeviceType;
    public string DeviceName => _inner.DeviceName;
    public string DeviceVendor => _inner.DeviceVendor;
    public int ComputeUnits => _inner.ComputeUnits;
    public long GlobalMemoryBytes => _inner.GlobalMemoryBytes;
    public long LocalMemoryBytes => _inner.LocalMemoryBytes;
    public long MaxBufferAllocBytes => _inner.MaxBufferAllocBytes;
    public double TheoreticalGflops => _inner.TheoreticalGflops;
    public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer AllocateBuffer(float[] data) => _inner.AllocateBuffer(data);
    public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer AllocateBuffer(int size) => _inner.AllocateBuffer(size);
    public bool SupportsElementType(AiDotNet.Tensors.Engines.DirectGpu.GpuElementType type) => _inner.SupportsElementType(type);
    public AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer AllocateBuffer(int size, AiDotNet.Tensors.Engines.DirectGpu.GpuElementType type)
        => _inner.AllocateBuffer(size, type);
    public Task<float[]> DownloadBufferAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer buffer) => _inner.DownloadBufferAsync(buffer);
    public Task AddAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer C, int size) => Count(_inner.AddAsync(A, B, C, size));
    public Task SubtractAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer C, int size) => Count(_inner.SubtractAsync(A, B, C, size));
    public Task MultiplyAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer C, int size) => Count(_inner.MultiplyAsync(A, B, C, size));
    public Task DivideAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer C, int size) => Count(_inner.DivideAsync(A, B, C, size));
    public Task ExpAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.ExpAsync(A, B, size));
    public Task LogAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.LogAsync(A, B, size));
    public Task SqrtAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.SqrtAsync(A, B, size));
    public Task AbsAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.AbsAsync(A, B, size));
    public Task NegateAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.NegateAsync(A, B, size));
    public Task ReluAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.ReluAsync(A, B, size));
    public Task SigmoidAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.SigmoidAsync(A, B, size));
    public Task TanhAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, int size) => Count(_inner.TanhAsync(A, B, size));
    public Task GemmAsync(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer A, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer B, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer C, int M, int N, int K, float alpha = 1f, float beta = 0f)
        => Count(_inner.GemmAsync(A, B, C, M, N, K, alpha, beta));
    public void Dispose() => _inner.Dispose();
}
#endif
