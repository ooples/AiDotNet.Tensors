// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="AwaitableGpuTensorEngine"/> against a backend that keeps its buffers in host arrays, so the engine's own
/// logic (which operations go to the backend, the buffer hand-over, every fallback to the CPU, ownership and disposal)
/// runs on a machine with no GPU. The kernels themselves are tested on a real device in AwaitableGpuTensorEngineTests.
/// </summary>
public sealed class AwaitableGpuTensorEngineHostBackendTests
{
    private static Tensor<float> Values(int seed, params int[] shape)
    {
        int n = shape.Aggregate(1, (x, y) => x * y);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => 0.25f + MathF.Abs(MathF.Sin(seed + 0.37f * i))).ToArray(), shape);
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        for (int i = 0; i < e.Length; i++)
            Assert.True(MathF.Abs(e[i] - a[i]) <= 1e-5f * MathF.Max(1f, MathF.Abs(e[i])), $"element {i}: expected {e[i]:R}, got {a[i]:R}");
    }

    [Fact]
    public async Task Every_float_operation_runs_one_backend_kernel_and_matches_the_cpu()
    {
        var backend = new HostAwaitableBackend();
        using var engine = new AwaitableGpuTensorEngine(backend);
        var cpu = new CpuEngine();
        Assert.Same(backend, engine.Backend);
        var a = Values(1, 3, 5);
        var b = Values(2, 3, 5);

        AssertClose(cpu.TensorAdd(a, b), await engine.TensorAddAsync(a, b));
        AssertClose(cpu.TensorSubtract(a, b), await engine.TensorSubtractAsync(a, b));
        AssertClose(cpu.TensorMultiply(a, b), await engine.TensorMultiplyAsync(a, b));
        AssertClose(cpu.TensorDivide(a, b), await engine.TensorDivideAsync(a, b));
        AssertClose(cpu.TensorExp(a), await engine.TensorExpAsync(a));
        AssertClose(cpu.TensorLog(a), await engine.TensorLogAsync(a));
        AssertClose(cpu.TensorSqrt(a), await engine.TensorSqrtAsync(a));
        AssertClose(cpu.TensorAbs(a), await engine.TensorAbsAsync(a));
        AssertClose(cpu.TensorNegate(a), await engine.TensorNegateAsync(a));
        AssertClose(cpu.ReLU(a), await engine.ReLUAsync(a));
        AssertClose(cpu.Sigmoid(a), await engine.SigmoidAsync(a));
        AssertClose(cpu.Tanh(a), await engine.TanhAsync(a));
        Assert.Equal(12, backend.Kernels);

        var left = Values(3, 4, 6);
        var right = Values(4, 6, 2);
        AssertClose(cpu.TensorMatMul(left, right), await engine.TensorMatMulAsync(left, right));
        Assert.Equal(13, backend.Kernels);
        // Every buffer the engine allocated was released again.
        Assert.Equal(0, backend.LiveBuffers);
    }

    [Fact]
    public async Task A_result_longer_than_the_tensor_is_cut_to_the_tensors_length()
    {
        // A pooled device buffer can be longer than what was asked for.
        var backend = new HostAwaitableBackend { DownloadPadding = 7 };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var a = Values(5, 2, 3);
        var sum = await engine.TensorAddAsync(a, a);
        Assert.Equal(6, sum.Length);
        AssertClose(new CpuEngine().TensorAdd(a, a), sum);
    }

    [Fact]
    public async Task What_the_kernels_do_not_cover_is_computed_on_the_cpu_without_the_backend()
    {
        var backend = new HostAwaitableBackend();
        using var engine = new AwaitableGpuTensorEngine(backend);
        var cpu = new CpuEngine();

        // Another element type.
        var d = new Tensor<double>(new[] { 1.5, -2.0, 4.0 }, new[] { 3 });
        Assert.Equal(cpu.TensorAdd(d, d).ToArray(), (await engine.TensorAddAsync(d, d)).ToArray());
        Assert.Equal(cpu.TensorAbs(d).ToArray(), (await engine.TensorAbsAsync(d)).ToArray());
        var dm = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 2 });
        Assert.Equal(cpu.TensorMatMul(dm, dm).ToArray(), (await engine.TensorMatMulAsync(dm, dm)).ToArray());

        // Operands of different shapes (the CPU broadcasts), and a product that is not matrix times matrix.
        var matrix = Values(1, 2, 3);
        var row = Values(2, 1, 3);
        AssertClose(cpu.TensorAdd(matrix, row), await engine.TensorAddAsync(matrix, row));
        var batch = Values(3, 2, 2, 3);
        var columns = Values(4, 2, 3, 2);
        AssertClose(cpu.TensorMatMul(batch, columns), await engine.TensorMatMulAsync(batch, columns));

        Assert.Equal(0, backend.Kernels);
        Assert.Equal(0, backend.Allocations);
    }

    [Fact]
    public async Task Under_a_gradient_tape_the_cpu_computes_so_the_operation_is_recorded()
    {
        var backend = new HostAwaitableBackend();
        using var engine = new AwaitableGpuTensorEngine(backend);
        var a = Values(1, 4);
        var b = Values(2, 4);
        var cpu = new CpuEngine();
        float[] gradient;
        using (var tape = new GradientTape<float>())
        {
            AssertClose(cpu.TensorExp(a), await engine.TensorExpAsync(a));
            var product = await engine.TensorMultiplyAsync(a, b);
            AssertClose(cpu.TensorMultiply(a, b), product);
            // Recorded, not only computed: the gradient of sum(a * b) with respect to a is b.
            gradient = tape.ComputeGradients(cpu.ReduceSum(product, null), new[] { a })[a].ToArray();
        }

        Assert.Equal(b.ToArray(), gradient);
        Assert.Equal(0, backend.Kernels);
    }

    [Fact]
    public async Task In_anomaly_mode_the_cpu_computes_so_a_division_by_zero_is_the_cpus_infinity()
    {
        // The backend answers a division by zero with a finite value, as a device is free to.
        var backend = new HostAwaitableBackend { DivisionByZeroResult = 0f };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var one = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var zero = new Tensor<float>(new[] { 0f, 4f }, new[] { 2 });

        Assert.Equal(new[] { 0f, 0.5f }, (await engine.TensorDivideAsync(one, zero)).ToArray());
        Assert.Equal(1, backend.Kernels);
        using (new AnomalyModeScope())
        {
            Assert.Equal(new[] { float.PositiveInfinity, 0.5f }, (await engine.TensorDivideAsync(one, zero)).ToArray());
        }

        Assert.Equal(1, backend.Kernels);
    }

    [Fact]
    public async Task A_product_too_large_for_a_device_buffer_never_reaches_the_backend()
    {
        // 65537 x 65537 elements: m * n wraps to 131073 as an int, which would size the output for a far smaller product.
        var backend = new HostAwaitableBackend();
        using var engine = new AwaitableGpuTensorEngine(backend);
        var column = new Tensor<float>(new float[65537], new[] { 65537, 1 });
        var row = new Tensor<float>(new float[65537], new[] { 1, 65537 });
        await Assert.ThrowsAnyAsync<Exception>(() => engine.TensorMatMulAsync(column, row).AsTask());
        Assert.Equal(0, backend.Kernels);
        Assert.Equal(0, backend.Allocations);
    }

    [Fact]
    public async Task A_capability_the_backend_lacks_falls_back_to_the_cpu_and_releases_its_buffers()
    {
        var backend = new HostAwaitableBackend { Unsupported = true };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var cpu = new CpuEngine();
        var a = Values(1, 2, 2);

        AssertClose(cpu.TensorAdd(a, a), await engine.TensorAddAsync(a, a));
        AssertClose(cpu.TensorSqrt(a), await engine.TensorSqrtAsync(a));
        AssertClose(cpu.TensorMatMul(a, a), await engine.TensorMatMulAsync(a, a));
        Assert.Equal(3, backend.Kernels);
        Assert.Equal(0, backend.LiveBuffers);
    }

    [Fact]
    public async Task A_backend_failure_that_is_not_a_capability_gap_reaches_the_caller()
    {
        var backend = new HostAwaitableBackend { Failure = new InvalidOperationException("device lost") };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var a = Values(1, 3);
        var thrown = await Assert.ThrowsAsync<InvalidOperationException>(() => engine.TensorAddAsync(a, a).AsTask());
        Assert.Equal("device lost", thrown.Message);
        Assert.Equal(0, backend.LiveBuffers);
    }

    [Fact]
    public async Task Arguments_and_cancellation_are_checked_before_anything_runs()
    {
        var backend = new HostAwaitableBackend();
        using var engine = new AwaitableGpuTensorEngine(backend);
        var a = Values(1, 2, 2);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TensorAddAsync(a, a, cancelled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TensorExpAsync(a, cancelled.Token).AsTask());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.TensorMatMulAsync(a, a, cancelled.Token).AsTask());
        // Null arguments on purpose: the annotations would otherwise warn about exactly what is under test.
#nullable disable
        Tensor<float> none = null;
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.TensorAddAsync(none, a).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.TensorAddAsync(a, none).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.TensorExpAsync(none).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.TensorMatMulAsync(none, a).AsTask());
        await Assert.ThrowsAsync<ArgumentNullException>(() => engine.TensorMatMulAsync(a, none).AsTask());
#nullable restore
        Assert.Equal(0, backend.Kernels);
    }

    [Fact]
    public void The_constructor_refuses_a_missing_or_unavailable_backend()
    {
#nullable disable
        IAwaitableGpuBackend none = null;
        Assert.Throws<ArgumentNullException>(() => new AwaitableGpuTensorEngine(none));
#nullable restore
        Assert.Throws<ArgumentException>(() => new AwaitableGpuTensorEngine(new HostAwaitableBackend { Available = false }));
    }

    [Fact]
    public async Task Disposal_releases_an_owned_backend_once_and_stops_device_work()
    {
        var owned = new HostAwaitableBackend();
        var owner = new AwaitableGpuTensorEngine(owned, ownsBackend: true);
        owner.Dispose();
        owner.Dispose();
        Assert.Equal(1, owned.Disposals);
        var a = Values(1, 3);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => owner.TensorAddAsync(a, a).AsTask());
        await Assert.ThrowsAsync<ObjectDisposedException>(() => owner.TensorExpAsync(a).AsTask());
        var m = Values(2, 2, 2);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => owner.TensorMatMulAsync(m, m).AsTask());

        var borrowed = new HostAwaitableBackend();
        new AwaitableGpuTensorEngine(borrowed).Dispose();
        Assert.Equal(0, borrowed.Disposals);
    }

    [Fact]
    public async Task The_factory_returns_an_engine_that_computes_whatever_the_machine_offers()
    {
        var engine = await AsyncEngine.CreateAsync();
        try
        {
            var a = Values(1, 8);
            AssertClose(new CpuEngine().TensorAdd(a, a), await engine.TensorAddAsync(a, a));
        }
        finally
        {
            (engine as IDisposable)?.Dispose();
        }
    }
}

/// <summary>An <see cref="IAwaitableGpuBackend"/> whose buffers are host arrays and whose kernels are loops.</summary>
/// <summary>
/// The per-allocation cap, which follows <see cref="GpuFallbackOptionsHolder.Current"/> (process-wide): serialized with
/// the other tests that set it.
/// </summary>
[Collection("BufferSizeCapTests")]
public sealed class AwaitableGpuTensorEngineBufferCapTests
{
    private static Tensor<float> Matrix(int rows, int columns)
        => new(Enumerable.Range(1, rows * columns).Select(i => (float)i).ToArray(), new[] { rows, columns });

    [Fact]
    public async Task An_output_past_the_cap_is_computed_by_the_cpu_and_leaves_no_buffer()
    {
        // [2,1] x [1,3]: both operands fit four elements, the six-element product does not.
        var backend = new HostAwaitableBackend { MaxBufferElements = 4 };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var cpu = new CpuEngine();
        Tensor<float> a = Matrix(2, 1), b = Matrix(1, 3);

        Assert.Equal(cpu.TensorMatMul(a, b).ToArray(), (await engine.TensorMatMulAsync(a, b)).ToArray());
        Assert.Equal(0, backend.Kernels);
        Assert.Equal(0, backend.LiveBuffers);
    }

    [Fact]
    public async Task Operands_past_the_cap_are_computed_by_the_cpu_and_fitting_ones_stay_on_the_backend()
    {
        var backend = new HostAwaitableBackend { MaxBufferElements = 4 };
        using var engine = new AwaitableGpuTensorEngine(backend);
        var cpu = new CpuEngine();
        Tensor<float> large = Matrix(2, 3), small = Matrix(2, 2);

        Assert.Equal(cpu.TensorAdd(large, large).ToArray(), (await engine.TensorAddAsync(large, large)).ToArray());
        Assert.Equal(cpu.TensorNegate(large).ToArray(), (await engine.TensorNegateAsync(large)).ToArray());
        Assert.Equal(0, backend.Kernels);
        Assert.Equal(0, backend.LiveBuffers);

        Assert.Equal(cpu.TensorAdd(small, small).ToArray(), (await engine.TensorAddAsync(small, small)).ToArray());
        Assert.Equal(cpu.TensorNegate(small).ToArray(), (await engine.TensorNegateAsync(small)).ToArray());
        Assert.Equal(2, backend.Kernels);
        Assert.Equal(0, backend.LiveBuffers);
    }

    [Fact]
    public async Task Under_the_fail_fast_policy_the_cap_reaches_the_caller()
    {
        var backend = new HostAwaitableBackend { MaxBufferElements = 4 };
        using var engine = new AwaitableGpuTensorEngine(backend);
        Tensor<float> a = Matrix(2, 1), b = Matrix(1, 3), large = Matrix(2, 3);
        var previous = GpuFallbackOptionsHolder.Current;
        try
        {
            GpuFallbackOptionsHolder.Current = new GpuFallbackOptions { ChunkingPolicy = GpuChunkingPolicy.NeverChunk_FailFast };
            await Assert.ThrowsAsync<GpuBufferTooLargeException>(() => engine.TensorMatMulAsync(a, b).AsTask());
            await Assert.ThrowsAsync<GpuBufferTooLargeException>(() => engine.TensorAddAsync(large, large).AsTask());
            await Assert.ThrowsAsync<GpuBufferTooLargeException>(() => engine.TensorNegateAsync(large).AsTask());
        }
        finally
        {
            GpuFallbackOptionsHolder.Current = previous;
        }

        Assert.Equal(0, backend.LiveBuffers);
    }
}

internal sealed class HostAwaitableBackend : IAwaitableGpuBackend
{
    private int _kernels, _allocations, _live, _disposals;

    public bool Available { get; set; } = true;

    /// <summary>Every kernel throws <see cref="NotSupportedException"/>: a capability the device lacks.</summary>
    public bool Unsupported { get; set; }

    /// <summary>Every kernel throws this instead of computing.</summary>
    public Exception? Failure { get; set; }

    /// <summary>What a division by zero yields, when set: a device may leave it indeterminate.</summary>
    public float? DivisionByZeroResult { get; set; }

    /// <summary>The most elements one buffer may hold, when set: a larger one is refused as past the device's cap.</summary>
    public int? MaxBufferElements { get; set; }

    /// <summary>Extra elements a download returns past the buffer's size, as a pooled buffer has.</summary>
    public int DownloadPadding { get; set; }

    public int Kernels => Volatile.Read(ref _kernels);
    public int Allocations => Volatile.Read(ref _allocations);
    public int LiveBuffers => Volatile.Read(ref _live);
    public int Disposals => Volatile.Read(ref _disposals);

    public bool IsAvailable => Available;
    public string BackendName => "Host";
    public TensorDevice DeviceType => TensorDevice.CPU;
    public string DeviceName => "host arrays";
    public string DeviceVendor => "test";
    public int ComputeUnits => 1;
    public long GlobalMemoryBytes => long.MaxValue;
    public long LocalMemoryBytes => 0;
    public long MaxBufferAllocBytes => int.MaxValue;
    public double TheoreticalGflops => 0;

    public IGpuBuffer AllocateBuffer(float[] data)
    {
        RequireUnderCap(data.Length);
        var copy = new float[data.Length];
        Array.Copy(data, copy, data.Length);
        return Track(copy);
    }

    public IGpuBuffer AllocateBuffer(int size)
    {
        RequireUnderCap(size);
        return Track(new float[size]);
    }

    private void RequireUnderCap(int elements)
    {
        if (MaxBufferElements is { } cap && elements > cap)
            throw new GpuBufferTooLargeException(BackendName, (long)elements * sizeof(float), (long)cap * sizeof(float), DeviceName);
    }

    public bool SupportsElementType(GpuElementType type) => type == GpuElementType.Float32;

    public IGpuBuffer AllocateBuffer(int size, GpuElementType type)
        => type == GpuElementType.Float32 ? AllocateBuffer(size) : throw new NotSupportedException(type.ToString());

    private HostBuffer Track(float[] data)
    {
        Interlocked.Increment(ref _allocations);
        Interlocked.Increment(ref _live);
        return new HostBuffer(data, () => Interlocked.Decrement(ref _live));
    }

    private static float[] Data(IGpuBuffer buffer)
        => buffer is HostBuffer host ? host.Data : throw new ArgumentException("Not a buffer of this backend.", nameof(buffer));

    public Task<float[]> DownloadBufferAsync(IGpuBuffer buffer)
    {
        var data = Data(buffer);
        var copy = new float[data.Length + DownloadPadding];
        Array.Copy(data, copy, data.Length);
        return Task.FromResult(copy);
    }

    // Completes asynchronously, so the engine's continuations run as they do after a device operation.
    private async Task Run(Action compute)
    {
        Interlocked.Increment(ref _kernels);
        await Task.Yield();
        if (Failure is { } failure) throw failure;
        if (Unsupported) throw new NotSupportedException("The host backend was told not to support this.");
        compute();
    }

    private Task Binary(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int size, Func<float, float, float> f)
        => Run(() =>
        {
            float[] x = Data(a), y = Data(b), o = Data(c);
            for (int i = 0; i < size; i++) o[i] = f(x[i], y[i]);
        });

    private Task Unary(IGpuBuffer a, IGpuBuffer b, int size, Func<float, float> f)
        => Run(() =>
        {
            float[] x = Data(a), o = Data(b);
            for (int i = 0; i < size; i++) o[i] = f(x[i]);
        });

    public Task AddAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Binary(A, B, C, size, (x, y) => x + y);
    public Task SubtractAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Binary(A, B, C, size, (x, y) => x - y);
    public Task MultiplyAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Binary(A, B, C, size, (x, y) => x * y);
    public Task DivideAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Binary(A, B, C, size, (x, y) => y == 0f && DivisionByZeroResult is { } indeterminate ? indeterminate : x / y);
    public Task ExpAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, MathF.Exp);
    public Task LogAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, MathF.Log);
    public Task SqrtAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, MathF.Sqrt);
    public Task AbsAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, MathF.Abs);
    public Task NegateAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, x => -x);
    public Task ReluAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, x => MathF.Max(x, 0f));
    public Task SigmoidAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, x => 1f / (1f + MathF.Exp(-x)));
    public Task TanhAsync(IGpuBuffer A, IGpuBuffer B, int size) => Unary(A, B, size, MathF.Tanh);

    public Task GemmAsync(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        => Run(() =>
        {
            float[] a = Data(A), b = Data(B), c = Data(C);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < K; k++) sum += a[i * K + k] * b[k * N + j];
                    c[i * N + j] = alpha * sum + beta * c[i * N + j];
                }
        });

    public void Dispose() => Interlocked.Increment(ref _disposals);

    private sealed class HostBuffer : IGpuBuffer
    {
        private Action? _released;

        public HostBuffer(float[] data, Action released)
        {
            Data = data;
            _released = released;
        }

        public float[] Data { get; }
        public int Size => Data.Length;
        public long SizeInBytes => (long)Data.Length * sizeof(float);
        public GpuElementType ElementType => GpuElementType.Float32;
        public IntPtr Handle => IntPtr.Zero;

        public void Dispose() => Interlocked.Exchange(ref _released, null)?.Invoke();
    }
}
