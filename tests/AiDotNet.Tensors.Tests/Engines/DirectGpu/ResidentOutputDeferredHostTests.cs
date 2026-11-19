using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ResidentOutputDeferredHostTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public ResidentOutputDeferredHostTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private DirectGpuTensorEngine RequireGpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend is available.");
        return _fixture.Engine ?? throw new InvalidOperationException("GPU fixture has no engine.");
    }

    private static void Resident(DirectGpuTensorEngine gpu, bool pooled, Action body)
    {
        var pool = pooled ? gpu.CreateScratchPool() : null;
        var previous = DirectGpuTensorEngine.EnterScratchPool(pool);
        gpu.SuspendActivationEviction();
        try
        {
            using (gpu.EnterCompiledCapturePath()) body();
        }
        finally
        {
            DirectGpuTensorEngine.RestoreScratchPool(previous);
            pool?.Release();
            gpu.ResumeActivationEviction();
        }
    }

    private static Tensor<float>[] Chain(DirectGpuTensorEngine gpu, Tensor<float> a, Tensor<float> b, Tensor<float> bias)
    {
        var product = gpu.TensorMultiply(a, b);
        var scaled = gpu.TensorMultiplyScalar(product, 0.75f);
        var shifted = gpu.TensorBroadcastAdd(scaled, bias);
        return new[] { product, scaled, shifted, gpu.ReluBackward(shifted, a) };
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Resident_chain_preserves_device_values_host_reads_and_scope_ownership(bool pooled)
    {
        var gpu = RequireGpu();
        var cpu = new CpuEngine();
        var a = new Tensor<float>(new float[32], new[] { 2, 2, 1, 8 });
        var b = new Tensor<float>(new float[32], a.Shape.ToArray());
        var bias = new Tensor<float>(new[] { -0.25f, 1.5f }, new[] { 1, 2, 1, 1 });
        IGpuBuffer? previousOutput = null;
        Resident(gpu, pooled, () =>
        {
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < a.Length; i++)
                {
                    a[i] = (i % 7 - 3) * 0.4f + step * 0.1f;
                    b[i] = (i % 5 - 2) * 0.3f - step * 0.2f;
                }
                var expected = cpu.ReluBackward(cpu.TensorBroadcastAdd(
                    cpu.TensorMultiplyScalar(cpu.TensorMultiply(a, b), 0.75f), bias), a).ToArray();
                gpu.EnsureResidentInput(a);
                gpu.EnsureResidentInput(b);
                gpu.EnsureResidentInput(bias);
                gpu.SetCurrentScratchAction(0);
                Tensor<float>[] results;
                IGpuBuffer dropped;
                using (var scope = TensorScope.Begin())
                {
                    results = Chain(gpu, a, b, bias);
                    foreach (var result in results)
                    {
                        Assert.True(result.HasDeferredHostStorage, "resident output allocated an unused host payload");
                        Assert.Equal(a.Shape.ToArray(), result.Shape.ToArray());
                        Assert.Equal(a.Length, result.Length);
                        Assert.True(result.IsContiguous);
                        Assert.Equal(!pooled, result._ownsGpuBuffer);
                        Assert.NotNull(result._gpuBuffer);
                    }
                    dropped = results[0]._gpuBuffer!;
                    scope.Keep(results[3]);
                }
                Assert.Equal(pooled, dropped.Handle != IntPtr.Zero);
                if (pooled && previousOutput is not null) Assert.Same(previousOutput, results[3]._gpuBuffer);
                previousOutput = results[3]._gpuBuffer;
                var actual = results[3].ToArray(); // First host read, after the scope has released its intermediates.
                Assert.False(results[3].HasDeferredHostStorage);
                for (int i = 0; i < actual.Length; i++) Assert.Equal(expected[i], actual[i], 5);
                foreach (var result in results) result.Dispose();
            }
        });
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Resident_single_matmul_gradient_defers_host_storage_and_matches_scalar_reference(bool leftOperand)
    {
        var gpu = RequireGpu();
        const int m = 4, k = 3, n = 5;
        var a = new Tensor<float>(new[] { 1f, -2, 3, -4, 5, 6, 7, 8, -9, 10, -11, 12 }, new[] { m, k });
        var b = new Tensor<float>(new[] { 1f, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12, 13, -14, 15 }, new[] { k, n });
        var g = new Tensor<float>(new float[m * n], new[] { m, n });
        for (int i = 0; i < g.Length; i++) g[i] = (i % 7 - 3) * 0.125f;
        gpu.EnsureResidentInput(a);
        gpu.EnsureResidentInput(b);
        gpu.EnsureResidentInput(g);
        Resident(gpu, true, () =>
        {
            for (int step = 0; step < 2; step++)
            {
                gpu.SetCurrentScratchAction(0);
                Assert.True(gpu.TryMatMulSingleGradient2D(g, a, b, leftOperand, out var result),
                    "The available backend must execute the selected resident gradient GEMM.");
                Assert.NotNull(result);
                Assert.True(result!.HasDeferredHostStorage);
                int rows = leftOperand ? m : k, cols = leftOperand ? k : n;
                Assert.Equal(new[] { rows, cols }, result.Shape.ToArray());
                var actual = result.ToArray();
                for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++)
                {
                    double expected = 0;
                    for (int q = 0; q < (leftOperand ? n : m); q++)
                        expected += leftOperand ? (double)g[i * n + q] * b[j * n + q]
                            : (double)a[q * k + i] * g[q * n + j];
                    Assert.True(Math.Abs(actual[i * cols + j] - expected) < 1e-5);
                }
                result.Dispose();
            }
        });
    }

    [SkippableFact]
    public void Warm_resident_backward_chain_allocates_metadata_instead_of_output_sized_host_arrays()
    {
        var gpu = RequireGpu();
        const int length = 1 << 18, repeats = 4;
        var a = new Tensor<float>(new float[length], new[] { 1, 2, 1, length / 2 });
        var b = new Tensor<float>(new float[length], a.Shape.ToArray());
        a.AsWritableSpan().Fill(2f);
        b.AsWritableSpan().Fill(3f);
        var bias = new Tensor<float>(new[] { 0.5f, -1f }, new[] { 1, 2, 1, 1 });
        gpu.EnsureResidentInput(a);
        gpu.EnsureResidentInput(b);
        gpu.EnsureResidentInput(bias);
        Resident(gpu, true, () =>
        {
            gpu.SetCurrentScratchAction(0);
            foreach (var result in Chain(gpu, a, b, bias)) result.Dispose();
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int step = 0; step < repeats; step++)
            {
                gpu.SetCurrentScratchAction(0);
                foreach (var result in Chain(gpu, a, b, bias))
                {
                    Assert.True(result.HasDeferredHostStorage);
                    result.Dispose();
                }
            }
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            // The old four output arrays alone cost 16 MiB here. Allow metadata/runtime bookkeeping up to one
            // output payload (1 MiB), keeping the assertion tied to the removed storage rather than an exact count.
            Assert.True(allocated < (long)length * sizeof(float), $"resident metadata allocated {allocated:N0} bytes");
        });
    }
}
