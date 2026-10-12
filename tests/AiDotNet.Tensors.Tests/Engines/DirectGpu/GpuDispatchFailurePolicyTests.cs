using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1165: a GPU dispatch falls back to the CPU only when the backend lacks the capability
/// (<see cref="NotSupportedException"/>, including <see cref="GpuCapabilityUnavailableException"/>), and records that
/// fallback. A kernel fault, a device error or an allocation failure reaches the caller instead of silently becoming a
/// CPU computation.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuDispatchFailurePolicyTests
{
    private static (DirectGpuEngine DirectGpu, DirectGpuTensorEngine Engine, MockBackendState State) Create()
    {
        var state = new MockBackendState();
        var directGpu = new DirectGpuEngine(MockDirectGpuBackend.Create(state));
        return (directGpu, new DirectGpuTensorEngine(directGpu), state);
    }

    private static Tensor<float> Input() => new(new[] { -1.25f, -0.2f, 0.4f, 1.8f }, new[] { 2, 2 });

    [Fact]
    public void Tanh_RunsOnTheBackend_WhenItSucceeds()
    {
        var (directGpu, engine, state) = Create();
        using (directGpu) using (engine)
        {
            var y = engine.TensorTanh(Input());
            Assert.Equal(1, state.TanhCalls);
            Assert.Equal(Input().ToArray().Select(MathF.Tanh).ToArray(), y.ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CapabilityGap_FallsBackToTheCpu_AndIsRecorded(bool namedGap)
    {
        var (directGpu, engine, state) = Create();
        using (directGpu) using (engine)
        {
            state.ThrowOnCall["Tanh"] = namedGap
                ? new GpuCapabilityUnavailableException("Mock tanh kernel not found")
                : new NotSupportedException("Mock backend has no tanh");
            GpuLaunchProbe.Reset();
            var y = engine.TensorTanh(Input());
            float[] expected = Input().ToArray().Select(MathF.Tanh).ToArray();
            float[] actual = y.ToArray();
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 6);
            Assert.Contains(GpuLaunchProbe.Fallbacks, f => f.StartsWith("TensorTanh", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void KernelFault_Propagates()
    {
        var (directGpu, engine, state) = Create();
        using (directGpu) using (engine)
        {
            state.ThrowOnCall["Tanh"] = new InvalidOperationException("CUDA error 700: an illegal memory access");
            var ex = Assert.Throws<InvalidOperationException>(() => engine.TensorTanh(Input()));
            Assert.Contains("illegal memory access", ex.Message);
        }
    }

    [Fact]
    public void OutOfMemory_Propagates()
    {
        var (directGpu, engine, state) = Create();
        using (directGpu) using (engine)
        {
            state.ThrowOnCall["Tanh"] = new OutOfMemoryException("device out of memory");
            Assert.Throws<OutOfMemoryException>(() => engine.TensorTanh(Input()));
        }
    }
}
