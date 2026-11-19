#if NET6_0_OR_GREATER
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class RfftMagnitudeBackendParityTests
{
    private static void Verify(IDirectGpuBackend backend, bool accumulate)
    {
        const int rows = 3, n = 16, bins = 9, count = rows * n;
        const float epsilon = .125f;
        var native = Assert.IsAssignableFrom<IRfftMagnitudeKernels>(backend);
        var values = Enumerable.Range(0, count).Select(i => i < n ? 0f : (i % 11 - 5) * .125f).ToArray();
        var weights = Enumerable.Range(0, rows * bins).Select(i => (i % 7 - 3) * .0625f).ToArray();
        using var input = backend.AllocateBuffer(values); using var zero = backend.AllocateBuffer(count);
        using var re = backend.AllocateBuffer(count); using var im = backend.AllocateBuffer(count);
        using var spectrum = backend.AllocateBuffer(count * 2); using var seed = backend.AllocateBuffer(count * 2);
        using var upstream = backend.AllocateBuffer(weights); using var magnitude = backend.AllocateBuffer(rows * bins);
        using var gradient = backend.AllocateBuffer(Enumerable.Repeat(.375f, count).ToArray());
        backend.Fill(zero, 0, count);
        long before = GpuLaunchProbe.ReadbackBytes;
        backend.BatchedFFT(input, zero, re, im, rows, n, inverse: false);
        backend.StridedScatter(re, spectrum, 0, 2, count); backend.StridedScatter(im, spectrum, 1, 2, count);
        native.RfftMagnitude(spectrum, magnitude, rows, n, n, epsilon);
        native.RfftMagnitudeSeed(spectrum, upstream, seed, rows, n, n, epsilon);
        backend.StridedGather(seed, input, 0, 2, count); backend.StridedGather(seed, zero, 1, 2, count);
        backend.BatchedFFT(input, zero, re, im, rows, n, inverse: true);
        backend.StridedScatter(re, spectrum, 0, 2, count); backend.StridedScatter(im, spectrum, 1, 2, count);
        native.RfftRealGradient(spectrum, gradient, count, n, accumulate);
        Assert.Equal(before, GpuLaunchProbe.ReadbackBytes);
        var cpu = new CpuEngine(); var source = new Tensor<float>(values, new[] { rows, n });
        var expected = new Tensor<float>(Enumerable.Repeat(.375f, count).ToArray(), new[] { rows, n });
        cpu.TensorRfftMagnitudeBackwardInto(new Tensor<float>(weights, new[] { rows, bins }), source, expected, epsilon, accumulate);
        Close(cpu.TensorRfftMagnitude(source, epsilon).ToArray(), backend.DownloadBuffer(magnitude));
        Close(expected.ToArray(), backend.DownloadBuffer(gradient));
    }
    private static void Close(float[] expected, float[] actual)
    {
        Assert.True(actual.Length >= expected.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-4 * (1 + Math.Abs(expected[i])),
                $"[{i}] actual={actual[i]:G9} expected={expected[i]:G9}");
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Cuda_native_magnitude_and_adjoint(bool accumulate)
    { using var b = new CudaBackend(); Skip.IfNot(b.IsAvailable, "CUDA unavailable."); Verify(b, accumulate); }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Hip_native_magnitude_and_adjoint(bool accumulate)
    { using var b = new HipBackend(); Skip.IfNot(b.IsAvailable, "HIP unavailable."); Verify(b, accumulate); }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void OpenCl_native_magnitude_and_adjoint(bool accumulate)
    { using var b = new OpenClBackend(); Skip.IfNot(b.IsAvailable, "OpenCL unavailable."); Verify(b, accumulate); }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Metal_native_magnitude_and_adjoint(bool accumulate)
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal unavailable on this OS.");
        using var b = new MetalBackend(); Skip.IfNot(b.IsAvailable, "Metal unavailable."); Verify(b, accumulate);
    }
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Vulkan_native_magnitude_and_adjoint(bool accumulate)
    { var b = VulkanBackend.Instance; Skip.IfNot(b.Initialize() && b.IsGlslCompilerAvailable, "Vulkan unavailable."); Verify(b, accumulate); }
#if NET7_0_OR_GREATER
    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void WebGpu_native_magnitude_and_adjoint(bool accumulate)
        => Verify(DawnHostedWebGpu.BackendOrSkip(), accumulate);
#endif
}
#endif
