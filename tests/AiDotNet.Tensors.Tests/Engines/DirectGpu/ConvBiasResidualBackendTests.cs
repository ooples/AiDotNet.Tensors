using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class ConvBiasResidualBackendTests : IDisposable
{
    private const string Feature = "AIDOTNET_GPU_INFERENCE_CONV_BIAS_RESIDUAL";
    private readonly string? _previousFeature = Environment.GetEnvironmentVariable(Feature);
    private readonly bool _previousCudnn = TensorCodecOptions.Current.UseCudnn;
    private readonly bool _previousTf32 = CudaDispatchPolicy.AllowTF32;
    private readonly bool _previousConvTf32 = CudaDispatchPolicy.ConvReducedPrecisionTF32;

    public ConvBiasResidualBackendTests()
    {
        Environment.SetEnvironmentVariable(Feature, "1");
        TensorCodecOptions.Current.UseCudnn = true;
        CudaDispatchPolicy.AllowTF32 = false;
        CudaDispatchPolicy.ConvReducedPrecisionTF32 = false;
    }

    public void Dispose()
    {
        CudaDispatchPolicy.AllowTF32 = _previousTf32;
        CudaDispatchPolicy.ConvReducedPrecisionTF32 = _previousConvTf32;
        TensorCodecOptions.Current.UseCudnn = _previousCudnn;
        Environment.SetEnvironmentVariable(Feature, _previousFeature);
    }

    [SkippableTheory]
    [InlineData(1, 8, 33, false)]
    [InlineData(1, 8, 33, true)]
    [InlineData(1, 32, 257, false)]
    [InlineData(1, 32, 257, true)]
    [InlineData(4, 32, 257, false)]
    [InlineData(4, 32, 257, true)]
    [InlineData(4, 64, 2048, false)]
    [InlineData(4, 64, 2048, true)]
    public void Cuda_composition_preserves_inputs_and_independent_outputs(int batch, int channels, int spatial, bool activate)
    {
        using var backend = new CudaBackend();
        CudaTestDevice.Require(backend);
        Skip.IfNot(CuDnnConvolution.IsAvailable, "The vendor composition requires cuDNN.");
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var rng = new Random(73529);
        float[] Values(int count) => Enumerable.Range(0, count).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var inputValues = Values(batch * channels * spatial); var kernelValues = Values(channels * channels * 3);
        var biasValues = Values(channels); var residualValues = Values(inputValues.Length);
        using var input = new Tensor<float>(inputValues.ToArray(), new[] { batch, channels, 1, spatial });
        using var kernel = new Tensor<float>(kernelValues.ToArray(), new[] { channels, channels, 1, 3 });
        using var bias = new Tensor<float>(biasValues.ToArray(), new[] { channels });
        using var residual = new Tensor<float>(residualValues.ToArray(), input.Shape.ToArray());
        using var convolution = engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
        using var biased = engine.TensorChannelBiasAdd(convolution, bias);
        using var expected = engine.TensorAdd(biased, residual);
        using var expectedActivated = engine.LeakyReLU(expected, .1f);
        long before = backend.ConvBiasResidualVendorLaunchCount;
        Assert.True(engine.TryConvBiasResidualInference(input, kernel, bias, residual, 1, 1, 0, 1, 1, 1,
            .1f, activate, out var actual, out var activated));
        using (actual)
        using (activated)
        {
            Assert.Equal(before + 1, backend.ConvBiasResidualVendorLaunchCount);
            Close(expected.ToArray(), actual.ToArray());
            if (activate)
            {
                Assert.NotNull(activated);
                Assert.False(actual.SharesStorageWith(activated));
                var saved = activated.ToArray();
                Close(expectedActivated.ToArray(), saved);
                engine.TensorAddInPlace(actual, actual);
                Assert.Equal(saved, activated.ToArray());
            }
            else Assert.Null(activated);
            Assert.Equal(inputValues, input.ToArray()); Assert.Equal(kernelValues, kernel.ToArray());
            Assert.Equal(biasValues, bias.ToArray()); Assert.Equal(residualValues, residual.ToArray());
        }
    }

    [SkippableFact]
    public void Cuda_rejects_output_aliases_and_geometry_before_native_work()
    {
        using var backend = new CudaBackend();
        CudaTestDevice.Require(backend);
        using var input = backend.AllocateBuffer(new[] { 1f, 2f, 3f, 4f });
        using var kernel = backend.AllocateBuffer(new[] { 2f }); using var bias = backend.AllocateBuffer(new[] { .5f });
        using var residual = backend.AllocateBuffer(new[] { 4f, 3f, 2f, 1f });
        using var output = backend.AllocateBuffer(new[] { 91f, 92f, 93f, 94f });
        using var activated = backend.AllocateBuffer(new[] { 81f, 82f, 83f, 84f });
        ConvBiasResidualArgs Args(IGpuBuffer y, IGpuBuffer a, int width = 4, float alpha = .1f)
            => new(input, kernel, bias, residual, y, a, 1, 1, 1, 4, 1, 1, width, 1, 1, 1, 1, 0, 0, 1, 1, alpha, true);
        var route = (IConvBiasResidualBackend)backend; long launches = GpuLaunchProbe.Count;
        Assert.False(route.TryConvBiasResidual(Args(input, activated)));
        Assert.False(route.TryConvBiasResidual(Args(residual, activated)));
        Assert.False(route.TryConvBiasResidual(Args(output, output)));
        Assert.False(route.TryConvBiasResidual(Args(output, input)));
        Assert.False(route.TryConvBiasResidual(Args(output, activated, 3)));
        Assert.False(route.TryConvBiasResidual(Args(output, activated, alpha: float.NaN)));
        Assert.Equal(launches, GpuLaunchProbe.Count);
        Assert.Equal(new[] { 91f, 92f, 93f, 94f }, backend.DownloadBuffer(output));
        Assert.Equal(new[] { 81f, 82f, 83f, 84f }, backend.DownloadBuffer(activated));
    }

    [Fact]
    public void Overflowing_geometry_is_rejected_without_wrapping()
    {
        using var buffer = new FakeBuffer();
        var args = new ConvBiasResidualArgs(buffer, buffer, buffer, buffer, buffer, buffer,
            int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue,
            1, 1, 1, 1, 0, 0, 1, 1, .1f, true);
        Assert.False(args.TryCount(out _));
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Shared_composition_matches_the_vendor_route_and_reuses_scratch(bool activate)
    {
        using var backend = new CudaBackend();
        CudaTestDevice.Require(backend);
        Skip.IfNot(CuDnnConvolution.IsAvailable, "The vendor composition requires cuDNN.");
        using var input = backend.AllocateBuffer(new[] { 1f, -2f, 3f, -4f });
        using var kernel = backend.AllocateBuffer(new[] { 2f }); using var bias = backend.AllocateBuffer(new[] { .5f });
        using var residual = backend.AllocateBuffer(new[] { 4f, 3f, -2f, 1f });
        using var output = backend.AllocateBuffer(4); using var activated = backend.AllocateBuffer(4);
        var args = new ConvBiasResidualArgs(input, kernel, bias, residual, output, activated,
            1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 0, 0, 1, 1, .1f, activate);
        long vendor = backend.ConvBiasResidualVendorLaunchCount;
        for (int warm = 0; warm < 2; warm++) Assert.True(ConvBiasResidualComposition.TryExecute(backend, args));
        var warmed = GpuMemoryStats.Snapshot(backend);
        for (int repeat = 0; repeat < 6; repeat++) Assert.True(ConvBiasResidualComposition.TryExecute(backend, args));
        var after = GpuMemoryStats.Snapshot(backend);
        Assert.Equal(warmed.DeviceAllocations, after.DeviceAllocations);
        Assert.Equal(warmed.DeviceFrees, after.DeviceFrees);
        Assert.Equal(vendor, backend.ConvBiasResidualVendorLaunchCount);
        var composed = backend.DownloadBuffer(output);
        var composedActivated = activate ? backend.DownloadBuffer(activated) : null;
        Assert.Equal(new[] { 6.5f, -.5f, 4.5f, -6.5f }, composed);
        Assert.True(((IConvBiasResidualBackend)backend).TryConvBiasResidual(args));
        Assert.Equal(vendor + 1, backend.ConvBiasResidualVendorLaunchCount);
        Close(composed, backend.DownloadBuffer(output));
        if (activate) Close(composedActivated!, backend.DownloadBuffer(activated));

        // Convolution precision is configured separately from GEMM. The strict
        // cuDNN epilogue must defer to the configured convolution in this case.
        CudaDispatchPolicy.ConvReducedPrecisionTF32 = true;
        Assert.True(((IConvBiasResidualBackend)backend).TryConvBiasResidual(args));
        Assert.Equal(vendor + 1, backend.ConvBiasResidualVendorLaunchCount);
        Close(composed, backend.DownloadBuffer(output));
    }

    [Fact]
    public void Six_backend_types_implement_the_composition_contract()
    {
        Type[] types = { typeof(CudaBackend), typeof(AiDotNet.Tensors.Engines.DirectGpu.HIP.HipBackend), typeof(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend),
            typeof(AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalBackend), typeof(AiDotNet.Tensors.Engines.DirectGpu.Vulkan.VulkanBackend), typeof(AiDotNet.Tensors.Engines.DirectGpu.WebGpu.WebGpuBackend) };
        foreach (Type type in types) Assert.True(typeof(IConvBiasResidualBackend).IsAssignableFrom(type), type.FullName);
    }

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 1e-6f + 1e-5f * Math.Abs(expected[i]),
                $"Output mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
    }

    private sealed class FakeBuffer : IGpuBuffer
    {
        public int Size => int.MaxValue;
        public long SizeInBytes => (long)int.MaxValue * sizeof(float);
        public IntPtr Handle => new IntPtr(1);
        public void Dispose() { }
    }
}
