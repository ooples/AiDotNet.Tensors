#if NET6_0_OR_GREATER
using System;
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

/// <summary>
/// A GPU training trace's node outputs defer their host storage, and a device-resident value written into one is a
/// device-to-device copy into a buffer the output owns, never a host array. These run that write on each backend:
/// the output must stay without a host array until read, and a host read must return the source values. The same
/// holds for an <c>*Into</c> op (binary, permute) whose destination is such an output.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeferredHostDeviceCopyBackendTests
{
    private static void Check(IDirectGpuBackend backend)
    {
        var previous = AiDotNetEngine.Current;
        using var engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend, borrowsBackend: true));
        try
        {
            AiDotNetEngine.Current = engine;
            var values = new float[3 * 257];
            for (int i = 0; i < values.Length; i++) values[i] = (i % 17) * 0.25f - 2f;
            var src = new Tensor<float>(values, [3, 257]);
            src.Gpu();
            Assert.NotNull(src._gpuBuffer);

            var dest = Tensor<float>.CreateHostDeferred([3, 257]);
            DirectGpuTensorEngine.CopyResultInto(engine, src, dest);

            Assert.True(dest.HasDeferredHostStorage, "the write allocated a host array for the deferred output");
            Assert.NotNull(dest._gpuBuffer);
            Assert.False(ReferenceEquals(dest._gpuBuffer, src._gpuBuffer), "the output must own a copy, not borrow src");
            Assert.True(dest._ownsGpuBuffer);

            var host = dest.ToArray();
            Assert.False(dest.HasDeferredHostStorage, "a host read must allocate the host array");
            Assert.Equal(values, host);

            // *Into ops write a deferred output by binding their device result to it.
            var other = new Tensor<float>(Array.ConvertAll(values, v => v * 0.5f + 1f), [3, 257]);
            other.Gpu();
            var sum = Tensor<float>.CreateHostDeferred([3, 257]);
            ((IEngine)engine).TensorAddInto(sum, src, other);
            Assert.True(sum.HasDeferredHostStorage, "TensorAddInto allocated a host array for the deferred output");
            var sumHost = sum.ToArray();
            for (int i = 0; i < values.Length; i++) Assert.Equal(values[i] + (values[i] * 0.5f + 1f), sumHost[i], 5);

            var transposed = Tensor<float>.CreateHostDeferred([257, 3]);
            engine.TensorPermuteInto(transposed, src, [1, 0]);
            Assert.True(transposed.HasDeferredHostStorage, "TensorPermuteInto allocated a host array for the deferred output");
            var tHost = transposed.ToArray();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 257; c++)
                    Assert.Equal(values[r * 257 + c], tHost[c * 3 + r]);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void Cuda_DeviceCopyIntoDeferredHostOutput()
    {
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_DeviceCopyIntoDeferredHostOutput()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void OpenCl_DeviceCopyIntoDeferredHostOutput()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_DeviceCopyIntoDeferredHostOutput()
    {
        var backend = VulkanBackend.Instance;   // process-wide; not disposed
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Metal_DeviceCopyIntoDeferredHostOutput()
    {
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsIOS(), "Metal is only available on macOS/iOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_DeviceCopyIntoDeferredHostOutput()
        => Check(DawnHostedWebGpu.BackendOrSkip());   // the real backend on Dawn; shared, so not disposed
#endif
}
#endif
