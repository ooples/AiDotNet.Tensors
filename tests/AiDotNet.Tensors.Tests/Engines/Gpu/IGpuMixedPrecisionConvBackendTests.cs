using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: validates the <see cref="IGpuMixedPrecisionConvBackend"/>
/// capability surface. CudaBackend should implement it when cuDNN is
/// available; consumers downcast a backend reference and dispatch through
/// it for mixed-precision conv.
/// </summary>
public class IGpuMixedPrecisionConvBackendTests
{
    [Fact]
    public void CudaBackend_ImplementsMixedPrecisionConvInterface_WhenAvailable()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is null)
            return; // CPU-only host — interface unreachable

        // Prefer the IDirectGpuBackend.MixedPrecisionConv accessor (the
        // canonical engine-extension discovery path) but also probe the
        // downcast — both must point to the same instance on CUDA.
        var mp = backend.MixedPrecisionConv;
        bool isCudaBackend = backend.GetType().Name.Contains("Cuda", System.StringComparison.Ordinal);
        if (isCudaBackend)
        {
            Assert.NotNull(mp);
            Assert.Same(backend, mp); // CUDA backend returns `this` from MixedPrecisionConv
        }
        else if (mp is null)
        {
            return; // Truly non-CUDA backend; interface is CUDA-only today.
        }

        // SupportsHalfConv tracks UseCudnnForConv; not strictly required to
        // be true (the policy may be disabled), but if cuDNN is on this
        // backend then half conv is available.
        Assert.True(mp!.SupportsHalfConv || !AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaDispatchPolicy.UseCudnnForConv,
            "When UseCudnnForConv is true, SupportsHalfConv must also be true (cuDNN supports fp16 conv on every release).");
    }

    [Fact]
    public void Conv2DMixed_NullBuffer_Throws()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is not IGpuMixedPrecisionConvBackend mp || !mp.SupportsHalfConv)
            return;

        // Validates the null-check contract — caller errors surface as
        // ArgumentNullException, not as cryptic native-layer crashes.
        Assert.Throws<System.ArgumentNullException>(() =>
            mp.Conv2DMixed(
                input: null!, kernel: null!, output: null!,
                batch: 1, inChannels: 1, inHeight: 8, inWidth: 8,
                outChannels: 1, outHeight: 6, outWidth: 6,
                kernelH: 3, kernelW: 3,
                strideH: 1, strideW: 1, padH: 0, padW: 0,
                dilationH: 1, dilationW: 1,
                dataType: GpuMixedPrecisionDataType.Half));
    }

    [Fact]
    public void BFloat16Conv_OnPreAmpereHost_RejectsWithClearMessage()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is not IGpuMixedPrecisionConvBackend mp || !mp.SupportsHalfConv)
            return;

        // If this host advertises BF16 conv support, we can't exercise the
        // rejection path — skip rather than fail.
        if (mp.SupportsBFloat16Conv)
            return;

        using var input = backend.AllocateBuffer(64);
        using var kernel = backend.AllocateBuffer(9);
        using var output = backend.AllocateBuffer(36);

        // The throw is NotSupportedException with the device's actual
        // compute capability in the message — that's how a user diagnoses
        // why BF16 didn't engage on their hardware.
        var ex = Assert.Throws<System.NotSupportedException>(() =>
            mp.Conv2DMixed(
                input, kernel, output,
                batch: 1, inChannels: 1, inHeight: 8, inWidth: 8,
                outChannels: 1, outHeight: 6, outWidth: 6,
                kernelH: 3, kernelW: 3,
                strideH: 1, strideW: 1, padH: 0, padW: 0,
                dilationH: 1, dilationW: 1,
                dataType: GpuMixedPrecisionDataType.BFloat16));
        Assert.Contains("8.0", ex.Message, System.StringComparison.Ordinal);
    }
}
