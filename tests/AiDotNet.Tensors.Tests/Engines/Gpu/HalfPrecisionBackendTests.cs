using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Issue #337: validates the FP16 GEMM capability interface that
/// AutocastScope / mixed-precision training paths dispatch through.
/// CudaBackend implements IGpuHalfPrecisionBackend on Maxwell+ hardware.
/// </summary>
public class HalfPrecisionBackendTests
{
    [Fact]
    public void CudaBackend_ImplementsHalfPrecisionInterface_WhenAvailable()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is null)
            return; // CPU-only host

        // The interface is optional — backends that don't ship HGEMM
        // simply don't implement it. CudaBackend should.
        var hp = backend as IGpuHalfPrecisionBackend;
        if (hp is null)
            return; // backend other than CUDA, no fp16 support

        // SupportsHgemm true iff compute capability >= 5.0
        Assert.True(hp.SupportsHgemm,
            "CUDA backend should support HGEMM on any Maxwell+ device " +
            "(every NVIDIA card 2014+). If this fires, the device probe " +
            "is wrong or the host has a pre-Maxwell card.");
    }

    [Fact]
    public void Hgemm_NullArgs_Throw()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is not IGpuHalfPrecisionBackend hp || !hp.SupportsHgemm)
            return;

        // Validate the no-null contract — caller errors surface as
        // ArgumentNullException, not as cryptic native-layer crashes.
        Assert.Throws<System.ArgumentNullException>(() =>
            hp.Hgemm(null!, null!, null!, m: 4, n: 4, k: 4));
    }

    [Fact]
    public void Hgemm_NonPositiveDim_Throws()
    {
        var engine = new DirectGpuTensorEngine();
        var backend = engine.GetBackend();
        if (backend is not IGpuHalfPrecisionBackend hp || !hp.SupportsHgemm)
            return;

        // We need real buffers to reach the m/n/k check; if the backend
        // can't allocate, skip.
        using var a = backend.AllocateBuffer(4);
        using var b = backend.AllocateBuffer(4);
        using var c = backend.AllocateBuffer(4);
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            hp.Hgemm(a, b, c, m: 0, n: 4, k: 4));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            hp.Hgemm(a, b, c, m: 4, n: -1, k: 4));
    }
}
