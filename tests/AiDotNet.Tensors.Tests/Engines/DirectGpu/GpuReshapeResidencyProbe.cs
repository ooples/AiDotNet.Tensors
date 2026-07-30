#if !NETFRAMEWORK
#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GpuReshapeResidencyProbe : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly bool _ready;
    public GpuReshapeResidencyProbe()
    {
        try { _gpu = new DirectGpuTensorEngine(); _ready = _gpu.IsGpuAvailable; }
        catch { _ready = false; }
    }
    public void Dispose() => _gpu?.Dispose();

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        int n = 1; foreach (int d in shape) n *= d;
        var rng = new Random(seed); var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void Reshape_OfGpuResult_StaysResident_NoDownload()
    {
        if (!_ready) return;
        var a = Rand(1, 64, 64);
        var b = Rand(2, 64, 64);
        var r = _gpu.TensorMatMul(a, b);
        var rVector = r.DataVector;
        bool rPending = AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(rVector);

        var rs = r.Reshape(4096);
        var rsVector = rs.DataVector;
        bool rsPending = AiDotNet.Tensors.Helpers.DeferredArrayMaterializer.IsPending(rsVector);

        // Emit findings so the run output answers the residency question directly.
        System.Console.WriteLine($"[PROBE] matmul-result pending(resident)={rPending}; " +
            $"reshape shares-vector={ReferenceEquals(rVector, rsVector)}; reshaped pending(resident)={rsPending}");

        Assert.True(r.IsGpuResident, "PRECONDITION: matmul result should be GPU-resident");
        Assert.True(rPending, "PRECONDITION: matmul result should be GPU-resident");
        Assert.False(rVector.GetBackingArrayUnsafe() is { Length: > 0 },
            "GPU-resident matmul result allocated logical host storage.");
        Assert.True(rs.IsGpuResident, "Reshape dropped the tensor's GPU device residency.");
        Assert.Same(rVector, rsVector);
        Assert.False(rsVector.GetBackingArrayUnsafe() is { Length: > 0 },
            "Reshape allocated logical host storage.");
        Assert.True(rsPending, "Reshape forced a host download — residency NOT preserved");
    }
}
#endif
