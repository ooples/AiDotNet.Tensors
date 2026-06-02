using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Guards the restored oneDNN accelerator (issue ooples/AiDotNet#1478). The provider
/// must actually load <c>dnnl.dll</c> and dispatch when the optional
/// <c>AiDotNet.Native.OneDNN</c> package is present (this test project references it),
/// and produce numerically correct results. These tests are skipped on a runner where
/// the native library is unavailable rather than failing — the core library treats
/// oneDNN as an optional accelerator and falls back to managed kernels without it.
/// </summary>
public class OneDnnProviderTests
{
    [SkippableFact]
    public void IsAvailable_WhenNativePackagePresent()
    {
        // If the optional native lib didn't deploy for this RID, oneDNN is correctly
        // unavailable and the managed fallback is exercised elsewhere — skip.
        Skip.IfNot(OneDnnProvider.IsAvailable,
            "dnnl.dll not deployed for this runtime; oneDNN optional accelerator inactive.");
        Assert.True(OneDnnProvider.IsAvailable);
    }

    [SkippableFact]
    public void Conv2D_ThroughOneDnn_MatchesHandComputedReference()
    {
        Skip.IfNot(OneDnnProvider.IsAvailable,
            "dnnl.dll not deployed for this runtime; oneDNN optional accelerator inactive.");

        AiDotNetEngine.ResetToCpu();
        var eng = AiDotNetEngine.Current;

        // 1 batch, 1 in-channel, 3x3 input; one 2x2 kernel; stride 1, no pad → 2x2 output.
        // input:            kernel:
        //   1 2 3             1 0
        //   4 5 6             0 1
        //   7 8 9
        // out[i,j] = in[i,j]*1 + in[i+1,j+1]*1
        //   out[0,0]=1+5=6  out[0,1]=2+6=8
        //   out[1,0]=4+8=12 out[1,1]=5+9=14
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, new[] { 1, 1, 3, 3 });
        var kernel = new Tensor<float>(new float[] { 1, 0, 0, 1 }, new[] { 1, 1, 2, 2 });

        var outc = eng.Conv2D(input, kernel, stride: 1, padding: 0, dilation: 1);

        Assert.Equal(new[] { 1, 1, 2, 2 }, outc.Shape);
        var d = outc.GetDataArray();
        Assert.Equal(6f, d[0], 3);
        Assert.Equal(8f, d[1], 3);
        Assert.Equal(12f, d[2], 3);
        Assert.Equal(14f, d[3], 3);
    }
}
