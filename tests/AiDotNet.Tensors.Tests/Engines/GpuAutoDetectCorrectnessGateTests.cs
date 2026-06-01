using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Guards the GPU auto-detect correctness gate. <see cref="AiDotNetEngine.AutoDetectAndConfigureGpu(bool)"/>
/// used to adopt a GPU backend whenever a device merely <c>SupportsGpu</c> (exists + constructs),
/// never verifying it actually computes correctly. On hosts where the detected backend throws or
/// returns garbage at runtime, that made EVERY downstream op crash or silently produce wrong
/// results. The gate now runs a known-answer probe and only adopts the GPU when it passes.
///
/// The invariant this pins is hardware-agnostic and holds on any CI host: <b>after auto-detect,
/// whatever engine is active must compute correctly</b> — a working GPU is adopted, a broken or
/// absent one falls back to a correct CPU engine. Before the gate, a "present but wrong" backend
/// would be adopted and fail this assertion.
/// </summary>
public class GpuAutoDetectCorrectnessGateTests
{
    [Fact]
    public void AutoDetect_LeavesActiveEngineComputingCorrectly()
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            // Adopts the GPU only if it passes the correctness probe; otherwise stays on CPU.
            AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: false);

            // [[1,2],[3,4]] @ [[5,6],[7,8]] == [[19,22],[43,50]] — exact in FP32.
            var a = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
            var b = new Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });
            var r = AiDotNetEngine.Current.TensorMatMul(a, b);

            Assert.Equal(4, r.Length);
            Assert.Equal(19f, r.GetFlatIndexValue(0), 3);
            Assert.Equal(22f, r.GetFlatIndexValue(1), 3);
            Assert.Equal(43f, r.GetFlatIndexValue(2), 3);
            Assert.Equal(50f, r.GetFlatIndexValue(3), 3);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
