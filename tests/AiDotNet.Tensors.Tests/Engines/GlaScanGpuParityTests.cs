using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity tests for the GPU GLA scan path (issue ooples/AiDotNet#1464). The HIP/CUDA/OpenCL
/// backends ship native kernels; Metal/Vulkan/WebGpu use the shared
/// <see cref="RecurrenceCpuKernels"/> host reference. These tests verify that shared reference
/// matches the differentiable <see cref="CpuEngine.GlaScanForward{T}"/> exactly (forward output
/// and BPTT gradients), so GPU-vs-CPU parity holds on the fallback backends. The native kernels
/// mirror the same math and are validated on GPU hardware separately.
/// </summary>
public class GlaScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 1.3 * s) * scale);
        return a;
    }

    private static float[] GenGate(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.3 * s))));
        return a;
    }

    [Fact]
    public void HostReferenceForward_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int batch = 2, seqLen = 5, modelDim = 6, numHeads = 2;
        var shape = new[] { batch, seqLen, modelDim };
        var gShape = new[] { batch, seqLen, numHeads };
        var q = new Tensor<float>(Gen(batch * seqLen * modelDim, 1), shape);
        var k = new Tensor<float>(Gen(batch * seqLen * modelDim, 2), shape);
        var v = new Tensor<float>(Gen(batch * seqLen * modelDim, 3), shape);
        var g = new Tensor<float>(GenGate(batch * seqLen * numHeads, 4), gShape);

        var reference = (float[])(object)engine.GlaScanForward(q, k, v, g, numHeads).GetDataArray()!;

        var hostOut = new float[batch * seqLen * modelDim];
        RecurrenceCpuKernels.GlaForward(
            (float[])(object)q.GetDataArray()!, (float[])(object)k.GetDataArray()!,
            (float[])(object)v.GetDataArray()!, (float[])(object)g.GetDataArray()!,
            hostOut, batch, seqLen, modelDim, numHeads, headDim: modelDim / numHeads);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-5f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    [Fact]
    public void HostReferenceBackward_MatchesCpuEngineTapeGradients()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4, numHeads = 2;
        var shape = new[] { batch, seqLen, modelDim };
        var gShape = new[] { batch, seqLen, numHeads };
        var q = new Tensor<float>(Gen(batch * seqLen * modelDim, 5), shape);
        var k = new Tensor<float>(Gen(batch * seqLen * modelDim, 6), shape);
        var v = new Tensor<float>(Gen(batch * seqLen * modelDim, 7), shape);
        var g = new Tensor<float>(GenGate(batch * seqLen * numHeads, 8), gShape);

        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var outp = engine.GlaScanForward(q, k, v, g, numHeads);
            grads = tape.ComputeGradients(outp, new[] { q, k, v, g });
        }

        // Tape seeds dOutput = ones (gradient of sum(output)).
        var dOut = new float[batch * seqLen * modelDim];
        for (int i = 0; i < dOut.Length; i++) dOut[i] = 1.0f;

        var dq = new float[batch * seqLen * modelDim];
        var dk = new float[batch * seqLen * modelDim];
        var dv = new float[batch * seqLen * modelDim];
        var dG = new float[batch * seqLen * numHeads];
        RecurrenceCpuKernels.GlaBackward(
            dOut, (float[])(object)q.GetDataArray()!, (float[])(object)k.GetDataArray()!,
            (float[])(object)v.GetDataArray()!, (float[])(object)g.GetDataArray()!,
            dq, dk, dv, dG, batch, seqLen, modelDim, numHeads, headDim: modelDim / numHeads);

        AssertMatch(grads[q], dq, "dQ");
        AssertMatch(grads[k], dk, "dK");
        AssertMatch(grads[v], dv, "dV");
        AssertMatch(grads[g], dG, "dG");
    }

    private static void AssertMatch(Tensor<float> expected, float[] actual, string name)
    {
        var e = (float[])(object)expected.GetDataArray()!;
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) < 1e-4f,
                $"{name}[{i}] tape={e[i]} host={actual[i]}");
    }
}
