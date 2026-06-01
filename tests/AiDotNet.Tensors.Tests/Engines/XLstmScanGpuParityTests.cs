using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity tests for the GPU xLSTM scan path (#1464): the shared
/// <see cref="RecurrenceCpuKernels"/> host reference (used by the Metal/Vulkan/WebGpu fallback)
/// must match the differentiable <see cref="CpuEngine.XLstmScanForward{T}"/> exactly.
/// </summary>
public class XLstmScanGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.35f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.5 * (i + 1) + 1.5 * s) * scale);
        return a;
    }
    private static float[] GenExp(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)Math.Exp(0.3 * Math.Sin(0.5 * (i + 1) + 1.5 * s));
        return a;
    }
    private static float[] GenSig(int n, int s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(1.0 / (1.0 + Math.Exp(-Math.Sin(0.5 * (i + 1) + 1.5 * s))));
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
        var ig = new Tensor<float>(GenExp(batch * seqLen * numHeads, 4), gShape);
        var fg = new Tensor<float>(GenSig(batch * seqLen * numHeads, 5), gShape);
        var og = new Tensor<float>(GenSig(batch * seqLen * numHeads, 6), gShape);

        var reference = (float[])(object)engine.XLstmScanForward(q, k, v, ig, fg, og, numHeads).GetDataArray()!;
        var hostOut = new float[batch * seqLen * modelDim];
        RecurrenceCpuKernels.XLstmForward(
            F(q), F(k), F(v), F(ig), F(fg), F(og), hostOut,
            batch, seqLen, modelDim, numHeads, modelDim / numHeads);

        for (int i = 0; i < reference.Length; i++)
            Assert.True(Math.Abs(reference[i] - hostOut[i]) < 1e-4f,
                $"forward[{i}] cpu={reference[i]} host={hostOut[i]}");
    }

    [Fact]
    public void HostReferenceBackward_MatchesCpuEngineTapeGradients()
    {
        var engine = new CpuEngine();
        int batch = 1, seqLen = 4, modelDim = 4, numHeads = 2;
        var shape = new[] { batch, seqLen, modelDim };
        var gShape = new[] { batch, seqLen, numHeads };
        var q = new Tensor<float>(Gen(batch * seqLen * modelDim, 1), shape);
        var k = new Tensor<float>(Gen(batch * seqLen * modelDim, 2), shape);
        var v = new Tensor<float>(Gen(batch * seqLen * modelDim, 3), shape);
        var ig = new Tensor<float>(GenExp(batch * seqLen * numHeads, 4), gShape);
        var fg = new Tensor<float>(GenSig(batch * seqLen * numHeads, 5), gShape);
        var og = new Tensor<float>(GenSig(batch * seqLen * numHeads, 6), gShape);

        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var outp = engine.XLstmScanForward(q, k, v, ig, fg, og, numHeads);
            grads = tape.ComputeGradients(outp, new[] { q, k, v, ig, fg, og });
        }

        var dOut = new float[batch * seqLen * modelDim];
        for (int i = 0; i < dOut.Length; i++) dOut[i] = 1.0f;
        var dq = new float[batch * seqLen * modelDim];
        var dk = new float[batch * seqLen * modelDim];
        var dv = new float[batch * seqLen * modelDim];
        var di = new float[batch * seqLen * numHeads];
        var df = new float[batch * seqLen * numHeads];
        var dO = new float[batch * seqLen * numHeads];
        RecurrenceCpuKernels.XLstmBackward(
            dOut, F(q), F(k), F(v), F(ig), F(fg), F(og), dq, dk, dv, di, df, dO,
            batch, seqLen, modelDim, numHeads, modelDim / numHeads);

        Match(grads[q], dq, "dQ"); Match(grads[k], dk, "dK"); Match(grads[v], dv, "dV");
        Match(grads[ig], di, "dI"); Match(grads[fg], df, "dF"); Match(grads[og], dO, "dO");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;

    private static void Match(Tensor<float> expected, float[] actual, string name)
    {
        var e = (float[])(object)expected.GetDataArray()!;
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) < 1e-3f, $"{name}[{i}] tape={e[i]} host={actual[i]}");
    }
}
