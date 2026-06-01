using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity tests for the GPU fused linear + cross-entropy path (#1464): the shared
/// <see cref="RecurrenceCpuKernels"/> host references (used by the Metal/Vulkan/WebGpu fallback)
/// must match <see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T},Tensor{T},Tensor{T},Tensor{int})"/>
/// and the dense overload.
/// </summary>
public class FusedLinearCeGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.7 * (i + 1) + 1.3 * s) * scale);
        return a;
    }

    [Fact]
    public void HostReferenceIndex_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int n = 4, d = 5, vocab = 6;
        var hidden = new Tensor<float>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<float>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<float>(Gen(vocab, 3, 0.3f), new[] { vocab });
        var ids = new int[n];
        for (int r = 0; r < n; r++) ids[r] = (r * 7 + 2) % vocab;

        float reference = ((float[])(object)engine.FusedLinearCrossEntropyWithLogits(
            hidden, weight, bias, new Tensor<int>(ids, new[] { n })).GetDataArray()!)[0];
        float host = RecurrenceCpuKernels.FusedLinearCeIndex(F(hidden), F(weight), F(bias), ids, n, d, vocab);

        Assert.True(Math.Abs(reference - host) < 1e-4f, $"index loss cpu={reference} host={host}");
    }

    [Fact]
    public void HostReferenceDense_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int n = 3, d = 4, vocab = 5;
        var hidden = new Tensor<float>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<float>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<float>(Gen(vocab, 3, 0.3f), new[] { vocab });
        var target = new float[n * vocab];
        for (int r = 0; r < n; r++) target[r * vocab + ((r * 3 + 1) % vocab)] = 1.0f; // one-hot

        float reference = ((float[])(object)engine.FusedLinearCrossEntropyWithLogits(
            hidden, weight, bias, new Tensor<float>(target, new[] { n, vocab })).GetDataArray()!)[0];
        float host = RecurrenceCpuKernels.FusedLinearCeDense(F(hidden), F(weight), F(bias), target, n, d, vocab);

        Assert.True(Math.Abs(reference - host) < 1e-4f, $"dense loss cpu={reference} host={host}");
    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
