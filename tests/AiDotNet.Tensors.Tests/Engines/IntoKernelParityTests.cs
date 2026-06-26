using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity tests for the destination-buffer (<c>*Into</c>) engine kernels added for the
/// #1672 diffusion-inference resident-scratch path (SDPA, FusedLinear, broadcast add/multiply).
/// Each kernel writes the SAME math into a caller-provided buffer that the allocating overload
/// returns, so the invariant under test is: <c>op*Into(dest, ...) ≡ opAllocating(...)</c>,
/// numerically identical. These also cover the stride-aware SDPA path (non-contiguous Q/K/V)
/// and the <see cref="TensorBase{T}"/> strided-read backing accessor.
/// </summary>
public class IntoKernelParityTests
{
    private readonly CpuEngine E = new();
    private const float Tol = 1e-5f;

    private Tensor<float> R(int[] s, int seed)
    {
        var r = new Random(seed);
        var d = new float[s.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(r.NextDouble() * 2 - 1);
        return new Tensor<float>(d, s);
    }

    private void AE(Tensor<float> a, Tensor<float> b, string m = "")
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        var ad = a.GetDataArray();
        var bd = b.GetDataArray();
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(ad[i] - bd[i]) <= Tol, $"{m} [{i}]: {ad[i]} vs {bd[i]}");
    }

    // ---- ScaledDotProductAttentionInto: contiguous [B,H,S,D] ----
    [Fact]
    public void SdpaInto_Contiguous_MatchesAllocating()
    {
        int b = 2, h = 4, s = 6, d = 8;
        var q = R([b, h, s, d], 1);
        var k = R([b, h, s, d], 2);
        var v = R([b, h, s, d], 3);
        double scale = 0.1875;

        var expected = E.ScaledDotProductAttention(q, k, v, mask: null, scale, out _);
        var dest = new Tensor<float>([b, h, s, d]);
        E.ScaledDotProductAttentionInto(dest, q, k, v, scale);
        AE(dest, expected, "SDPA-into contiguous");
    }

    // ---- ScaledDotProductAttentionInto: stride-aware (permuted, non-contiguous) ----
    [Fact]
    public void SdpaInto_StridedPermutedInputs_MatchesAllocating()
    {
        // Build [B,S,H,D] then permute to [B,H,S,D] — the exact reshape+permute the DiT
        // attention produces. This exercises ResolveStridedAttnOperand / strided backing read.
        int b = 2, h = 3, s = 5, d = 8;
        var qp = R([b, s, h, d], 11);
        var kp = R([b, s, h, d], 12);
        var vp = R([b, s, h, d], 13);
        var q = E.TensorPermute(qp, [0, 2, 1, 3]);
        var k = E.TensorPermute(kp, [0, 2, 1, 3]);
        var v = E.TensorPermute(vp, [0, 2, 1, 3]);
        double scale = 0.125;

        var expected = E.ScaledDotProductAttention(q, k, v, mask: null, scale, out _);
        var dest = new Tensor<float>([b, h, s, d]);
        E.ScaledDotProductAttentionInto(dest, q, k, v, scale);
        AE(dest, expected, "SDPA-into strided");
    }

    // ---- FusedLinearInto: with bias, no activation ----
    [Fact]
    public void FusedLinearInto_BiasNoActivation_MatchesAllocating()
    {
        int m = 7, kdim = 12, n = 9;
        var input = R([m, kdim], 21);
        var w = R([kdim, n], 22);
        var bias = R([n], 23);

        var expected = E.FusedLinear(input, w, bias, FusedActivationType.None);
        var dest = new Tensor<float>([m, n]);
        E.FusedLinearInto(dest, input, w, bias, FusedActivationType.None);
        AE(dest, expected, "FusedLinear-into bias/none");
    }

    // ---- FusedLinearInto: ReLU activation ----
    [Fact]
    public void FusedLinearInto_ReluActivation_MatchesAllocating()
    {
        int m = 5, kdim = 16, n = 8;
        var input = R([m, kdim], 31);
        var w = R([kdim, n], 32);
        var bias = R([n], 33);

        var expected = E.FusedLinear(input, w, bias, FusedActivationType.ReLU);
        var dest = new Tensor<float>([m, n]);
        E.FusedLinearInto(dest, input, w, bias, FusedActivationType.ReLU);
        AE(dest, expected, "FusedLinear-into relu");
    }

    // ---- FusedLinearInto: no bias ----
    [Fact]
    public void FusedLinearInto_NoBias_MatchesAllocating()
    {
        int m = 6, kdim = 10, n = 7;
        var input = R([m, kdim], 41);
        var w = R([kdim, n], 42);

        var expected = E.FusedLinear(input, w, null, FusedActivationType.None);
        var dest = new Tensor<float>([m, n]);
        E.FusedLinearInto(dest, input, w, null, FusedActivationType.None);
        AE(dest, expected, "FusedLinear-into no-bias");
    }

    // ---- TensorBroadcastAddInto ----
    [Fact]
    public void BroadcastAddInto_MatchesAllocating()
    {
        var a = R([4, 8], 51);
        var b = R([1, 8], 52);
        var expected = E.TensorBroadcastAdd(a, b);
        var dest = new Tensor<float>([4, 8]);
        E.TensorBroadcastAddInto(dest, a, b);
        AE(dest, expected, "broadcast-add-into");
    }

    // ---- TensorBroadcastMultiplyInto ----
    [Fact]
    public void BroadcastMultiplyInto_MatchesAllocating()
    {
        var a = R([4, 8], 61);
        var b = R([1, 8], 62);
        var expected = E.TensorBroadcastMultiply(a, b);
        var dest = new Tensor<float>([4, 8]);
        E.TensorBroadcastMultiplyInto(dest, a, b);
        AE(dest, expected, "broadcast-mul-into");
    }
}
