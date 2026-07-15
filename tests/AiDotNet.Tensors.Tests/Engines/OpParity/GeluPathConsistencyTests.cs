// Copyright (c) AiDotNet. All rights reserved.
// #775: the CpuEngine had TWO float GELU kernels — the eager path (x·sigmoid, stable) and the
// GELUInto / compiled-training path (previously 2·sigmoid(2z)−1, which cancels near zero). They
// silently disagreed in the ViT flat region, and only the compiled one — the path CpuEngine
// training actually runs — drifted, feeding the BCE-with-logits amplification #775 describes.
// These tests pin that the two paths now agree, and that the compiled path is accurate near zero
// against a double reference. CPU-only, so they run everywhere (no GPU required).
#if !NETFRAMEWORK

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public sealed class GeluPathConsistencyTests
{
    private static Tensor<float> Vec(float[] d) => new Tensor<float>(d, new[] { d.Length });

    /// <summary>Eager GELU and the compiled-path GELUInto must produce the same float result —
    /// including the near-zero region where the old cancellation diverged.</summary>
    [Fact]
    public void EagerGelu_MatchesCompiledGeluInto_IncludingNearZero()
    {
        var cpu = new CpuEngine();
        // A spread that stresses the flat region (tiny magnitudes) plus normal values, length 64
        // so it exercises the SIMD body (>= 32) rather than only the scalar tail.
        var data = new float[64];
        var rng = new Random(775);
        for (int i = 0; i < data.Length; i++)
        {
            double u = rng.NextDouble();
            data[i] = i < 24
                ? (float)((u * 2 - 1) * 3e-5)     // near-zero flat region (where cancellation bit)
                : (float)((u * 2 - 1) * 6.0);      // normal range
        }

        var eager = cpu.GELU(Vec(data)).ToArray();
        var dst = new Tensor<float>(new[] { data.Length });
        cpu.GELUInto(dst, Vec(data));
        var into = dst.ToArray();

        for (int i = 0; i < data.Length; i++)
            Assert.True(Math.Abs(eager[i] - into[i]) <= 1e-7f,
                $"eager vs GELUInto diverged @[{i}] x={data[i]:R}: eager={eager[i]:R} into={into[i]:R}");
    }

    /// <summary>The compiled-path GELUInto is accurate near zero: within a tight absolute bound of
    /// the double reference (which uses accurate tanh). Pre-fix, tiny inputs lost most of their
    /// significant bits and this failed.</summary>
    [Fact]
    public void CompiledGeluInto_IsAccurate_NearZero()
    {
        var cpu = new CpuEngine();
        var xs = new float[] { -5e-5f, -2e-5f, -7e-6f, -1e-6f, 1e-6f, 7e-6f, 2e-5f, 5e-5f,
                               -3e-4f, 3e-4f, -1e-3f, 1e-3f, -0.5f, 0.5f, -2f, 2f };

        var dst = new Tensor<float>(new[] { xs.Length });
        cpu.GELUInto(dst, Vec(xs));
        var got = dst.ToArray();

        for (int i = 0; i < xs.Length; i++)
        {
            // Reference: tanh-approx GELU in double with accurate Math.Tanh.
            double x = xs[i];
            double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
            double refv = 0.5 * x * (1.0 + Math.Tanh(inner));
            double err = Math.Abs(got[i] - refv);
            // Near-zero GELU(x) ≈ 0.5x, so an accurate result tracks the reference to ~fp32 eps·|x|;
            // 5e-7 absolute cleanly separates that from the old cancellation garbage (~|x|).
            Assert.True(err <= 5e-7, $"GELUInto inaccurate @[{i}] x={x:R}: got={got[i]:R} ref={refv:R} err={err:E2}");
        }
    }
}
#endif
