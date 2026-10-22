using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1502: OpenCL's device-side finiteness guard. The multi-tensor sum of squares finishes as the double the guard
/// promises (NaN exactly when an element is not finite, finite when only the float sum overflowed); the guarded SGD,
/// Adam and AdamW kernels match the unguarded ones on a finite guard and write nothing on a non-finite one; and the
/// readback returns the value its source held when it was queued.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClDeviceFiniteGuardTests
{
    private static OpenClBackend BackendOrSkip()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        var backend = new OpenClBackend(deviceIndex: 0);
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "OpenCL backend did not initialize.");
        }
        return backend;
    }

    private static float[] Rand(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
    }

    private static double ReadDouble(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        float[] slots = backend.DownloadBuffer(buffer);
        var bytes = new byte[sizeof(double)];
        Buffer.BlockCopy(slots, 0, bytes, 0, sizeof(double));
        return BitConverter.ToDouble(bytes, 0);
    }

    private static double SumOfSquares(OpenClBackend backend, params float[][] tensors)
    {
        var ops = (IDeviceScalarOps)backend;
        Assert.True(ops.SupportsMultiTensorSumOfSquares);
        var buffers = tensors.Select(t => backend.AllocateBuffer(t)).ToArray();
        using var sum = backend.AllocateBuffer(2);
        try
        {
            ops.MultiTensorSumOfSquares(buffers, tensors.Select(t => t.Length).ToArray(), sum);
            backend.Synchronize();
            return ReadDouble(backend, sum);
        }
        finally
        {
            foreach (var b in buffers) b.Dispose();
        }
    }

    [SkippableFact]
    public void The_sum_of_squares_is_the_double_the_guard_reads()
    {
        using var backend = BackendOrSkip();
        var tensors = new[] { Rand(1000, 1), Rand(7, 2), Rand(300, 3, 4f) };
        double expected = tensors.SelectMany(t => t).Sum(v => (double)v * v);
        double got = SumOfSquares(backend, tensors);
        Assert.True(Math.Abs(got - expected) <= 1e-5 * expected, $"{got} vs {expected}");
        Assert.Equal(0.0, SumOfSquares(backend, new float[16]));
        // A subnormal float sum converts exactly.
        float tiny = 1e-23f;
        Assert.Equal((double)(tiny * tiny), SumOfSquares(backend, new[] { tiny }));
    }

    [SkippableFact]
    public void A_non_finite_element_makes_the_guard_nan_and_a_float_overflow_does_not()
    {
        using var backend = BackendOrSkip();
        var withNan = Rand(500, 4);
        withNan[123] = float.NaN;
        Assert.True(double.IsNaN(SumOfSquares(backend, Rand(64, 5), withNan)));
        var withInf = Rand(500, 6);
        withInf[7] = float.NegativeInfinity;
        Assert.True(double.IsNaN(SumOfSquares(backend, withInf)));
        // Every element finite, but their float sum of squares overflows: CUDA's double sum is finite, and so is this.
        double overflow = SumOfSquares(backend, Enumerable.Repeat(1e30f, 8).ToArray());
        Assert.False(double.IsNaN(overflow) || double.IsInfinity(overflow), $"{overflow}");
    }

    [SkippableTheory]
    [InlineData("sgd")]
    [InlineData("adam")]
    [InlineData("adamw")]
    public void The_guarded_update_matches_the_unguarded_one_and_writes_nothing_on_a_non_finite_guard(string kind)
    {
        using var backend = BackendOrSkip();
        var opt = (IMultiTensorGpuOptimizerBackend)backend;
        Assert.True(opt.GuardsFiniteOnDevice);
        int[] sizes = { 300, 17 };
        float[][] p0 = sizes.Select((n, i) => Rand(n, 10 + i)).ToArray();
        float[][] g0 = sizes.Select((n, i) => Rand(n, 20 + i)).ToArray();
        float[][] m0 = sizes.Select((n, i) => Rand(n, 30 + i, 0.1f)).ToArray();
        float[][] v0 = sizes.Select((n, i) => Rand(n, 40 + i, 0.1f).Select(Math.Abs).ToArray()).ToArray();

        float[][][] Run(double? guardValue)
        {
            var p = p0.Select(a => backend.AllocateBuffer(a)).ToArray();
            var g = g0.Select(a => backend.AllocateBuffer(a)).ToArray();
            var m = m0.Select(a => backend.AllocateBuffer(a)).ToArray();
            var v = v0.Select(a => backend.AllocateBuffer(a)).ToArray();
            using var guard = backend.AllocateBuffer(2);
            try
            {
                if (guardValue is { } gv)
                {
                    var bits = BitConverter.GetBytes(gv);
                    backend.UploadBufferInPlace(new[] { BitConverter.ToSingle(bits, 0), BitConverter.ToSingle(bits, 4) }, guard);
                }
                switch (kind)
                {
                    case "sgd" when guardValue is null: opt.SgdMultiTensorUpdate(p, g, sizes, 0.05f, 0.01f); break;
                    case "sgd": opt.SgdMultiTensorUpdateGuarded(p, g, sizes, 0.05f, 0.01f, guard); break;
                    case "adam" when guardValue is null:
                        opt.AdamMultiTensorUpdate(p, g, m, v, sizes, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 3); break;
                    case "adamw" when guardValue is null:
                        opt.AdamWMultiTensorUpdate(p, g, m, v, sizes, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 3); break;
                    default:
                        opt.AdamMultiTensorUpdateGuarded(kind == "adamw", p, g, m, v, sizes, 1e-3f, 0.9f, 0.999f, 1e-8f,
                            0.01f, 3, guard);
                        break;
                }
                backend.Synchronize();
                return new[] { p, m, v }.Select(list => list.Select(b => backend.DownloadBuffer(b)).ToArray()).ToArray();
            }
            finally
            {
                foreach (var b in p.Concat(g).Concat(m).Concat(v)) b.Dispose();
            }
        }

        var unguarded = Run(null);
        var finite = Run(12.5);
        var skipped = Run(double.NaN);
        var skippedInf = Run(double.PositiveInfinity);
        for (int s = 0; s < 3; s++)
            for (int t = 0; t < sizes.Length; t++)
            {
                float[] original = s == 0 ? p0[t] : s == 1 ? m0[t] : v0[t];
                Assert.Equal(unguarded[s][t].Select(f => TestHelpers.MathCompat.SingleToInt32Bits(f)),
                    finite[s][t].Select(f => TestHelpers.MathCompat.SingleToInt32Bits(f)));
                Assert.Equal(original, skipped[s][t].Take(original.Length).ToArray());
                Assert.Equal(original, skippedInf[s][t].Take(original.Length).ToArray());
            }
        Assert.NotEqual(p0[0], unguarded[0][0].Take(p0[0].Length).ToArray());   // the control: the update did something
    }

    [SkippableFact]
    public void The_readback_returns_the_value_its_source_held_when_queued()
    {
        using var backend = BackendOrSkip();
        var factory = (IDeviceReadbackFactory)backend;
        using var readback = factory.CreateReadback(sizeof(double));
        using var source = backend.AllocateBuffer(new[] { 1.5f, -2.25f, 9f, 9f });
        Assert.False(readback.HasUnreadValue);
        readback.Enqueue(source, 2 * sizeof(float));
        Assert.True(readback.HasUnreadValue);
        backend.Fill(source, 7f, 4);   // queued after the copy: must not reach it
        var bytes = readback.Read().ToArray();
        Assert.False(readback.HasUnreadValue);
        Assert.Equal(2 * sizeof(float), bytes.Length);
        Assert.Equal(1.5f, BitConverter.ToSingle(bytes, 0));
        Assert.Equal(-2.25f, BitConverter.ToSingle(bytes, 4));
        readback.Enqueue(source, sizeof(float));
        Assert.Equal(7f, BitConverter.ToSingle(readback.Read().ToArray(), 0));
    }
}
