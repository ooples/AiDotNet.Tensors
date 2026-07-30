using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression test for the 3D trilinear (GridSample-3D) backward scatter-add.
/// The prior implementation sized a fixed <c>threadLocalGrads[MaxDegreeOf-
/// Parallelism]</c> array and selected a thread's accumulator by
/// <c>(Thread.CurrentThread.ManagedThreadId % numThreads)</c> — NOT collision
/// free: on a many-core box the internal parallel workers' managed ids collide
/// mod numThreads, so two workers scatter into the SAME buffer concurrently and
/// race (lost/torn updates, observed as NaN under concurrent training). The fix
/// uses a per-thread <see cref="System.Threading.ThreadLocal{T}"/> accumulator,
/// making the scatter and the aggregation race-free by construction. This test
/// exercises the many-worker parallel path and asserts the result stays finite
/// and matches a serial (single-thread) reference across repeated runs — which
/// the racing implementation could not guarantee.
/// </summary>
public class GridSample3DBackwardConcurrencyTests
{
    private readonly CpuEngine E = new();

    [Fact]
    public void TensorTrilinearInterpolateBackward_StaysFinite_AndMatchesSerial_UnderParallelism()
    {
        const int D = 6, H = 6, W = 6, C = 4, N = 40000;
        var rng = new Random(123);

        var grid = new Tensor<double>(new[] { D, H, W, C });     // gradient target shape; values unused
        var positions = new Tensor<double>(new[] { N, 3 });
        var gradOut = new Tensor<double>(new[] { N, C });
        for (int i = 0; i < N; i++)
        {
            positions[i, 0] = rng.NextDouble() * (D - 1);
            positions[i, 1] = rng.NextDouble() * (H - 1);
            positions[i, 2] = rng.NextDouble() * (W - 1);
            for (int c = 0; c < C; c++) gradOut[i, c] = rng.NextDouble() * 2.0 - 1.0;
        }

        // Serial ground truth (no worker fan-out → no possible collision).
        int savedDop = CpuParallelSettings.MaxDegreeOfParallelism;
        Tensor<double> reference;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            reference = E.TensorTrilinearInterpolateBackward(gradOut, grid, positions);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = savedDop;
        }

        int len = reference.Length;
        for (int i = 0; i < len; i++)
            Assert.True((!double.IsNaN(reference[i]) && !double.IsInfinity(reference[i])), $"serial reference[{i}] is non-finite");

        // Full-parallel path, repeated: the racing implementation produced
        // dropped contributions (values off the serial sum) or NaN when worker
        // ids collided; the fix keeps every run finite and within FP-reduction
        // tolerance of the serial reference.
        for (int rep = 0; rep < 25; rep++)
        {
            var r = E.TensorTrilinearInterpolateBackward(gradOut, grid, positions);
            Assert.Equal(len, r.Length);
            for (int i = 0; i < len; i++)
            {
                double v = r[i];
                Assert.True((!double.IsNaN(v) && !double.IsInfinity(v)), $"rep {rep}: result[{i}] is non-finite (scatter race)");
                Assert.True(Math.Abs(v - reference[i]) <= 1e-9 + 1e-9 * Math.Abs(reference[i]),
                    $"rep {rep}: result[{i}]={v} diverges from serial reference {reference[i]} (lost contributions)");
            }
        }
    }
}
