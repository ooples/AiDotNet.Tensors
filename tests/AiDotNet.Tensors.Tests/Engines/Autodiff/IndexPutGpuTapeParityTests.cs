using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// GPU and CPU <c>TensorIndexPut</c> must produce the same GRADIENTS, not just the same forward.
/// </summary>
/// <remarks>
/// <para>
/// This closes a specific hole. GpuCpuAutoDifferentialTests compares forward results and out-parameters
/// but never builds a GradientTape; DifferentiableOpsGradCheckSweep and the CPU IndexPut suite both use
/// CpuEngine only. So nothing compared the two engines' gradients for this op.
/// </para>
/// <para>
/// That matters because the GPU override derives the flat destination positions TWICE: the kernel
/// computes them on the device from a stride vector, and the tape needs them host-side as an int[] for
/// IndexPutBackward. If those disagree, the forward writes one set of cells while the backward credits
/// another — a silently wrong gradient with a perfect forward, which is exactly what a forward-only
/// comparison cannot see. (They now share one stride vector, and this test is what keeps them shared.)
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class IndexPutGpuTapeParityTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public IndexPutGpuTapeParityTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor<int> Idx(params int[] v)
    {
        var t = new Tensor<int>([v.Length]);
        for (int i = 0; i < v.Length; i++) t[i] = v[i];
        return t;
    }

    /// <summary>
    /// Runs <c>sum(IndexPut(tensor, indices, source))</c> on one engine and returns both gradients.
    /// </summary>
    private static (Tensor<float> dTensor, Tensor<float> dSource) GradsOn(
        IEngine engine, Tensor<float> tensor, Tensor<int>[] indices, Tensor<float> source)
    {
        using var tape = new GradientTape<float>();
        tape.BindEngineIfUnset(engine);
        var written = engine.TensorIndexPut(tensor, indices, source, accumulate: false);
        var loss = engine.ReduceSum(written, null);
        var grads = tape.ComputeGradients(loss, [tensor, source]);
        return (grads[tensor], grads[source]);
    }

    /// <remarks>
    /// The index sets deliberately do NOT address a contiguous prefix: the positions are scattered
    /// across rows and columns so a stride error moves them rather than merely reordering them, and
    /// row 1 is left untouched so the destination gradient has a structural zero to get wrong.
    /// </remarks>
    [SkippableTheory]
    [InlineData(3, 2)]
    [InlineData(4, 5)]
    public void IndexPutGradients_GpuMatchesCpu(int rows, int cols)
    {
        Skip.If(!_available, "GPU backend not available");

        var tensor = Rand([rows, cols], 41);
        var source = Rand([3], 43);
        var indices = new[] { Idx(0, 2, 2), Idx(cols - 1, 0, cols - 1) };

        var (cpuDT, cpuDS) = GradsOn(_cpu, tensor, indices, source);
        var (gpuDT, gpuDS) = GradsOn(_gpu!, tensor, indices, source);

        double worstT = 0, worstS = 0;
        for (int i = 0; i < cpuDT.Length; i++)
            worstT = Math.Max(worstT, Math.Abs((double)cpuDT[i] - gpuDT[i]));
        for (int i = 0; i < cpuDS.Length; i++)
            worstS = Math.Max(worstS, Math.Abs((double)cpuDS[i] - gpuDS[i]));

        _out.WriteLine($"[{rows},{cols}] d/dtensor maxAbsDiff={worstT:E3}  d/dsource maxAbsDiff={worstS:E3}");
        Assert.True(worstT < 1e-6,
            $"GPU and CPU disagree by {worstT:E3} on d/dtensor — the kernel's device-computed write " +
            $"positions and the backward's host-computed positions have drifted apart.");
        Assert.True(worstS < 1e-6, $"GPU and CPU disagree by {worstS:E3} on d/dsource.");
    }

    /// <summary>
    /// The gradient is also checked against its CLOSED FORM, so the two engines agreeing on a wrong
    /// answer cannot pass.
    /// </summary>
    /// <remarks>
    /// For <c>sum(IndexPut(t, idx, s))</c> with accumulate: false, every written cell takes its value
    /// from <c>source</c>, so d/dtensor is 1 everywhere EXCEPT the written cells (0 there) and d/dsource
    /// is 1 for each element. Duplicate positions are avoided here so the expected pattern is
    /// unambiguous.
    /// </remarks>
    [SkippableFact]
    public void IndexPutGradients_MatchClosedForm()
    {
        Skip.If(!_available, "GPU backend not available");

        const int Rows = 3, Cols = 2;
        var tensor = Rand([Rows, Cols], 51);
        var source = Rand([3], 53);
        // (0,0), (1,1), (2,0) — distinct cells, one per row.
        var indices = new[] { Idx(0, 1, 2), Idx(0, 1, 0) };
        var written = new HashSet<int> { 0 * Cols + 0, 1 * Cols + 1, 2 * Cols + 0 };

        var (gpuDT, gpuDS) = GradsOn(_gpu!, tensor, indices, source);

        for (int i = 0; i < Rows * Cols; i++)
        {
            double expected = written.Contains(i) ? 0.0 : 1.0;
            Assert.True(Math.Abs((double)gpuDT[i] - expected) < 1e-6,
                $"d/dtensor[{i}] = {gpuDT[i]:G9}, expected {expected} " +
                $"({(written.Contains(i) ? "overwritten, so the destination cannot influence the output" : "untouched, so it passes through")})");
        }
        for (int i = 0; i < source.Length; i++)
            Assert.True(Math.Abs((double)gpuDS[i] - 1.0) < 1e-6,
                $"d/dsource[{i}] = {gpuDS[i]:G9}, expected 1");
    }
}
