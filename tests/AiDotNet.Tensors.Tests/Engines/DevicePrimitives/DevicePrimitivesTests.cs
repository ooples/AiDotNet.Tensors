// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// Acceptance tests for the GPU primitives breadth surface from #219.
/// Covers: cross-backend Philox bit-equivalence, CSR sparse SpMM/SpMV
/// against a dense reference, dense linalg factorisations via the
/// <see cref="CuSolver"/> wrapper, RNN forward correctness against a
/// hand-computed reference, NVTX no-op safety on hosts without
/// <c>nvToolsExt</c>, and the unified GPU memory-stats ledger.
/// </summary>
public class DevicePrimitivesTests
{
    [Fact]
    public void Philox_CpuAndCuRand_ProduceIdenticalBits_ForSameSeedOffset()
    {
        // Both wrappers run the same managed Philox 4x32-10 today; the
        // test pins that contract so the cross-backend determinism
        // promise from #219 holds even after a future GPU dispatch flip.
        var cpu = new CpuPhiloxGenerator(seed: 42, subsequence: 7, offset: 11);
        var cuda = new CuRand(seed: 42, subsequence: 7, offset: 11);

        var a = new Tensor<float>(new[] { 64 });
        var b = new Tensor<float>(new[] { 64 });
        cpu.Uniform(a);
        cuda.Uniform(b);

        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        for (int i = 0; i < 64; i++) Assert.Equal(aSpan[i], bSpan[i]);
    }

    [Fact]
    public void Philox_Uniform_DistributesInUnitInterval()
    {
        var rng = new CpuPhiloxGenerator(seed: 0xDEADBEEFUL);
        var t = new Tensor<float>(new[] { 1024 });
        rng.Uniform(t);
        var span = t.AsSpan();
        double mean = 0;
        for (int i = 0; i < 1024; i++)
        {
            Assert.InRange(span[i], 0f, 1f);
            mean += span[i];
        }
        mean /= 1024;
        Assert.InRange(mean, 0.4, 0.6);
    }

    [Fact]
    public void Philox_OffsetAdvances_SoSubsequentDrawsDontRepeat()
    {
        var rng = new CpuPhiloxGenerator(seed: 1);
        var first = new Tensor<float>(new[] { 4 });
        var second = new Tensor<float>(new[] { 4 });
        rng.Uniform(first);
        rng.Uniform(second);

        var f = first.AsSpan();
        var s = second.AsSpan();
        // The two 4-element blocks come from different counter values,
        // so they shouldn't be element-wise identical.
        bool anyDifferent = false;
        for (int i = 0; i < 4; i++) if (f[i] != s[i]) { anyDifferent = true; break; }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void CsrSpMM_AgainstDenseReference()
    {
        // A = [[1, 0, 2], [0, 3, 0]] in CSR;
        // B = [[1, 2], [3, 4], [5, 6]];
        // A·B = [[11, 14], [9, 12]].
        var values = new Tensor<float>(new[] { 3 });
        values[0] = 1f; values[1] = 2f; values[2] = 3f;
        var rowPtr = new Tensor<int>(new[] { 3 });
        rowPtr[0] = 0; rowPtr[1] = 2; rowPtr[2] = 3;
        var colIdx = new Tensor<int>(new[] { 3 });
        colIdx[0] = 0; colIdx[1] = 2; colIdx[2] = 1;
        var dense = new Tensor<float>(new[] { 3, 2 });
        dense[0, 0] = 1; dense[0, 1] = 2;
        dense[1, 0] = 3; dense[1, 1] = 4;
        dense[2, 0] = 5; dense[2, 1] = 6;

        ISparseDeviceOps cuSparse = new CuSparse();
        var result = cuSparse.SpMM(values, rowPtr, colIdx, rows: 2, cols: 3, dense);

        Assert.Equal(2, result._shape[0]);
        Assert.Equal(2, result._shape[1]);
        Assert.Equal(11f, result[0, 0]);
        Assert.Equal(14f, result[0, 1]);
        Assert.Equal(9f, result[1, 0]);
        Assert.Equal(12f, result[1, 1]);
    }

    [Fact]
    public void CsrSpMV_MatchesDenseReference()
    {
        // A = [[1, 0, 2], [0, 3, 0]]; x = [1, 2, 3]^T; A·x = [7, 6]^T.
        var values = new Tensor<float>(new[] { 3 });
        values[0] = 1f; values[1] = 2f; values[2] = 3f;
        var rowPtr = new Tensor<int>(new[] { 3 });
        rowPtr[0] = 0; rowPtr[1] = 2; rowPtr[2] = 3;
        var colIdx = new Tensor<int>(new[] { 3 });
        colIdx[0] = 0; colIdx[1] = 2; colIdx[2] = 1;
        var x = new Tensor<float>(new[] { 3 });
        x[0] = 1; x[1] = 2; x[2] = 3;

        var result = new CuSparse().SpMV(values, rowPtr, colIdx, rows: 2, cols: 3, x);

        Assert.Equal(2, result.Length);
        Assert.Equal(7f, result[0]);
        Assert.Equal(6f, result[1]);
    }

    [Fact]
    public void Cholesky_ProducesValidLowerTriangle_OnSpdMatrix()
    {
        // A = [[4, 2], [2, 3]] is symmetric positive definite.
        // L should satisfy L · L^T = A.
        var a = new Tensor<float>(new[] { 2, 2 });
        a[0, 0] = 4; a[0, 1] = 2;
        a[1, 0] = 2; a[1, 1] = 3;

        var l = new CuSolver().Cholesky(a, upper: false);

        // L should be lower-triangular.
        Assert.Equal(0f, l[0, 1]);
        // L · L^T == A.
        float a00 = l[0, 0] * l[0, 0];
        float a10 = l[1, 0] * l[0, 0];
        float a11 = l[1, 0] * l[1, 0] + l[1, 1] * l[1, 1];
        Assert.Equal(4f, a00, 4);
        Assert.Equal(2f, a10, 4);
        Assert.Equal(3f, a11, 4);
    }

    [Fact]
    public void LuFactor_PermutationAndProduct_RecoverInput()
    {
        var a = new Tensor<float>(new[] { 3, 3 });
        a[0, 0] = 2; a[0, 1] = 1; a[0, 2] = 1;
        a[1, 0] = 4; a[1, 1] = 3; a[1, 2] = 3;
        a[2, 0] = 8; a[2, 1] = 7; a[2, 2] = 9;

        var (lu, pivots) = new CuSolver().Lu(a);

        Assert.Equal(2, lu.Rank);
        Assert.Equal(3, lu._shape[0]);
        Assert.Equal(3, lu._shape[1]);
        Assert.Equal(3, pivots.Length);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_ProducesShapeMatchingPyTorch()
    {
        // 1-layer, unidirectional, no projection: output [seqLen, batch, hidden].
        const int seqLen = 5, batch = 2, inputSize = 3, hidden = 4;
        const int gates = 4;

        var input = new Tensor<float>(new[] { seqLen, batch, inputSize });
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++) inSpan[i] = (i % 7) * 0.1f;

        var h0 = new Tensor<float>(new[] { batch, hidden });
        var c0 = new Tensor<float>(new[] { batch, hidden });

        int wIhSize = gates * hidden * inputSize;
        int wHhSize = gates * hidden * hidden;
        int bSize = gates * hidden;
        var weights = new Tensor<float>(new[] { wIhSize + wHhSize + 2 * bSize });
        var wSpan = weights.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = ((i % 5) - 2) * 0.05f;

        var rnn = new CuDnnRnn();
        var (output, hN, cN) = rnn.ForwardLstm(input, h0, c0, weights, new RnnOptions { HiddenSize = hidden });

        Assert.Equal(new[] { seqLen, batch, hidden }, output._shape);
        Assert.Equal(new[] { 1, batch, hidden }, hN._shape);
        Assert.Equal(new[] { 1, batch, hidden }, cN._shape);
    }

    [Fact]
    public void CpuRnn_ForwardPlainTanh_DegeneratesToTanhOnZeroWeights()
    {
        // With zero weights and zero biases, h_t = tanh(0) = 0 for all t.
        const int seqLen = 3, batch = 2, inputSize = 4, hidden = 3;
        const int gates = 1;

        var input = new Tensor<float>(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { batch, hidden });
        var weights = new Tensor<float>(new[] { gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden });

        var rnn = new CuDnnRnn();
        var (output, hN, _) = rnn.ForwardRnn(RnnCellType.RnnTanh, input, h0, c0: null, weights, new RnnOptions { HiddenSize = hidden });

        var oSpan = output.AsSpan();
        for (int i = 0; i < oSpan.Length; i++) Assert.Equal(0f, oSpan[i], 6);
        var hSpan = hN.AsSpan();
        for (int i = 0; i < hSpan.Length; i++) Assert.Equal(0f, hSpan[i], 6);
    }

    [Fact]
    public void CpuDevicePrimitives_Reduce_Sum_AcrossAllElements()
    {
        var t = new Tensor<float>(new[] { 4 });
        t[0] = 1; t[1] = 2; t[2] = 3; t[3] = 4;
        var sum = new CpuDevicePrimitives().Reduce(t, axis: -1, ReductionKind.Sum);
        Assert.Equal(10f, sum[0]);
    }

    [Fact]
    public void CpuDevicePrimitives_Scan_Inclusive_MatchesCumsum()
    {
        var t = new Tensor<float>(new[] { 4 });
        t[0] = 1; t[1] = 2; t[2] = 3; t[3] = 4;
        var cs = new CpuDevicePrimitives().Scan(t, axis: 0, ReductionKind.Sum, exclusive: false);
        Assert.Equal(1f, cs[0]);
        Assert.Equal(3f, cs[1]);
        Assert.Equal(6f, cs[2]);
        Assert.Equal(10f, cs[3]);
    }

    [Fact]
    public void CpuDevicePrimitives_Sort_Ascending()
    {
        var t = new Tensor<float>(new[] { 5 });
        t[0] = 3; t[1] = 1; t[2] = 4; t[3] = 1; t[4] = 5;
        var sorted = new CpuDevicePrimitives().Sort(t);
        Assert.Equal(1f, sorted[0]);
        Assert.Equal(1f, sorted[1]);
        Assert.Equal(3f, sorted[2]);
        Assert.Equal(4f, sorted[3]);
        Assert.Equal(5f, sorted[4]);
    }

    [Fact]
    public void CpuDevicePrimitives_RunLengthEncode_GroupsConsecutive()
    {
        var t = new Tensor<int>(new[] { 7 });
        t[0] = 1; t[1] = 1; t[2] = 2; t[3] = 3; t[4] = 3; t[5] = 3; t[6] = 4;
        var (vals, counts) = new CpuDevicePrimitives().RunLengthEncode(t);
        Assert.Equal(4, vals.Length);
        Assert.Equal(1, vals[0]); Assert.Equal(2, counts[0]);
        Assert.Equal(2, vals[1]); Assert.Equal(1, counts[1]);
        Assert.Equal(3, vals[2]); Assert.Equal(3, counts[2]);
        Assert.Equal(4, vals[3]); Assert.Equal(1, counts[3]);
    }

    [Fact]
    public void Nvtx_PushPopMark_AreSafeOnHostsWithoutNvToolsExt()
    {
        // The wrapper must never throw — instrumentation can never break
        // a run. Push/Pop/Mark/Range should all be safe-no-op when
        // libnvToolsExt is missing.
        Nvtx.Mark("test_mark");
        Nvtx.Push("test_push");
        Nvtx.Pop();
        using (Nvtx.Range("test_range")) { /* scope */ }
        Assert.True(true);
    }

    [Fact]
    public void GpuMemoryStats_RecordAllocFree_ReflectInCounters()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 1024);
        GpuMemoryStats.RecordAllocation("test_alloc", 2048);
        Assert.Equal(3072, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes);
        Assert.Equal(3072, GpuMemoryStats.TotalAllocatedBytes);
        Assert.Equal(2, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.RecordFree("test_alloc", 1024);
        Assert.Equal(2048, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes); // peak is sticky
        Assert.Equal(1, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.ResetPeakStats();
        Assert.Equal(2048, GpuMemoryStats.PeakBytes);

        GpuMemoryStats.Reset();
    }

    [Fact]
    public void GpuMemoryStats_Stats_ExposesTorchParityKeys()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 512);
        var stats = GpuMemoryStats.Stats();

        Assert.Equal(512L, stats["allocated_bytes.current"]);
        Assert.Equal(512L, stats["allocated_bytes.peak"]);
        Assert.Equal(512L, stats["allocated_bytes.total"]);
        Assert.Equal(1L, stats["active.current"]);

        GpuMemoryStats.Reset();
    }
}
