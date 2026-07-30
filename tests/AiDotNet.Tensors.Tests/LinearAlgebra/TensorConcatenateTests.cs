using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Tests for <see cref="Tensor{T}.Concatenate"/> and the engine-level
/// <c>TensorConcatenate</c> dispatch path. Includes the diagnostic-message
/// invariants and the issue #291 concurrency-stress regression coverage:
/// the concat path must produce stable results across many threads issuing
/// the same call shape, even when sibling threads are toggling
/// <c>GraphMode</c>/tape state.
/// </summary>
public class TensorConcatenateTests
{
    [Fact]
    public void Concatenate_Rank2_LastAxis_ReturnsCombinedShape()
    {
        var a = new Tensor<double>([1, 768]);
        var b = new Tensor<double>([1, 1280]);

        var r = Tensor<double>.Concatenate(new[] { a, b }, axis: 1);

        Assert.Equal(2, r.Shape.Length);
        Assert.Equal(1, r.Shape[0]);
        Assert.Equal(2048, r.Shape[1]);
    }

    [Fact]
    public void Concatenate_InvalidPositiveAxis_ThrowsWithDiagnosticMessage()
    {
        var a = new Tensor<double>([1, 768]);
        var b = new Tensor<double>([1, 1280]);

        var ex = Assert.Throws<ArgumentException>(() => Tensor<double>.Concatenate(new[] { a, b }, axis: 2));

        // Diagnostic message must include the supplied axis value, the rank,
        // AND the offending tensor's full shape — the prior message ("Must be
        // between 0 and {rank-1}") was ambiguous when the underlying race
        // (issue #291) made it appear the validator had a different rank in
        // mind than the caller did.
        Assert.Contains("Invalid axis 2", ex.Message);
        Assert.Contains("rank 2", ex.Message);
        Assert.Contains("[1, 768]", ex.Message);
    }

    [Fact]
    public void Concatenate_NegativeAxis_ThrowsWithDiagnosticMessage()
    {
        var a = new Tensor<double>([1, 768]);
        var b = new Tensor<double>([1, 1280]);

        var ex = Assert.Throws<ArgumentException>(() => Tensor<double>.Concatenate(new[] { a, b }, axis: -1));

        Assert.Contains("Invalid axis -1", ex.Message);
    }

    [Fact]
    public void Concatenate_RankMismatch_ThrowsWithBothShapes()
    {
        var a = new Tensor<double>([1, 768]);
        var b = new Tensor<double>([1, 4, 1280]);

        var ex = Assert.Throws<ArgumentException>(() => Tensor<double>.Concatenate(new[] { a, b }, axis: 1));

        Assert.Contains("[1, 768]", ex.Message);
        Assert.Contains("[1, 4, 1280]", ex.Message);
    }

    [Fact]
    public void Concatenate_NonAxisDimMismatch_ThrowsWithDiagnosticMessage()
    {
        // Non-feature axis mismatch — must surface the offending axis and values.
        var a = new Tensor<double>([1, 768]);
        var b = new Tensor<double>([2, 1280]);

        var ex = Assert.Throws<ArgumentException>(() => Tensor<double>.Concatenate(new[] { a, b }, axis: 1));

        Assert.Contains("axis 0", ex.Message);
        Assert.Contains("tensors[0][0]=1", ex.Message);
        Assert.Contains("tensors[1][0]=2", ex.Message);
    }

    [Fact]
    public void Concatenate_NullTensorEntry_ThrowsArgumentException()
    {
        var a = new Tensor<double>([1, 8]);
        Tensor<double>? b = null;

        var ex = Assert.Throws<ArgumentException>(() => Tensor<double>.Concatenate(new[] { a, b! }, axis: 1));

        Assert.Contains("tensors[1]", ex.Message);
    }

    /// <summary>
    /// Issue #291: concatenation of rank-2 tensors at axis = rank - 1 was
    /// observed to intermittently throw "Invalid axis. Must be between 0
    /// and 1." in xunit parallel test runs, even though the same call
    /// passed in isolation. This stress test issues many concat calls in
    /// parallel and verifies that 100% succeed with the expected shape.
    /// Even-indexed threads run with <see cref="GraphMode"/> active and
    /// odd-indexed threads run with a <see cref="GradientTape{T}"/>
    /// open, so the engine's GraphMode replay-closure path AND the
    /// tape-record path are both exercised concurrently with the eager
    /// path — the original bug surfaced when sibling threads toggled
    /// the same shape arrays mid-call from those branches.
    /// </summary>
    [Fact]
    public void Concatenate_Rank2_ConcurrentStress_AllSucceed()
    {
        const int parallelism = 32;
        const int iterationsPerThread = 500;
        const int expectedRank = 2;
        const int expectedDim0 = 1;
        const int expectedDim1 = 2048;

        var engine = new CpuEngine();
        int successCount = 0;
        int failCount = 0;
        string? firstFailure = null;
        var fff = new object();

        var tasks = new Task[parallelism];
        for (int t = 0; t < parallelism; t++)
        {
            int threadIdx = t;
            tasks[t] = Task.Run(() =>
            {
                // Pick a mode for this entire thread:
                //   threadIdx % 3 == 0 → eager (no scope)
                //   threadIdx % 3 == 1 → GraphMode active
                //   threadIdx % 3 == 2 → GradientTape open
                // Splitting the population across all three modes makes
                // sure the engine's Concatenate dispatch hits the
                // GraphMode replay-closure and tape-record branches
                // (both of which snapshot input shapes) concurrently
                // with the eager path that calls Tensor<T>.Concatenate.
                int mode = threadIdx % 3;
                for (int i = 0; i < iterationsPerThread; i++)
                {
                    try
                    {
                        var a = new Tensor<double>([1, 768]);
                        var b = new Tensor<double>([1, 1280]);
                        int axis = a.Rank - 1;

                        Tensor<double> r;
                        if (mode == 1)
                        {
                            using var graphScope = GraphMode.Enable();
                            r = engine.TensorConcatenate(new[] { a, b }, axis);
                        }
                        else if (mode == 2)
                        {
                            using var tape = new GradientTape<double>();
                            r = engine.TensorConcatenate(new[] { a, b }, axis);
                        }
                        else
                        {
                            r = engine.TensorConcatenate(new[] { a, b }, axis);
                        }

                        // Verify the FULL expected shape, not just dim1.
                        // Indexing r.Shape[1] in the failure message is
                        // unsafe when rank < 2 — format defensively so
                        // the wrong-shape diagnostic never throws and
                        // mask the underlying regression.
                        if (r.Shape.Length == expectedRank
                            && r.Shape[0] == expectedDim0
                            && r.Shape[1] == expectedDim1)
                        {
                            Interlocked.Increment(ref successCount);
                        }
                        else
                        {
                            Interlocked.Increment(ref failCount);
                            lock (fff)
                            {
                                firstFailure ??=
                                    $"WRONG SHAPE (mode={mode}): " +
                                    $"rank={r.Shape.Length}, " +
                                    $"shape=[{string.Join(", ", r.Shape)}], " +
                                    $"expected=[{expectedDim0}, {expectedDim1}]";
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Interlocked.Increment(ref failCount);
                        lock (fff) firstFailure ??= $"(mode={mode}) {ex.GetType().Name}: {ex.Message}";
                    }
                }
            });
        }
        Task.WaitAll(tasks);

        Assert.True(failCount == 0,
            $"Expected 0 failures across {parallelism * iterationsPerThread} concat calls, got {failCount}. " +
            $"First failure: {firstFailure}");
        Assert.Equal(parallelism * iterationsPerThread, successCount);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    //  Regression: CpuEngine.TensorConcatenate axis-0 fast path must never leak an
    //  uninitialized (pool-recycled) tail.
    //
    //  The axis-0 fast path rents an output buffer sized by SUMMING _shape[0] but
    //  fills it by COPYING each input's Length. Those agree only when every input
    //  shares tensors[0]'s trailing dims. For ragged inputs the output is larger
    //  than the copied total, so the rented buffer keeps a tail the copy never
    //  wrote. Because the buffer comes from AutoTensorCache UNINITIALIZED, a tail
    //  left over from a prior op (e.g. NaN/Inf activations from a training step on
    //  the same thread) leaked straight into the result — an intermittent, timing-
    //  sensitive NaN that only reproduced when the pool happened to hand back a
    //  dirty buffer. The fix zeroes any unwritten remainder.
    //
    //  These tests reproduce the leak deterministically by POISONING the thread-
    //  local pool with a NaN buffer of the exact output shape, so RentOrAllocate is
    //  guaranteed to hand it to the concat. Pre-fix they fail (NaN leaks); post-fix
    //  they pass (tail zeroed). Compatible-shape concats are also covered to prove
    //  the fast path still fully overwrites the poison (no behavior change there).
    // ─────────────────────────────────────────────────────────────────────────────

    /// <summary>Rents a buffer of <paramref name="shape"/>, fills it with NaN, and
    /// returns it to the thread-local pool so the next same-shape RentOrAllocate on
    /// this thread hands the dirty buffer back.</summary>
    private static void PoisonPoolWithNaN<T>(int[] shape) where T : struct
    {
        var poison = AutoTensorCache.RentOrAllocate<T>(shape);
        var data = poison.GetDataArray();
        var nan = (T)Convert.ChangeType(double.NaN, typeof(T));
        for (int i = 0; i < data.Length; i++) data[i] = nan;
        AutoTensorCache.Return(poison);
    }

    private static void AssertAllFinite(Tensor<double> t, string context)
    {
        var d = t.ToArray();
        for (int i = 0; i < d.Length; i++)
            Assert.False(double.IsNaN(d[i]) || double.IsInfinity(d[i]),
                $"{context}: result[{i}] = {d[i]} leaked from an uninitialized pool tail.");
    }

    [Fact]
    public void TensorConcatenate_Axis0_RaggedInputs_DoesNotLeakPoolTail()
    {
        var engine = new CpuEngine();
        bool prev = AutoTensorCache.Enabled;
        AutoTensorCache.Enabled = true;
        try
        {
            AutoTensorCache.Clear();

            // [1,49,64] ++ [1,64] on axis 0: the fast path sizes the output [2,49,64]
            // (6272) but copies only 3136 + 64 = 3200 — a 3072-element tail.
            PoisonPoolWithNaN<double>(new[] { 2, 49, 64 });

            var a = Filled(new[] { 1, 49, 64 }, 1.0);
            var b = Filled(new[] { 1, 64 }, 2.0);

            var result = engine.TensorConcatenate(new[] { a, b }, axis: 0);

            AssertAllFinite(result, "ragged axis-0 concat");
            var rd = result.ToArray();
            Assert.Equal(1.0, rd[0]);       // start of a
            Assert.Equal(2.0, rd[3136]);    // start of b
            Assert.Equal(0.0, rd[3200]);    // first tail element — must be zeroed, not NaN
            Assert.Equal(0.0, rd[rd.Length - 1]);
        }
        finally
        {
            AutoTensorCache.Clear();
            AutoTensorCache.Enabled = prev;
        }
    }

    [Fact]
    public void TensorConcatenate_Axis0_ThreeRaggedInputs_DoesNotLeakPoolTail()
    {
        var engine = new CpuEngine();
        bool prev = AutoTensorCache.Enabled;
        AutoTensorCache.Enabled = true;
        try
        {
            AutoTensorCache.Clear();
            // Output shape = [3,10,8] (240), copied = 80 + 8 + 40 = 128 → 112-element tail.
            PoisonPoolWithNaN<double>(new[] { 3, 10, 8 });

            var a = Filled(new[] { 1, 10, 8 }, 1.0);  // 80
            var b = Filled(new[] { 1, 8 }, 2.0);      // 8
            var c = Filled(new[] { 1, 5, 8 }, 3.0);   // 40

            var result = engine.TensorConcatenate(new[] { a, b, c }, axis: 0);

            AssertAllFinite(result, "three ragged axis-0 concat");
        }
        finally
        {
            AutoTensorCache.Clear();
            AutoTensorCache.Enabled = prev;
        }
    }

    [Fact]
    public void TensorConcatenate_Axis0_CompatibleInputs_FullyOverwritesPoison()
    {
        // Edge case: when inputs share trailing dims the copy fills the whole buffer,
        // so there is NO tail to zero. The poison must be fully overwritten and the
        // result must exactly equal the concatenated inputs (no behavior change, no
        // accidental zeroing of real data).
        var engine = new CpuEngine();
        bool prev = AutoTensorCache.Enabled;
        AutoTensorCache.Enabled = true;
        try
        {
            AutoTensorCache.Clear();
            PoisonPoolWithNaN<double>(new[] { 2, 64 });

            var a = Filled(new[] { 1, 64 }, 7.0);
            var b = Filled(new[] { 1, 64 }, 9.0);

            var result = engine.TensorConcatenate(new[] { a, b }, axis: 0);

            AssertAllFinite(result, "compatible axis-0 concat");
            var rd = result.ToArray();
            for (int i = 0; i < 64; i++) Assert.Equal(7.0, rd[i]);
            for (int i = 64; i < 128; i++) Assert.Equal(9.0, rd[i]);
        }
        finally
        {
            AutoTensorCache.Clear();
            AutoTensorCache.Enabled = prev;
        }
    }

    [Fact]
    public void TensorConcatenate_Axis0_SingleRaggedTensor_NoLeak()
    {
        // Degenerate edge case: a one-element array. outShape[0] == tensors[0]._shape[0]
        // and the copy fills the whole buffer, so nothing to zero — must still be finite.
        var engine = new CpuEngine();
        bool prev = AutoTensorCache.Enabled;
        AutoTensorCache.Enabled = true;
        try
        {
            AutoTensorCache.Clear();
            PoisonPoolWithNaN<double>(new[] { 1, 49, 64 });
            var a = Filled(new[] { 1, 49, 64 }, 5.0);

            var result = engine.TensorConcatenate(new[] { a }, axis: 0);

            AssertAllFinite(result, "single-tensor axis-0 concat");
        }
        finally
        {
            AutoTensorCache.Clear();
            AutoTensorCache.Enabled = prev;
        }
    }

    [Fact]
    public void TensorConcatenate_Axis0_Float_RaggedInputs_DoesNotLeakPoolTail()
    {
        // Generic-type edge case: the fix lives in the shared generic method, so the
        // float path must be protected too.
        var engine = new CpuEngine();
        bool prev = AutoTensorCache.Enabled;
        AutoTensorCache.Enabled = true;
        try
        {
            AutoTensorCache.Clear();
            PoisonPoolWithNaN<float>(new[] { 2, 20, 4 });

            var a = new Tensor<float>(new[] { 1, 20, 4 });
            for (int i = 0; i < a.Length; i++) a[i] = 1f;
            var b = new Tensor<float>(new[] { 1, 4 });
            for (int i = 0; i < b.Length; i++) b[i] = 2f;

            var result = engine.TensorConcatenate(new[] { a, b }, axis: 0);

            var d = result.ToArray();
            for (int i = 0; i < d.Length; i++)
                Assert.False(float.IsNaN(d[i]) || float.IsInfinity(d[i]),
                    $"float ragged concat: result[{i}] = {d[i]} leaked from the pool tail.");
        }
        finally
        {
            AutoTensorCache.Clear();
            AutoTensorCache.Enabled = prev;
        }
    }

    [Fact]
    public void TensorConcatenate_SequenceAxis_RankMismatch_Throws()
    {
        // The validating (non-axis-0) path must reject genuinely incompatible inputs
        // with a clear error rather than silently producing a wrong-sized result. This
        // is the path the corrected LayoutXLM/LayoutLMv2 fusion relies on when it aligns
        // both streams to [B, L, D] and concatenates on the sequence axis.
        var engine = new CpuEngine();
        var a = new Tensor<double>(new[] { 1, 49, 64 });
        var b = new Tensor<double>(new[] { 1, 64 });

        Assert.Throws<ArgumentException>(() => engine.TensorConcatenate(new[] { a, b }, axis: 1));
    }

    private static Tensor<double> Filled(int[] shape, double value)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = value;
        return t;
    }
}
