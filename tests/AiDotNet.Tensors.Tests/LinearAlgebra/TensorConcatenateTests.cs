using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
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
}
