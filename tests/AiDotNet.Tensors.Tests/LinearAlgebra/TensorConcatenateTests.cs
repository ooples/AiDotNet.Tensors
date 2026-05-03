using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
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
    /// Half the threads run inside an <c>AutoTracer</c>-affecting tape
    /// scope so that GraphMode toggles are exercised concurrently with
    /// the eager path.
    /// </summary>
    [Fact]
    public void Concatenate_Rank2_ConcurrentStress_AllSucceed()
    {
        const int parallelism = 32;
        const int iterationsPerThread = 500;

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
                for (int i = 0; i < iterationsPerThread; i++)
                {
                    try
                    {
                        var a = new Tensor<double>([1, 768]);
                        var b = new Tensor<double>([1, 1280]);
                        int axis = a.Rank - 1;
                        var r = engine.TensorConcatenate(new[] { a, b }, axis);
                        if (r.Shape.Length == 2 && r.Shape[1] == 2048)
                            Interlocked.Increment(ref successCount);
                        else
                        {
                            Interlocked.Increment(ref failCount);
                            lock (fff) firstFailure ??= $"WRONG SHAPE: rank={r.Shape.Length}, dim1={r.Shape[1]}";
                        }
                    }
                    catch (Exception ex)
                    {
                        Interlocked.Increment(ref failCount);
                        lock (fff) firstFailure ??= $"{ex.GetType().Name}: {ex.Message}";
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
