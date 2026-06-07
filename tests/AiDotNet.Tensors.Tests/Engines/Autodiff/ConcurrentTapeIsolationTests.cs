using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for the concurrent-tape contamination (issue: parallel
/// independent gradient tapes on the shared process-wide engine corrupting each
/// other's results). Root cause was shared mutable state in accelerator providers
/// executed concurrently — e.g. oneDNN cached primitives rebinding the SAME
/// dnnl_memory data handles per call on the shared stream, so two threads' in-place
/// adds (the double-accumulation in a backward pass for a tensor used as multiple
/// inputs, x*x) stomped each other → ~50% wrong gradients. Fixed by serializing the
/// provider execute sequences.
///
/// These run their own thread pool (independent of xUnit parallelism) so they fail
/// deterministically if the race regresses.
/// </summary>
public class ConcurrentTapeIsolationTests
{
    // Use a CpuEngine explicitly: this guards the CPU accelerator (oneDNN) fix
    // regardless of whether GPU auto-detection swapped AiDotNetEngine.Current to a
    // DirectGpu engine on the test host.
    private readonly IEngine _engine = new CpuEngine();

    [Fact]
    public void TensorAddInPlace_ConcurrentCallers_ProduceCorrectResults()
    {
        int threads = Math.Max(4, Environment.ProcessorCount);
        int iters = 4000;
        int wrong = 0;
        Parallel.For(0, threads, _ =>
        {
            for (int i = 0; i < iters; i++)
            {
                var a = new Tensor<float>(new[] { 2f }, new[] { 1 });
                var b = new Tensor<float>(new[] { 3f }, new[] { 1 });
                _engine.TensorAddInPlace(a, b); // a := a + b = 5
                if (MathF.Abs(a[0] - 5f) > 1e-3f) Interlocked.Increment(ref wrong);
            }
        });
        Assert.Equal(0, wrong);
    }

    [Fact]
    public void ConcurrentTapes_DoubleAccumulationGradient_IsCorrect()
    {
        // y = x*x -> dy/dx = 2x (x appears as BOTH inputs -> two accumulations to x's
        // gradient via in-place add). At x=3, grad = 6.
        int threads = Math.Max(4, Environment.ProcessorCount);
        int iters = 2000;
        int wrong = 0;
        Parallel.For(0, threads, _ =>
        {
            for (int i = 0; i < iters; i++)
            {
                using var tape = new GradientTape<float>();
                var x = new Tensor<float>(new[] { 3f }, new[] { 1 });
                var y = _engine.TensorMultiply(x, x);
                var grads = tape.ComputeGradients(y, new[] { x });
                if (MathF.Abs(grads[x][0] - 6f) > 1e-3f) Interlocked.Increment(ref wrong);
            }
        });
        Assert.Equal(0, wrong);
    }
}
