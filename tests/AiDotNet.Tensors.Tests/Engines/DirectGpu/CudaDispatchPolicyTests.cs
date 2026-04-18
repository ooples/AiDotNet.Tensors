using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Tests for issue #201 — <see cref="CudaDispatchPolicy"/> and the
/// <c>UseCudnn</c> / <c>UseCudnnBatchNorm</c> / <c>UseCublas</c> opt-out
/// flags on <see cref="TensorCodecOptions"/>.
///
/// <para>Runs on CPU — no CUDA device required. The dispatch-policy
/// helpers expose the policy logic independently of whether a real CUDA
/// backend initialises successfully, so the flag semantics are testable
/// without GPU hardware.</para>
/// </summary>
public class CudaDispatchPolicyTests
{
    [Fact]
    public void DefaultOptions_UseCudnnAndUseCublas_AreTrue()
    {
        var opts = new TensorCodecOptions();
        Assert.True(opts.UseCudnn);
        Assert.True(opts.UseCudnnBatchNorm);
        Assert.True(opts.UseCublas);
    }

    [Fact]
    public void UseCublasForMatMul_ReflectsFlag()
    {
        var prev = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { UseCublas = true });
            Assert.True(CudaDispatchPolicy.UseCublasForMatMul);

            TensorCodecOptions.SetCurrent(new TensorCodecOptions { UseCublas = false });
            Assert.False(CudaDispatchPolicy.UseCublasForMatMul);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prev);
        }
    }

    [Fact]
    public void UseCudnnForConv_FalseWhenFlagOff_RegardlessOfAvailability()
    {
        var prev = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { UseCudnn = false });
            Assert.False(CudaDispatchPolicy.UseCudnnForConv);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prev);
        }
    }

    [Fact]
    public void UseCudnnForBatchNorm_FalseWhenFlagOff_RegardlessOfAvailability()
    {
        var prev = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { UseCudnnBatchNorm = false });
            Assert.False(CudaDispatchPolicy.UseCudnnForBatchNorm);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prev);
        }
    }

    [Fact]
    public void Scope_ReturnsProfilerScopeWithExpectedLabel()
    {
        // Scope should produce a PerformanceProfiler scope whose Dispose
        // records the operation. We verify by clearing profiler stats,
        // opening + disposing the scope, and checking that GetStats
        // returns a hit under the expected label.
        PerformanceProfiler.Instance.Enabled = true;
        PerformanceProfiler.Instance.Clear();
        try
        {
            using (CudaDispatchPolicy.Scope("Conv2D", useVendor: false))
            {
                // no-op body; scope disposal records the op
            }
            var generic = PerformanceProfiler.Instance.GetStats("Conv2D.generic");
            Assert.NotNull(generic);
            Assert.Equal(1, generic!.CallCount);
        }
        finally
        {
            PerformanceProfiler.Instance.Clear();
            PerformanceProfiler.Instance.Enabled = false;
        }
    }

    [Fact]
    public void Scope_MatMulWithVendor_LabelsAsCuBLAS()
    {
        PerformanceProfiler.Instance.Enabled = true;
        PerformanceProfiler.Instance.Clear();
        try
        {
            using (CudaDispatchPolicy.Scope("MatMul", useVendor: true)) { }
            Assert.NotNull(PerformanceProfiler.Instance.GetStats("MatMul.cuBLAS"));
            Assert.Null(PerformanceProfiler.Instance.GetStats("MatMul.cuDNN"));
        }
        finally
        {
            PerformanceProfiler.Instance.Clear();
            PerformanceProfiler.Instance.Enabled = false;
        }
    }

    [Fact]
    public void Scope_ConvWithVendor_LabelsAsCuDNN()
    {
        PerformanceProfiler.Instance.Enabled = true;
        PerformanceProfiler.Instance.Clear();
        try
        {
            using (CudaDispatchPolicy.Scope("Conv2D", useVendor: true)) { }
            Assert.NotNull(PerformanceProfiler.Instance.GetStats("Conv2D.cuDNN"));
        }
        finally
        {
            PerformanceProfiler.Instance.Clear();
            PerformanceProfiler.Instance.Enabled = false;
        }
    }
}
