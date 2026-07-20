using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Pins the SELU backward branch at the zero boundary across CPU and GPU.
///
/// CpuEngine computes deriv = x &gt;= 0 ? scale : scale*alpha*exp(x). The OpenCL and WebGpu kernels matched
/// that, but the CUDA, HIP and Metal kernels used a strict `x &gt; 0` — while the CUDA caller in
/// DirectGpuTensorEngine carried a comment asserting the kernel already used `&gt;=`. Measured divergence was
/// 1.712E-01 at x==0 and 1.804E-01 at x==-0 (a factor of alpha), and exactly zero everywhere else.
///
/// SIGNED ZERO IS THE INTERESTING CASE: `-0.0 &gt;= 0` is TRUE in IEEE while `-0.0 &gt; 0` is false, so negative
/// zero silently took the exponential branch. Random-input tests cannot catch either case — the difference
/// is measure-zero — which is why this pins the boundary explicitly.
/// </summary>
[Collection("OpParity")]
public class SeluBackwardZeroBoundaryTests
{
    private readonly OpParityFixture _fx;
    public SeluBackwardZeroBoundaryTests(OpParityFixture fx) => _fx = fx;

    private const float Scale = 1.0507009873554805f;

    [Fact]
    public void Cpu_uses_the_positive_branch_at_both_zeros()
    {
        var input = new Tensor<float>(new[] { 0f, -0f }, new[] { 2 });
        var grad = new Tensor<float>(new[] { 1f, 1f }, new[] { 2 });

        var r = _fx.Cpu.SeluBackward(grad, input);

        // deriv(0) must be `scale`, NOT scale*alpha — the >= branch.
        Assert.Equal(Scale, r[0], 5);
        Assert.Equal(Scale, r[1], 5);
    }

    [SkippableFact]
    public void Gpu_matches_cpu_at_both_zeros()
    {
        Skip.If(!_fx.GpuReady, "No DirectGpu backend available.");
        var gpu = _fx.Gpu;
        Skip.If(gpu is null, "GPU engine unavailable.");

        var input = new Tensor<float>(new[] { 0f, -0f, 0.5f, -0.5f }, new[] { 4 });
        var grad = new Tensor<float>(new[] { 1f, 1f, 1f, 1f }, new[] { 4 });

        var cpu = _fx.Cpu.SeluBackward(grad, input);
        var dev = gpu!.SeluBackward(grad, input);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(cpu[i] - dev[i]) <= 1e-6,
                $"SeluBackward diverged at x={input[i]}: cpu={cpu[i]:G9} gpu={dev[i]:G9}. The GPU kernel must "
                + "use `x >= 0` to match CpuEngine; a strict `x > 0` takes the exponential branch at zero and "
                + "at NEGATIVE zero, which is a factor-of-alpha error.");
        }
    }
}
