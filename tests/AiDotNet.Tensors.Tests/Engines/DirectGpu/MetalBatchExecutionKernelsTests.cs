using System;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Metal's batch-execution MSL is translated from the shared CUDA source (#1142) and only compiles on Apple
/// hardware, so this runs everywhere and checks the translation itself: every kernel is declared, no CUDA-only
/// construct survives, and every local pointer carries an address space.
/// </summary>
public sealed class MetalBatchExecutionKernelsTests
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(2);

    [Fact]
    public void Translation_DeclaresEveryKernel()
    {
        string msl = MetalBatchExecutionKernels.Source;
        foreach (string name in BatchExecutionCudaKernels.GetKernelNames())
            Assert.Contains($"kernel void {name}(", msl);
    }

    [Theory]
    [InlineData("__global__")]
    [InlineData("__shared__")]
    [InlineData("__syncthreads")]
    [InlineData("__device__")]
    [InlineData("blockIdx")]
    [InlineData("threadIdx")]
    [InlineData("blockDim")]
    [InlineData("long long")]
    [InlineData("extern \"C\"")]
    [InlineData("bx_gid()")]
    [InlineData("device const device")]
    [InlineData("device device")]
    public void Translation_KeepsNoCudaOnlyConstruct(string token)
    {
        Assert.DoesNotContain(token, MetalBatchExecutionKernels.Source);
    }

    [Fact]
    public void Translation_RenamesSinglePrecisionIntrinsics()
    {
        Assert.False(Regex.IsMatch(MetalBatchExecutionKernels.Source,
            @"\b(exp|log|sqrt|fmax|fmin|fabs|tanh|rsqrt|sin|cos)f\(", RegexOptions.None, RegexTimeout));
    }

    [Fact]
    public void Translation_GivesEveryLocalPointerAnAddressSpace()
    {
        // Per line, not one pattern over the whole source: a leading [^\n]* is retried from every position of every
        // line, which on the .NET Framework interpreter ran past the regex timeout under a loaded full-suite run.
        foreach (string line in MetalBatchExecutionKernels.Source.Split('\n'))
            if (Regex.IsMatch(line, @"\bfloat\* \w+ =", RegexOptions.None, RegexTimeout))
                Assert.StartsWith("device ", line.TrimStart());
    }
}
