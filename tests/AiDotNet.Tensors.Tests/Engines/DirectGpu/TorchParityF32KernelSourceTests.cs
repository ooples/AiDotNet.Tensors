using System;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The FP32 parity source (#1125, #1128) as MSL requires it: a kernel function cannot be overloaded, so no
/// __device__ helper may share a kernel's name. WGSL hides the clash (every kernel becomes its own module's main), and
/// a C++ compile accepts it as an overload, so only Metal's compiler would reject it.
/// </summary>
public sealed class TorchParityF32KernelSourceTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(5);

    [Fact]
    public void NoHelper_SharesAKernelsName()
    {
        string source = TorchParityF32Kernels.GetSource();
        var kernels = Regex.Matches(source, @"__global__ __launch_bounds__\(256\) void (\w+)\(", RegexOptions.None, Timeout)
            .Cast<Match>().Select(m => m.Groups[1].Value).Distinct(StringComparer.Ordinal).ToArray();
        var helpers = Regex.Matches(source, @"__device__ __forceinline__ (?:unsigned int|\w+) (\w+)\(", RegexOptions.None, Timeout)
            .Cast<Match>().Select(m => m.Groups[1].Value).Distinct(StringComparer.Ordinal).ToArray();
        Assert.True(kernels.Length > 50, $"only {kernels.Length} kernels parsed");
        Assert.True(helpers.Length > 50, $"only {helpers.Length} helpers parsed");
        var clashes = helpers.Intersect(kernels, StringComparer.Ordinal).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        Assert.True(clashes.Length == 0, "helpers named like kernels: " + string.Join(", ", clashes));
    }

    [Fact]
    public void EveryUnaryAndBinaryOp_HasAKernel()
    {
        string source = TorchParityF32Kernels.GetSource();
        foreach (TorchParityUnaryOp op in Enum.GetValues(typeof(TorchParityUnaryOp)))
            Assert.Contains("void " + PortableTorchParity.UnaryKernel(op) + "(", source);
        foreach (TorchParityBinaryOp op in Enum.GetValues(typeof(TorchParityBinaryOp)))
            Assert.Contains("void " + PortableTorchParity.BinaryKernel(op) + "(", source);
    }
}