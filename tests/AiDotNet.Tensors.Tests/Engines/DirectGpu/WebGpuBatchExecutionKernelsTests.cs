#if NET7_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// WebGPU's batch-execution WGSL is a hand port of the shared CUDA kernels (#1142) and no CI host has a WebGPU
/// adapter. WebGPU passes each kernel the same argument list HIP passes the CUDA kernel, buffers in order and
/// every scalar as one 4-byte field, so each port's binding layout must mirror the CUDA signature exactly.
/// </summary>
public sealed class WebGpuBatchExecutionKernelsTests
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(2);

    public static IEnumerable<object[]> KernelNames()
        => BatchExecutionCudaKernels.GetKernelNames().Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(KernelNames))]
    public void Port_MirrorsTheCudaSignature(string name)
    {
        Assert.True(WebGpuBatchExecutionKernels.ByKernelName.TryGetValue(name, out var wgsl), $"{name} has no WGSL port.");

        Match signature = Regex.Match(BatchExecutionCudaKernels.GetSource(), $@"void {name}\(([^)]*)\)",
            RegexOptions.None, RegexTimeout);
        Assert.True(signature.Success, $"{name} is not in the CUDA source.");
        var expectedAccess = new List<string>();
        var expectedFields = new List<string>();
        foreach (string raw in signature.Groups[1].Value.Split(','))
        {
            string param = Regex.Replace(raw.Trim(), @"\s+", " ", RegexOptions.None, RegexTimeout);
            if (param.StartsWith("const float* ", StringComparison.Ordinal) || param.StartsWith("const unsigned int* ", StringComparison.Ordinal)
                || param.StartsWith("const int* ", StringComparison.Ordinal))
                expectedAccess.Add("read");
            else if (param.StartsWith("unsigned int* ", StringComparison.Ordinal) || param.StartsWith("int* ", StringComparison.Ordinal))
                expectedAccess.Add("read_write");
            else if (param.StartsWith("float* ", StringComparison.Ordinal)) expectedAccess.Add("read_write");
            else if (param.StartsWith("int ", StringComparison.Ordinal)) expectedFields.Add("i32");
            else if (param.StartsWith("float ", StringComparison.Ordinal)) expectedFields.Add("f32");
            else if (param.StartsWith("unsigned long long ", StringComparison.Ordinal)) expectedFields.AddRange(new[] { "u32", "u32" });
            else throw new InvalidOperationException($"Unexpected CUDA parameter '{param}' in {name}.");
        }

        var bindings = Regex.Matches(wgsl, @"@binding\((\d+)\) var<storage, (read|read_write)>", RegexOptions.None, RegexTimeout);
        Assert.Equal(expectedAccess, bindings.Select(m => m.Groups[2].Value).ToList());
        Assert.Equal(Enumerable.Range(0, expectedAccess.Count).Select(i => i.ToString()),
            bindings.Select(m => m.Groups[1].Value));
        Assert.Contains($"@binding({expectedAccess.Count}) var<uniform> p : P;", wgsl);

        Match fields = Regex.Match(wgsl, @"struct P \{([^}]*)\}", RegexOptions.None, RegexTimeout);
        Assert.True(fields.Success, $"{name} declares no parameter struct.");
        var actualFields = fields.Groups[1].Value.Split(',')
            .Select(f => f.Split(':')[1].Trim()).ToList();
        Assert.Equal(expectedFields, actualFields);
    }

    [Theory]
    [MemberData(nameof(KernelNames))]
    public void Port_UsesNoCOnlySyntax(string name)
    {
        string wgsl = WebGpuBatchExecutionKernels.ByKernelName[name];
        Assert.Contains("fn main(", wgsl);
        Assert.DoesNotContain("?", wgsl);
        Assert.False(Regex.IsMatch(wgsl, @"\b(float|int|unsigned|uint)\s+\w+\s*[=;]", RegexOptions.None, RegexTimeout),
            $"{name} declares a C-style typed variable.");
    }
}
#endif
