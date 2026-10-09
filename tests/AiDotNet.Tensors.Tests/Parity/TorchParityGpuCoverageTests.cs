using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Parity;

/// <summary>
/// The GPU coverage decision for the PyTorch-parity ops: every host-computed op in a <c>CpuEngine.Torch*</c>
/// partial must be overridden by <see cref="DirectGpuTensorEngine"/> (a device route, or an explicit fall-through
/// that records a fallback), so a new parity op cannot silently run on the host under a GPU engine. Composed ops
/// (<c>OpRegistry.DelegatorOps</c>) run on the device through the primitives they call.
/// </summary>
public class TorchParityGpuCoverageTests
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(5);

    // Type and shape queries that read no tensor data (BroadcastShapes works on int[] shapes).
    private static readonly string[] Metadata =
        { "TensorIsFloatingPoint", "TensorIsSigned", "TensorIsComplex", "TensorIsNonzero", "TensorIsSameSize", "BroadcastShapes" };

    [Fact]
    public void EveryHostComputedParityOp_HasAGpuCoverageDecision()
    {
        string? root = PyTorchParityInventory.FindRepositoryRoot();
        Assert.True(root is not null, "repository root not found");
        var engines = Path.Combine(root ?? string.Empty, "src", "AiDotNet.Tensors", "Engines");
        var files = Directory.GetFiles(engines, "CpuEngine.Torch*.cs");
        Assert.NotEmpty(files);
        var declared = files
            // The identifier just before "(" or "<...>(": any return type (tuples included) and any type-parameter
            // count, generic or not. A pattern requiring "<T>(" and no "(" in the return type missed eight ops.
            .SelectMany(f => Regex.Matches(File.ReadAllText(f), @"public virtual .*?\b(\w+)(?:<[\w, ]+>)?\(", RegexOptions.None, RegexTimeout)
                .Cast<Match>().Select(m => m.Groups[1].Value))
            .Distinct().ToList();
        // new HashSet, not ToHashSet: .NET Framework 4.7.1 has no Enumerable.ToHashSet.
        var overridden = new HashSet<string>(typeof(DirectGpuTensorEngine)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(m => m.Name), StringComparer.Ordinal);
        var missing = declared
            .Where(n => !OpRegistry.DelegatorOps.Contains(n) && !Metadata.Contains(n) && !overridden.Contains(n))
            .OrderBy(n => n, StringComparer.Ordinal).ToList();
        Assert.True(missing.Count == 0,
            "host-computed parity ops with no DirectGpuTensorEngine override (add a device route or an explicit "
            + "fall-through in DirectGpuTensorEngine.TorchParity.cs): " + string.Join(", ", missing));
        Assert.True(declared.Count > 100, $"only {declared.Count} parity virtuals found; the source scan is broken");
    }
}
