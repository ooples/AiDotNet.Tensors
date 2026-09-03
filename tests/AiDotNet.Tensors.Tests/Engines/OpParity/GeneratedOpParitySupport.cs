// Copyright (c) AiDotNet. All rights reserved.
// Runtime bridge for the generated per-op parity facts (#775). The source generator emits one
// [SkippableFact] per tensor-returning IEngine op that calls into here; this looks up the
// registered parity case(s) for that op and runs the harness, or skips with a visible NEEDS-SPEC
// message so the test explorer shows the full IEngine surface and exactly which ops still lack a spec.
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public static class GeneratedOpParitySupport
{
    private static readonly ILookup<string, OpCase> ByMethod =
        OpParityRegistry.All().ToLookup(o => o.OpMethod, System.StringComparer.Ordinal);

    /// <summary>Run every registered parity case that exercises the given IEngine op method; if none
    /// is registered yet, skip with a NEEDS-SPEC note (so full-surface coverage stays auditable).</summary>
    public static void RunForwardByMethod(string opMethod, OpParityFixture fx)
    {
        var cases = ByMethod[opMethod].ToList();
        Skip.If(cases.Count == 0, $"NEEDS SPEC: IEngine.{opMethod} has no CPU-vs-GPU parity case yet.");
        foreach (var c in cases)
        {
            switch (c.TensorOutputContract)
            {
                case TensorOutputContract.HomogeneousMultiple:
                    OpParityHarness.CheckMultipleOutputs(c, fx);
                    break;
                case TensorOutputContract.HeterogeneousMultiple:
                    OpParityHarness.CheckHeterogeneousOutputs(c, fx);
                    break;
                default:
                    OpParityHarness.CheckForward(c, fx);
                    break;
            }
        }
    }

    /// <summary>
    /// Called by source-generated tests for every IEngine operation with more than one tensor
    /// result. The registry must explicitly describe the tuple and, when homogeneous, expose every
    /// result through the executable multi-output delegates.
    /// </summary>
    public static void VerifyTensorOutputContract(
        string opMethod,
        TensorOutputContract expected,
        TensorOutputOverload expectedOverload)
    {
        var cases = ByMethod[opMethod].ToList();
        Assert.NotEmpty(cases);
        Assert.Contains(cases, op =>
            op.TensorOutputContract == expected &&
            (expectedOverload == TensorOutputOverload.Unspecified ||
                op.TensorOutputOverload == expectedOverload) &&
            (expected == TensorOutputContract.HomogeneousMultiple
                ? op.HasMultipleOutputs
                : op.HasHeterogeneousOutputs));
    }

    /// <summary>
    /// Lets reflection-driven GPU coverage consume the same generated multi-output contract
    /// scaffold. No method signature is copied into a string allowlist: the reflected method
    /// must itself expose homogeneous tensor outputs, and the typed registry contract must
    /// provide executable delegates for every result.
    /// </summary>
    public static bool HasGeneratedHomogeneousOutputCoverage(MethodInfo method)
    {
        if (method is null ||
            !GeneratedTensorOutputContractCatalog.TryGetHomogeneousOverload(
                method, out TensorOutputOverload expectedOverload))
            return false;

        return ByMethod[method.Name].Any(op =>
            op.TensorOutputContract == TensorOutputContract.HomogeneousMultiple &&
            op.HasMultipleOutputs &&
            op.TensorOutputOverload == expectedOverload);
    }

    /// <summary>
    /// Called by source-generated tests for every IEngine overload whose tensor operands/results
    /// do not all share one element type. The registry must classify the exact overload so its
    /// executable graph-capture case proves that compilation fails closed.
    /// </summary>
    public static void VerifyGraphCaptureSignature(
        string opMethod,
        GraphCaptureSignatureConstraint expected,
        GraphCaptureSignatureOverload expectedOverload)
    {
        var cases = ByMethod[opMethod].ToList();
        Assert.NotEmpty(cases);
        bool found = cases.Any(op =>
            op.GraphCaptureSignatureConstraint == expected &&
            (expectedOverload == GraphCaptureSignatureOverload.Unspecified ||
                op.GraphCaptureSignatureOverload == expectedOverload));
        Assert.True(found,
            $"IEngine.{opMethod} requires {expected}/{expectedOverload}; registered cases: " +
            string.Join(", ", cases.Select(op =>
                $"{op.Name}={op.GraphCaptureSignatureConstraint}/{op.GraphCaptureSignatureOverload}")));
    }
}
#endif
