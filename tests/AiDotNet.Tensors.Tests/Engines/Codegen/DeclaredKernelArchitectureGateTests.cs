// Copyright (c) AiDotNet. All rights reserved.
// The production gate for declared programs is not the development gate.
//
// HasCodegenConveyor's own documentation calls itself "a development gate, not a promotion:
// which generated kernel ships on an architecture is still decided by that architecture's own
// release evidence". DeclaredKernelsEnabled used it to route production calls anyway, so a
// lowered program launched anywhere ptxas accepted it -- and docs/CODEGEN_MULTI_ARCH_VALIDATION.md
// says the SM80/86/89/90 matrix is assembly portability evidence only, with the
// second-architecture performance gate "external and remains unsatisfied".
//
// These need no device: the gate is a function of the compute capability.

using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class DeclaredKernelArchitectureGateTests
{
    // The default set only holds while the deployment override is unset; it is read once at type initialisation.
    private static void RequireNoOverride() =>
        Skip.If(!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("AIDOTNET_DECLARED_KERNELS_ARCH")),
            "AIDOTNET_DECLARED_KERNELS_ARCH widens the released set for this process.");

    /// <summary>The two lanes the repository documents as physical: the conveyor's SM75 and GA102 SM86.</summary>
    [SkippableTheory]
    [InlineData(7, 5)]
    [InlineData(8, 6)]
    public void TheDocumentedLanesAreReleased(int major, int minor)
    {
        RequireNoOverride();
        Assert.True(DirectPtxArchitecture.HasReleasedCodegenKernels(major, minor));
    }

    /// <summary>
    /// Assembly portability is not release evidence: SM80, SM89, SM90 and SM100 lower and assemble, and must still
    /// leave the call to the hand-written kernel.
    /// </summary>
    [SkippableTheory]
    [InlineData(8, 0)]
    [InlineData(8, 9)]
    [InlineData(9, 0)]
    [InlineData(10, 0)]
    public void AnArchitectureWithoutItsOwnEvidenceIsNotReleased(int major, int minor)
    {
        RequireNoOverride();
        Assert.True(DirectPtxArchitecture.HasCodegenConveyor(major, minor), "the development gate should admit it");
        Assert.False(DirectPtxArchitecture.HasReleasedCodegenKernels(major, minor));
    }

    /// <summary>Below the conveyor neither gate admits the architecture, override or not.</summary>
    [Theory]
    [InlineData(6, 1)]
    [InlineData(7, 0)]
    public void BelowTheConveyorNeitherGateAdmits(int major, int minor)
    {
        Assert.False(DirectPtxArchitecture.HasCodegenConveyor(major, minor));
        Assert.False(DirectPtxArchitecture.HasReleasedCodegenKernels(major, minor));
    }

    /// <summary>
    /// The released set is a subset of the development set for every capability, whatever the override says: an
    /// architecture that cannot lower cannot ship.
    /// </summary>
    [Fact]
    public void TheReleasedSetIsASubsetOfTheDevelopmentSet()
    {
        for (int major = 5; major <= 12; major++)
            for (int minor = 0; minor <= 9; minor++)
                if (DirectPtxArchitecture.HasReleasedCodegenKernels(major, minor))
                    Assert.True(DirectPtxArchitecture.HasCodegenConveyor(major, minor),
                        $"sm_{major}{minor} is released but not on the conveyor");
    }
}
