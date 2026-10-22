// Copyright (c) AiDotNet. All rights reserved.
// The deterministic scatter-add as a declared program (#890 phase 4e). #890's determinism gap is that a scatter
// accumulates through an atomic float add, which sums in arrival order; these tests hold the program to the summation
// order the repo's deterministic kernels promise (increasing source index) AND assert that no atomic is emitted, since
// a program that quietly took the atomic path would still agree with the reference on a run where nothing collided.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenDeterministicScatterTests
{
    [Theory]
    [InlineData(9, 4, 3)]
    [InlineData(7, 1, 2)]     // every source into one destination: the worst collision case
    [InlineData(5, 6, 1)]     // more destinations than sources: some receive nothing
    public void ScatterAdd_SumsEachDestinationsSourcesInIndexOrder(int count, int destinations, int features)
    {
        // Repeats, a destination nothing targets, and two sentinels that must contribute nowhere.
        var indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = i % 3 == 2 ? -1 : (i * 2) % destinations;
        indices[count - 1] = destinations;                  // one past the end: also a miss

        var source = new double[(long)count * features];
        for (int i = 0; i < source.Length; i++) source[i] = ((i * 37) % 23) - 11;
        var indexTensor = new double[count];
        for (int i = 0; i < count; i++) indexTensor[i] = indices[i];

        var outputs = CodegenBackendPrograms.ScatterAddDeterministic(count, destinations, features)
            .Interpret(new Dictionary<string, double[]> { ["indices"] = indexTensor, ["source"] = source });

        for (int d = 0; d < destinations; d++)
            for (int f = 0; f < features; f++)
            {
                double expected = 0;
                for (int i = 0; i < count; i++)
                    if (indices[i] == d) expected += source[(long)i * features + f];
                Assert.Equal(expected, outputs["out"][d * features + f], 12);
            }
    }

    /// <summary>
    /// The point of the whole inversion: the emitted PTX must contain no atomic. An atomic here would be correct on
    /// average and non-reproducible, which is the gap #890 recorded.
    /// </summary>
    [Fact]
    public void ScatterAdd_EmitsNoAtomic()
    {
        var program = CodegenBackendPrograms.ScatterAddDeterministic(16, 4, 8);
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(program.Passes[0], 7, 5);
        Assert.Equal(0, emitter.AtomicStores);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
        // The shift itself: the loaded index is compared against the destination, not used as an address.
        Assert.Contains("sub.s32", ptx, StringComparison.Ordinal);
        Assert.True(emitter.IndirectIndexLoads > 0, "the index tensor must still be read");
    }

    /// <summary>A segment sum is the same operator with segment ids for indices, so unsorted ids are fine.</summary>
    [Fact]
    public void SegmentSum_AcceptsUnsortedSegmentIds()
    {
        int[] ids = { 2, 0, 2, 1, 0, 2 };
        const int Segments = 3, Features = 2;
        var source = new double[ids.Length * Features];
        for (int i = 0; i < source.Length; i++) source[i] = i + 1;
        var idTensor = new double[ids.Length];
        for (int i = 0; i < ids.Length; i++) idTensor[i] = ids[i];

        var outputs = CodegenBackendPrograms.SegmentSum(ids.Length, Segments, Features)
            .Interpret(new Dictionary<string, double[]> { ["indices"] = idTensor, ["source"] = source });

        for (int s = 0; s < Segments; s++)
            for (int f = 0; f < Features; f++)
            {
                double expected = 0;
                for (int i = 0; i < ids.Length; i++) if (ids[i] == s) expected += source[i * Features + f];
                Assert.Equal(expected, outputs["out"][s * Features + f], 12);
            }
    }

    /// <summary>A shifted index that clamped would hand every miss to destination 0.</summary>
    [Fact]
    public void AShiftedIndexRefusesToClamp()
    {
        Assert.Throws<ArgumentException>(() => new CodegenIndirectIndex(
            0, CodegenAffineExpr.Axis(0), 1, CodegenIndexOutOfRange.Clamp, floatIndices: false,
            offset: CodegenAffineExpr.Axis(1)));
    }
}
