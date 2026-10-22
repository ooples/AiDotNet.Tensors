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

    /// <summary>
    /// The accumulating variant adds onto the destination's existing value, which the caller binds to both the seed
    /// and the output. The scanning kernel it replaces does the same, one thread per destination cell.
    /// </summary>
    [Theory]
    [InlineData(9, 4, 3)]
    [InlineData(6, 1, 1)]
    public void ScatterAddAccumulate_AddsOntoTheSeed(int count, int destinations, int features)
    {
        var indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = i % 4 == 3 ? -1 : (i * 3) % destinations;
        var source = new double[(long)count * features];
        for (int i = 0; i < source.Length; i++) source[i] = ((i * 29) % 17) - 8;
        var seed = new double[(long)destinations * features];
        for (int i = 0; i < seed.Length; i++) seed[i] = 100 + i;
        var idTensor = new double[count];
        for (int i = 0; i < count; i++) idTensor[i] = indices[i];

        var outputs = CodegenBackendPrograms.ScatterAddAccumulateDeterministic(count, destinations, features)
            .Interpret(new Dictionary<string, double[]>
                { ["indices"] = idTensor, ["source"] = source, ["seed"] = seed });

        for (int d = 0; d < destinations; d++)
            for (int f = 0; f < features; f++)
            {
                double expected = seed[d * features + f];
                for (int i = 0; i < count; i++)
                    if (indices[i] == d) expected += source[(long)i * features + f];
                Assert.Equal(expected, outputs["out"][d * features + f], 12);
            }
    }

    /// <summary>
    /// The segment maximum and minimum over unsorted ids, with an empty segment left at each reduction's identity --
    /// not at zero, which for a max over negative values would be a wrong answer that looks plausible.
    /// </summary>
    [Fact]
    public void SegmentMaxAndMin_ReduceEachSegmentAndLeaveAnEmptyOneAtTheIdentity()
    {
        int[] ids = { 2, 0, 2, 3, 0, 2 };        // segment 1 receives nothing
        const int Segments = 4, Features = 2;
        var source = new double[ids.Length * Features];
        // All negative, so a zero identity would win a maximum it should not.
        for (int i = 0; i < source.Length; i++) source[i] = -(i + 1);
        var idTensor = new double[ids.Length];
        for (int i = 0; i < ids.Length; i++) idTensor[i] = ids[i];
        var inputs = new Dictionary<string, double[]> { ["indices"] = idTensor, ["source"] = source };

        var maxima = CodegenBackendPrograms.SegmentMax(ids.Length, Segments, Features).Interpret(inputs);
        var minima = CodegenBackendPrograms.SegmentMin(ids.Length, Segments, Features).Interpret(inputs);

        for (int s = 0; s < Segments; s++)
            for (int f = 0; f < Features; f++)
            {
                double high = double.NegativeInfinity, low = double.PositiveInfinity;
                for (int i = 0; i < ids.Length; i++)
                    if (ids[i] == s)
                    {
                        high = Math.Max(high, source[i * Features + f]);
                        low = Math.Min(low, source[i * Features + f]);
                    }
                Assert.Equal(high, maxima["out"][s * Features + f], 12);
                Assert.Equal(low, minima["out"][s * Features + f], 12);
            }
        Assert.Equal(double.NegativeInfinity, maxima["out"][1 * Features]);
        Assert.Equal(double.PositiveInfinity, minima["out"][1 * Features]);
    }

    /// <summary>The segment reductions must be atomic-free for the same reason the scatter-add is.</summary>
    [Fact]
    public void SegmentReductions_EmitNoAtomic()
    {
        foreach (var program in new[]
                 {
                     CodegenBackendPrograms.SegmentMax(16, 4, 8),
                     CodegenBackendPrograms.SegmentMin(16, 4, 8),
                     CodegenBackendPrograms.ScatterAddAccumulateDeterministic(16, 4, 8),
                 })
        {
            var emitter = new PtxAffineEmitter();
            string ptx = emitter.Emit(program.Passes[0], 7, 5);
            Assert.Equal(0, emitter.AtomicStores);
            Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("atom.", ptx, StringComparison.Ordinal);
            Assert.True(emitter.IndirectIndexLoads > 0, program.Passes[0].Name + " must read the index tensor");
        }
        // The minimum reduces with a maximum over negated terms, so its padding identity must be -inf: a 0 fill
        // would make an empty segment return -0 instead of +inf, and a segment of positives return 0.
        var minPtx = new PtxAffineEmitter().Emit(CodegenBackendPrograms.SegmentMin(16, 4, 8).Passes[0], 7, 5);
        Assert.Contains("0fFF800000", minPtx, StringComparison.Ordinal);
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
