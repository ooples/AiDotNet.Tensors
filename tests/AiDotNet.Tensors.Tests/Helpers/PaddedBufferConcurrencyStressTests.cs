// Copyright (c) AiDotNet. All rights reserved.
// #486 drift hunt: the PaddedBuffer/pre-pack drift passes in isolation but fails
// intermittently under the parallel CI pool. This test deliberately reproduces the
// concurrency pressure locally — many threads running the matmul→sigmoid→matmul chain
// with pooled, garbage-padded operands at once — and asserts every result still
// matches the single-thread fresh reference. If a shared GEMM buffer is being raced,
// this surfaces it as a drift; if it stays green even under heavy local concurrency,
// the contamination is specific to the CI environment (4-vCPU + coverage).

using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class PaddedBufferConcurrencyStressTests
{
    private static Tensor<float> RentWithGarbagePadding(int[] shape, float[] logical, float pad)
    {
        int logicalLen = 1;
        for (int i = 0; i < shape.Length; i++) logicalLen *= shape[i];
        Assert.Equal(logicalLen, logical.Length);

        // Construct an EXPLICITLY OVER-SIZED backing so the matmul pad-respect
        // guard has actual garbage past t.Length to misread. The earlier
        // implementation rented through TensorAllocator.Rent and patched the
        // overflow past t.Length, but in test runs the allocator falls through
        // to `new Tensor<T>(shape)` (TensorPool disabled / ForceFreshAllocations)
        // which produces an exact-fit backing — so no pad ever got written and
        // the test was a false green. Wrap an oversized array via
        // Tensor<float>.FromMemory (the same pattern the arena tier uses) so
        // GetDataArray() returns a backing strictly larger than t.Length.
        const int padSlots = 16;  // enough for any SIMD over-read past t.Length
        var backing = new float[logicalLen + padSlots];
        Array.Copy(logical, backing, logicalLen);
        for (int i = logicalLen; i < backing.Length; i++) backing[i] = pad;
        var t = Tensor<float>.FromMemory(new Memory<float>(backing, 0, logicalLen), shape);

        // Defend against accidental future regressions of the FromMemory wrap
        // semantics: if it ever started returning a backing trimmed to t.Length,
        // every rent here would silently become a false green again.
        var actualBacking = t.GetDataArray();
        Assert.True(actualBacking.Length > t.Length,
            $"Expected padded backing for shape [{string.Join(", ", shape)}], " +
            $"got exact length {actualBacking.Length}; FromMemory contract changed?");
        return t;
    }

    [Fact]
    public void MatMulChain_UnderHeavyConcurrency_StaysBitIdenticalToFreshReference()
    {
        const int batch = 4, hidden = 256, outFeat = 128;
        var rng = new Random(1);
        var inputVals = new float[batch * hidden];
        for (int i = 0; i < inputVals.Length; i++) inputVals[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var wVals = new float[hidden * outFeat];
        for (int i = 0; i < wVals.Length; i++) wVals[i] = (float)(rng.NextDouble() * 0.1);
        var w2Vals = new float[outFeat * hidden];
        for (int i = 0; i < w2Vals.Length; i++) w2Vals[i] = (float)(rng.NextDouble() * 0.1);

        // Single-thread fresh reference (ground truth).
        var engine = new CpuEngine();
        var inF = new Tensor<float>(new[] { batch, hidden }); inputVals.CopyTo(inF.AsWritableSpan());
        var wF = new Tensor<float>(new[] { hidden, outFeat }); wVals.CopyTo(wF.AsWritableSpan());
        var w2F = new Tensor<float>(new[] { outFeat, hidden }); w2Vals.CopyTo(w2F.AsWritableSpan());
        var refOut = engine.TensorMatMul(engine.Sigmoid(engine.TensorMatMul(inF, wF)), w2F).AsSpan().ToArray();

        int threads = Math.Max(4, Environment.ProcessorCount * 2);
        const int itersPerThread = 200;
        var drifts = new ConcurrentBag<string>();

        Parallel.For(0, threads, _ =>
        {
            var eng = new CpuEngine();
            for (int it = 0; it < itersPerThread; it++)
            {
                var inPad = RentWithGarbagePadding(new[] { batch, hidden }, inputVals, 999_999f);
                var wPad = RentWithGarbagePadding(new[] { hidden, outFeat }, wVals, -999_999f);
                var w2Pad = RentWithGarbagePadding(new[] { outFeat, hidden }, w2Vals, 777_777f);
                var outPad = eng.TensorMatMul(eng.Sigmoid(eng.TensorMatMul(inPad, wPad)), w2Pad).AsSpan();
                for (int i = 0; i < refOut.Length; i++)
                    if (outPad[i] != refOut[i])
                    {
                        drifts.Add($"idx {i}: {outPad[i]} vs ref {refOut[i]} (Δ={Math.Abs((double)outPad[i] - refOut[i]):G4})");
                        break;
                    }
            }
        });

        Assert.True(drifts.IsEmpty,
            $"{drifts.Count} concurrent iterations drifted from the fresh reference. First: {(drifts.IsEmpty ? "" : System.Linq.Enumerable.First(drifts))}");
    }
}
