// Copyright (c) AiDotNet. All rights reserved.
// Issue #311 regression — ensure pooled-padded tensors and freshly-
// allocated tensors with identical logical content produce bit-
// identical forward output. The original symptom was a 3-4% drift in
// DeepBoltzmannMachine.Predict between an in-place-trained model and
// its Clone()'d copy: identical params byte-for-byte, but padding-
// region garbage in the trained-tensor path leaked into downstream
// SIMD reads via overhang and biased the matmul/sigmoid chain.

using System;
using System.Buffers;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

public class PaddedBufferDeterminismTests
{
    /// <summary>
    /// Forces an ArrayPool-backed Tensor whose padding region we then
    /// fill with high-magnitude garbage. After the fix, padding is
    /// always zeroed by Rent — so we have to bypass it via reflection
    /// to reproduce the original failure mode and assert the kernel
    /// no longer reads it.
    /// </summary>
    private static Tensor<float> RentWithGarbagePadding(int[] shape, float[] logicalValues, float padValue)
    {
        // Rent through TensorAllocator so the resulting tensor has the
        // pooled, potentially-padded backing array.
        var t = TensorAllocator.Rent<float>(shape);
        var span = t.AsWritableSpan();
        Assert.Equal(logicalValues.Length, span.Length);
        for (int i = 0; i < logicalValues.Length; i++) span[i] = logicalValues[i];

        // Reach the underlying T[] (may exceed Length) and write
        // garbage into the padding region. If pooling didn't pad this
        // shape we just return the tensor — the test still verifies
        // determinism vs the fresh path.
        var backing = t.GetDataArray();
        if (backing.Length > t.Length)
        {
            for (int i = t.Length; i < backing.Length; i++)
                backing[i] = padValue;
        }
        return t;
    }

    private static Tensor<float> FreshTensor(int[] shape, float[] logicalValues)
    {
        // Bypass TensorAllocator entirely — get a non-padded backing
        // array so this is the canonical reference path.
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        Assert.Equal(logicalValues.Length, span.Length);
        for (int i = 0; i < logicalValues.Length; i++) span[i] = logicalValues[i];
        return t;
    }

    /// <summary>
    /// Full forward chain (matmul → sigmoid → matmul) with all
    /// operands allocated via ArrayPool with garbage padding vs via
    /// the fresh CLR allocator. Same logical values; outputs after
    /// the SECOND matmul must match bit-for-bit.
    ///
    /// <para>Two matmuls (not one) so we exercise the original DBM
    /// drift path: a sigmoid output (post-pool-allocated) is then
    /// fed back into another matmul, and any padding-leak in the
    /// second-stage matmul accumulates the divergence introduced
    /// by the first stage's sigmoid output.</para>
    /// </summary>
    [Theory]
    [InlineData(8, 64, 32)]
    [InlineData(16, 128, 64)]
    [InlineData(4, 256, 128)]
    public void MatMulSigmoidChain_PooledPaddedAndFreshTensor_ProduceIdenticalOutput(
        int batch, int hidden, int outFeat)
    {
        var rng = new Random(1);
        var inputVals = new float[batch * hidden];
        for (int i = 0; i < inputVals.Length; i++) inputVals[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var weightsVals = new float[hidden * outFeat];
        for (int i = 0; i < weightsVals.Length; i++) weightsVals[i] = (float)(rng.NextDouble() * 0.1);

        // Second-stage weights map [outFeat] → [hidden] so the chain
        // has shape [batch, hidden] → [batch, outFeat] → [batch, hidden].
        var weights2Vals = new float[outFeat * hidden];
        for (int i = 0; i < weights2Vals.Length; i++) weights2Vals[i] = (float)(rng.NextDouble() * 0.1);

        // PADDED path — every operand rented from the pool with
        // large-magnitude garbage written into its padding region so
        // any overhang read is detectable.
        var inputPad = RentWithGarbagePadding(new[] { batch, hidden }, inputVals, padValue: 999_999f);
        var weightsPad = RentWithGarbagePadding(new[] { hidden, outFeat }, weightsVals, padValue: -999_999f);
        var weights2Pad = RentWithGarbagePadding(new[] { outFeat, hidden }, weights2Vals, padValue: 777_777f);

        // FRESH path — non-padded backing storage on every operand.
        var inputFresh = FreshTensor(new[] { batch, hidden }, inputVals);
        var weightsFresh = FreshTensor(new[] { hidden, outFeat }, weightsVals);
        var weights2Fresh = FreshTensor(new[] { outFeat, hidden }, weights2Vals);

        var engine = new CpuEngine();
        // Stage 1: matmul + sigmoid.
        var z1Pad = engine.TensorMatMul(inputPad, weightsPad);
        var a1Pad = engine.Sigmoid(z1Pad);
        var z1Fresh = engine.TensorMatMul(inputFresh, weightsFresh);
        var a1Fresh = engine.Sigmoid(z1Fresh);
        // Stage 2: feed the (pool-allocated) sigmoid output back into
        // another matmul — the part that previously produced DBM drift.
        var z2Pad = engine.TensorMatMul(a1Pad, weights2Pad);
        var z2Fresh = engine.TensorMatMul(a1Fresh, weights2Fresh);

        var padSpan = z2Pad.AsSpan();
        var freshSpan = z2Fresh.AsSpan();
        Assert.Equal(freshSpan.Length, padSpan.Length);
        for (int i = 0; i < padSpan.Length; i++)
        {
            Assert.True(padSpan[i] == freshSpan[i],
                $"Mismatch at idx {i}: pooled-padded={padSpan[i]} vs fresh={freshSpan[i]} — "
                + "padding region is leaking into the matmul→sigmoid→matmul output.");
        }
    }

    /// <summary>
    /// Direct allocator-level invariant: a freshly-rented tensor must
    /// have its full backing array zeroed (including the padding region)
    /// — that's the contract issue #311 enforces.
    /// </summary>
    [Theory]
    [InlineData(401_408)]
    [InlineData(1_000_000)]
    public void Rent_ZerosPaddingRegionForValueTypes(int totalSize)
    {
        // Pre-warm the pool with a large rent that we taint, then
        // return — this gives the next Rent a recycled, dirty buffer.
        // If padding-zeroing is missing, the next Rent's padding will
        // hold our taint values.
        var first = TensorAllocator.Rent<float>(new[] { totalSize });
        var firstBacking = first.GetDataArray();
        for (int i = 0; i < firstBacking.Length; i++) firstBacking[i] = 0xCAFEf;

        // Force the buffer back into the pool via TensorPool.Return
        // (the convention used by the engine to recycle).
        TensorPool.Return(first);

        // Re-rent at the same logical size and check the padding bytes.
        var second = TensorAllocator.Rent<float>(new[] { totalSize });
        var secondBacking = second.GetDataArray();

        // Logical region must be zeroed (Rent contract).
        for (int i = 0; i < second.Length; i++)
            Assert.True(secondBacking[i] == 0f,
                $"Logical idx {i} not zero: {secondBacking[i]}.");

        // Padding region must be zeroed too (issue #311).
        for (int i = second.Length; i < secondBacking.Length; i++)
            Assert.True(secondBacking[i] == 0f,
                $"Padding idx {i} not zero — leaks prior renter's garbage: {secondBacking[i]}.");
    }
}
