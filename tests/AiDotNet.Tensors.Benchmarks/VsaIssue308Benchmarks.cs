// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Issue #308 acceptance harness for cross-position HRR sequence primitives.
/// Shape matches the HarmonicEngine L3 isolation target: T=64 slots, D=64
/// split-complex code dimensions, stored as [T, 2D].
/// </summary>
[SimpleJob(RuntimeMoniker.Net10_0, launchCount: 1, warmupCount: 3, iterationCount: 10)]
[MemoryDiagnoser]
[MarkdownExporterAttribute.GitHub]
public class VsaIssue308Benchmarks
{
    private const int Slots = 64;
    private const int CodeDim = 64;
    private const int StateDim = 2 * CodeDim;
    private const int SlotOffset = 2;

    private Tensor<double> _seqA = null!;
    private Tensor<double> _seqB = null!;
    private Tensor<double> _identity = null!;
    private Tensor<double> _shifted = null!;
    private Tensor<double> _scalarOut = null!;
    private Tensor<double> _fusedOut = null!;

    [GlobalSetup]
    public void Setup()
    {
        _seqA = RandomSplitComplex(seed: 30801);
        _seqB = RandomSplitComplex(seed: 30802);
        _identity = BuildComplexIdentitySlot();
        _shifted = new Tensor<double>(new[] { Slots, StateDim });
        _scalarOut = new Tensor<double>(new[] { Slots, StateDim });
        _fusedOut = new Tensor<double>(new[] { Slots, StateDim });
    }

    [Benchmark(Description = "Scalar reference: ShiftSlots")]
    public Tensor<double> ShiftSlots_ScalarReference()
    {
        ShiftSlotsScalar(_seqB.AsSpan(), _identity.AsSpan(), SlotOffset, _scalarOut.AsWritableSpan());
        return _scalarOut;
    }

    [Benchmark(Description = "AiDotNet #308: ShiftSlots span-copy")]
    public Tensor<double> ShiftSlots_SpanCopy()
    {
        CpuVsaOperations.ShiftSlots(_seqB, SlotOffset, _identity, _fusedOut);
        return _fusedOut;
    }

    [Benchmark(Description = "Scalar reference: shifted HRR bind")]
    public Tensor<double> HrrBindShifted_ScalarReference()
    {
        HrrBindShiftedScalar(_seqA.AsSpan(), _seqB.AsSpan(), SlotOffset, _scalarOut.AsWritableSpan());
        return _scalarOut;
    }

    [Benchmark(Description = "AiDotNet #308: fused HrrBindShifted")]
    public Tensor<double> HrrBindShifted_Fused()
    {
        CpuVsaOperations.HrrBindShifted(_seqA, _seqB, SlotOffset, _fusedOut);
        return _fusedOut;
    }

    [Benchmark(Description = "Decomposed: ShiftSlots + scalar HRR bind")]
    public Tensor<double> HrrBindShifted_DecomposedShiftThenBind()
    {
        CpuVsaOperations.ShiftSlots(_seqB, SlotOffset, _identity, _shifted);
        BindAlignedScalar(_seqA.AsSpan(), _shifted.AsSpan(), _scalarOut.AsWritableSpan());
        return _scalarOut;
    }

    private static Tensor<double> RandomSplitComplex(int seed)
    {
        var rng = new Random(seed);
        var data = new double[Slots * StateDim];
        for (int i = 0; i < data.Length; i++)
            data[i] = rng.NextDouble() * 2.0 - 1.0;
        return new Tensor<double>(data, new[] { Slots, StateDim });
    }

    private static Tensor<double> BuildComplexIdentitySlot()
    {
        var identity = new double[StateDim];
        for (int d = 0; d < CodeDim; d++)
            identity[d] = 1.0;
        return new Tensor<double>(identity, new[] { StateDim });
    }

    private static void ShiftSlotsScalar(
        ReadOnlySpan<double> input,
        ReadOnlySpan<double> identity,
        int slotShift,
        Span<double> output)
    {
        for (int t = 0; t < Slots; t++)
        {
            int srcSlot = t - slotShift;
            int dstOff = t * StateDim;
            if ((uint)srcSlot >= (uint)Slots)
            {
                for (int i = 0; i < StateDim; i++)
                    output[dstOff + i] = identity[i];
                continue;
            }

            int srcOff = srcSlot * StateDim;
            for (int i = 0; i < StateDim; i++)
                output[dstOff + i] = input[srcOff + i];
        }
    }

    private static void HrrBindShiftedScalar(
        ReadOnlySpan<double> seqA,
        ReadOnlySpan<double> seqB,
        int slotOffset,
        Span<double> output)
    {
        for (int t = 0; t < Slots; t++)
        {
            int bSlot = t - slotOffset;
            int off = t * StateDim;
            if ((uint)bSlot >= (uint)Slots)
            {
                for (int i = 0; i < StateDim; i++)
                    output[off + i] = seqA[off + i];
                continue;
            }

            BindSlotScalar(seqA, off, seqB, bSlot * StateDim, output);
        }
    }

    private static void BindAlignedScalar(
        ReadOnlySpan<double> seqA,
        ReadOnlySpan<double> seqB,
        Span<double> output)
    {
        for (int t = 0; t < Slots; t++)
            BindSlotScalar(seqA, t * StateDim, seqB, t * StateDim, output);
    }

    private static void BindSlotScalar(
        ReadOnlySpan<double> seqA,
        int aOff,
        ReadOnlySpan<double> seqB,
        int bOff,
        Span<double> output)
    {
        int outOff = aOff;
        for (int d = 0; d < CodeDim; d++)
        {
            double ar = seqA[aOff + d];
            double ai = seqA[aOff + CodeDim + d];
            double br = seqB[bOff + d];
            double bi = seqB[bOff + CodeDim + d];
            output[outOff + d] = ar * br - ai * bi;
            output[outOff + CodeDim + d] = ar * bi + ai * br;
        }
    }
}
