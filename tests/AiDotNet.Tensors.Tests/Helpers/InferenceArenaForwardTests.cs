// Copyright (c) AiDotNet. All rights reserved.
// #653/#657 follow-up — forward caching allocator. A repeated inference forward run inside a
// TensorArena (Reset() between forwards) must (a) produce BIT-IDENTICAL results to the
// non-arena path on every iteration — the arena hands back RentUninitialized buffers holding
// the PRIOR forward's data, so this catches any op that relies on zero-initialised scratch —
// and (b) collapse per-forward GC allocation toward zero once the arena is warm (the
// PyTorch-caching-allocator behaviour: the eager forward's intermediates are recycled instead
// of re-allocated). Measured on the attnblock probe: 13.57 MB/fwd (no arena) -> ~0.26 MB/fwd
// (arena, steady state) — a ~98% reduction.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// Serialize with the other arena tests — TensorArena.Current is [ThreadStatic] and xUnit
// reuses worker threads across collections.
[Collection(nameof(TensorArenaPinnedTests))]
public class InferenceArenaForwardTests
{
    private static readonly CpuEngine Eng = new CpuEngine();

    // A small transformer-ish forward op chain (the attnblock op mix): the same ops whose
    // per-op TensorAllocator.Rent calls route through TensorArena.Current when one is active.
    private static Tensor<float> Forward(Tensor<float> x, Tensor<float> w, Tensor<float> gamma, Tensor<float> beta)
    {
        var h = Eng.BatchMatMul(x, w);                                   // [S,D] x [D,D] -> [S,D]
        var s = Eng.Softmax(h, -1);
        var ln = Eng.LayerNorm(s, gamma, beta, 1e-5, out _, out _);
        return Eng.TensorAdd(ln, h);                                     // residual
    }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float[] Snapshot(Tensor<float> t)
    {
        var a = new float[t.Length];
        for (int i = 0; i < t.Length; i++) a[i] = t[i];
        return a;
    }

    [Fact]
    public void RepeatedForward_InArena_BitIdenticalToNonArena_AcrossResets()
    {
        const int S = 64, D = 128;
        // Inputs (and weights) allocated OUTSIDE the arena, so they survive Reset() and are a
        // safe re-entry point each forward — the consumer pattern (weights/inputs are persistent;
        // only transient intermediates live in the arena).
        var x = Rand(new[] { S, D }, 1);
        var w = Rand(new[] { D, D }, 2);
        var gamma = Rand(new[] { D }, 3);
        var beta = Rand(new[] { D }, 4);

        var reference = Snapshot(Forward(x, w, gamma, beta)); // no arena active

        using var arena = TensorArena.Create();
        for (int iter = 0; iter < 8; iter++)
        {
            arena.Reset(); // recycle the prior forward's intermediates (the buffers now hold stale data)
            var y = Forward(x, w, gamma, beta);
            // Copy the result out BEFORE the next Reset recycles it (the "detach output" pattern
            // a consumer uses to return a prediction from an arena-scoped forward).
            var got = Snapshot(y);
            Assert.Equal(reference.Length, got.Length);
            for (int i = 0; i < reference.Length; i++)
                Assert.Equal(reference[i], got[i]); // bit-exact: reuse must not perturb the result
        }
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void RepeatedForward_InArena_AllocatesFarLessThanNonArena()
    {
        const int S = 128, D = 256, N = 40;
        var x = Rand(new[] { S, D }, 11);
        var w = Rand(new[] { D, D }, 12);
        var gamma = Rand(new[] { D }, 13);
        var beta = Rand(new[] { D }, 14);

        // Warm both paths once (JIT, first-touch pools) so the measurement is steady-state.
        float sink = 0;
        sink += Forward(x, w, gamma, beta)[0];

        long a0 = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < N; i++) sink += Forward(x, w, gamma, beta)[0];
        long noArenaBytes = GC.GetTotalAllocatedBytes(precise: true) - a0;

        using (var arena = TensorArena.Create())
        {
            sink += Forward(x, w, gamma, beta)[0]; // warm the arena (first forward fills it)
            long a1 = GC.GetTotalAllocatedBytes(precise: true);
            for (int i = 0; i < N; i++)
            {
                arena.Reset();
                sink += Forward(x, w, gamma, beta)[0];
            }
            long arenaBytes = GC.GetTotalAllocatedBytes(precise: true) - a1;

            // Conservative gate (real ratio is ~0.05): the arena must cut forward allocation by
            // at least half. This guards the Rent->TensorArena.Current routing for the forward
            // ops — if a hot op regresses to a non-arena allocation path, this trips.
            Assert.True(arenaBytes * 2 < noArenaBytes,
                $"arena forward allocated {arenaBytes} B over {N} forwards; non-arena {noArenaBytes} B — " +
                $"expected arena < half. The per-op TensorAllocator.Rent path is not hitting the arena.");
        }

        if (sink == float.PositiveInfinity) Console.Write(""); // keep sink live
    }
#endif
}
