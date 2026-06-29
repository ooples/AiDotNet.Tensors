// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1715: when a caller materializes MANY streaming weights and holds them all resident at once — a
/// foundation-scale GetParameters round-trip / Clone — the pool only accounts its own byte[] snapshots,
/// so without owner-resident bounding the per-weight resident copies accumulate unbounded and OOM
/// before the snapshot-budget eviction fires. WeightRegistry now tracks materialized owners in an LRU
/// and sheds the coldest (write-back the current bytes, then drop the resident copy) once their bytes
/// exceed the budget. These tests pin BOTH halves of correctness: the held set stays bounded, AND a
/// MUTATION made to a since-shed weight survives the shed (write-back) so re-materializing reads it back.
/// </summary>
public class StreamingMaterializedOwnerBoundTests
{
    private const int L = 4096;                 // 16 KB per fp32 weight
    private const int N = 16;                   // 256 KB total — 2x the pool cap
    private const long Cap = 8L * L * sizeof(float); // resident cap = 8 weights; owner budget = cap/2 = 4

    private static float Base(int i, int j) => (i * 31 + j) % 251 * 0.5f - 60f;

    [Fact]
    public void ManyMaterializedOwners_StayBounded_AndMutationsSurviveShed()
    {
        var dir = Path.Combine(Path.GetTempPath(), "aidotnet-1715-" + Guid.NewGuid().ToString("N"));
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingBackingStorePath = dir,
                StreamingPoolMaxResidentBytes = Cap,
                TransparentAutoEviction = true,
                // Full precision so the write-back round-trip is exact (the round-trip contract test
                // asserts exact equality; bf16/lossy stores would lose the mutation's low bits).
                StreamingStoreDtype = StreamingStoreDtype.FullPrecision,
            });

            // Register N streaming weights. RegisterWeight snapshots each into the pool and drops its
            // resident copy, so nothing is resident yet.
            var weights = new Tensor<float>[N];
            for (int i = 0; i < N; i++)
            {
                var t = new Tensor<float>(new[] { L });
                for (int j = 0; j < L; j++) t[j] = Base(i, j);
                t.Lifetime = WeightLifetime.Streaming;
                WeightRegistry.RegisterWeight(t);
                Assert.True(t.StreamingPoolHandle >= 0);
                weights[i] = t;
            }

            // Materialize each and MUTATE element 0 — holding every reference, exactly as a parameter
            // round-trip's write pass does. The owner-resident bounding must shed the cold ones.
            for (int i = 0; i < N; i++)
            {
                WeightRegistry.Materialize(weights[i]);
                Assert.Equal(L, weights[i].DataVector.Length); // resident right after materialize
                weights[i][0] = 1000f + i;                     // dirty the resident copy
            }

            // BOUNDED: not all N owner copies are still resident — the budget (cap/2 = 4 weights)
            // forced the coldest to shed. (Without the fix all N stay resident → the OOM mode.)
            int residentNow = weights.Count(w => w.DataVector.Length == L);
            Assert.True(residentNow < N,
                $"Expected the materialized-owner budget to shed cold owners, but all {N} are still resident.");
            // Assert against the OWNER budget (Cap/2), not the full pool cap — the implementation
            // budgets materialized owners at Cap/2, so checking Cap would still pass if the owner set
            // regressed to twice the intended limit.
            Assert.True(residentNow * (long)L * sizeof(float) <= Cap / 2,
                $"Resident owner set ({residentNow} weights) exceeds the owner budget ({Cap / 2} bytes).");

            // EXACT WRITE-BACK: re-materialize each weight and confirm the mutation survived the shed
            // (shed wrote the dirty bytes back to the pool/disk before dropping), and the untouched
            // elements are intact.
            for (int i = 0; i < N; i++)
            {
                WeightRegistry.Materialize(weights[i]);
                Assert.Equal(L, weights[i].DataVector.Length);
                Assert.Equal(1000f + i, weights[i][0]);     // the mutation
                Assert.Equal(Base(i, 1), weights[i][1]);    // an untouched element
                Assert.Equal(Base(i, L - 1), weights[i][L - 1]);
            }
        }
        finally
        {
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
