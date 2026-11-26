// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Seeds for the device generators. The mask kernels hash an element index with a 32-bit key in two rounds, which
/// separates the elements of one mask but not the masks of nearby keys: seeds 1 and 2 agreed on 51% of a 30% dropout
/// mask where independent draws agree on 58%. Callers seed consecutive stochastic ops with consecutive counters, so
/// the seed is mixed first.
/// </summary>
internal static class GpuSeed
{
    /// <summary>
    /// The SplitMix64 output function of <paramref name="seed"/>: a bijection whose outputs for nearby seeds differ in
    /// about half their bits.
    /// </summary>
    public static ulong Mix(ulong seed)
    {
        unchecked
        {
            ulong z = seed + 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }
}
