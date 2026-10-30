// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Keys a device-side metadata block (shapes, strides, flags) by its int32 words, compared element-wise, so a cache hit
/// costs the words array the caller already built and nothing else (#1570: these lookups run on every strided gather and
/// every portable attention launch).
/// </summary>
internal sealed class Int32WordsComparer : IEqualityComparer<int[]>
{
    internal static readonly Int32WordsComparer Instance = new();

    private const uint FnvOffsetBasis = 2166136261;
    private const uint FnvPrime = 16777619;

    public bool Equals(int[]? x, int[]? y)
    {
        if (ReferenceEquals(x, y)) return true;
        if (x is null || y is null || x.Length != y.Length) return false;
        for (int i = 0; i < x.Length; i++)
            if (x[i] != y[i]) return false;
        return true;
    }

    public int GetHashCode(int[] words)
    {
        uint h = FnvOffsetBasis;
        foreach (int w in words) { h ^= (uint)w; h *= FnvPrime; }
        return (int)h;
    }

    /// <summary>The words bit-cast into the float-typed allocation the kernels read as int32.</summary>
    internal static float[] AsFloatBits(int[] words, int length)
    {
        var floats = new float[length];
        Buffer.BlockCopy(words, 0, floats, 0, words.Length * sizeof(int));
        return floats;
    }
}
