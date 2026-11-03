// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static class CompressedMomentBufferValidation
{
    internal static int CheckedInt8WorkSize(int numBlocks)
    {
        long workSize = (long)numBlocks * 256;
        if (numBlocks < 0 || workSize > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(numBlocks), "The int8 optimizer dispatch is too large.");
        return (int)workSize;
    }

    internal static void ValidateInt8(int paramLength, int blockSize, int numBlocks,
        IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer mQuant, IGpuBuffer vQuant, IGpuBuffer mScales, IGpuBuffer vScales)
    {
        if (paramLength <= 0) throw new ArgumentOutOfRangeException(nameof(paramLength));
        if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));
        if (numBlocks != (int)(((long)paramLength + blockSize - 1) / blockSize))
            throw new ArgumentException("numBlocks must match the parameter length and block size.", nameof(numBlocks));
        RequireCapacity(param, (long)paramLength * sizeof(float), nameof(param));
        RequireCapacity(gradient, (long)paramLength * sizeof(float), nameof(gradient));
        RequireCapacity(mQuant, paramLength, nameof(mQuant));
        RequireCapacity(vQuant, paramLength, nameof(vQuant));
        RequireCapacity(mScales, (long)numBlocks * sizeof(float), nameof(mScales));
        RequireCapacity(vScales, (long)numBlocks * sizeof(float), nameof(vScales));
    }

    internal static void RequireCapacity(IGpuBuffer buffer, long requiredBytes, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (buffer.SizeInBytes < requiredBytes)
            throw new ArgumentException($"Buffer needs at least {requiredBytes} bytes; capacity is {buffer.SizeInBytes}.", name);
    }
}
