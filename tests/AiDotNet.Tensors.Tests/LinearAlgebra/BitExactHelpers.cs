// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// net471-safe float↔int32 bit reinterpretation for bit-exact streaming-store tests.
/// <see cref="BitConverter.SingleToInt32Bits"/> / <see cref="BitConverter.Int32BitsToSingle"/>
/// only exist on .NET Core 2.0+/.NET Standard 2.1+, not net471 — so the cross-TFM test
/// assembly routes through these instead.
/// </summary>
internal static class BitExactHelpers
{
    public static int SingleBits(float value)
#if NET5_0_OR_GREATER
        => BitConverter.SingleToInt32Bits(value);
#else
        => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
#endif

    public static float BitsSingle(int bits)
#if NET5_0_OR_GREATER
        => BitConverter.Int32BitsToSingle(bits);
#else
        => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
#endif
}
