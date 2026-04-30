// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.NumericOperations;

/// <summary>Bit-level helpers for <see cref="Half"/>. .NET 5+ has
/// <c>BitConverter.HalfToUInt16Bits</c> but net471 doesn't; this helper
/// keeps the call site portable.</summary>
internal static class HalfBits
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ushort GetBits(Half value)
    {
#if NET5_0_OR_GREATER
        return BitConverter.HalfToUInt16Bits(value);
#else
        // Half is a 16-bit struct on every TFM that ships it. Bit-cast.
        Half local = value;
        return *(ushort*)&local;
#endif
    }
}
