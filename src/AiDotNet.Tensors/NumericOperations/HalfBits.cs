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
        // The net471 compatibility Half stores a float, so it must be converted
        // to IEEE-754 binary16 rather than reinterpreted as a 16-bit struct.
        float single = (float)value;
        uint bits = *(uint*)&single;
        uint sign = (bits >> 16) & 0x8000u;
        uint exponent = (bits >> 23) & 0xffu;
        uint mantissa = bits & 0x7fffffu;

        if (exponent == 0xffu)
        {
            if (mantissa == 0) return (ushort)(sign | 0x7c00u);
            uint payload = mantissa >> 13;
            return (ushort)(sign | 0x7c00u | payload | (payload == 0 ? 1u : 0u));
        }

        int halfExponent = (int)exponent - 127 + 15;
        if (halfExponent >= 31) return (ushort)(sign | 0x7c00u);
        if (halfExponent <= 0)
        {
            if (halfExponent < -10) return (ushort)sign;
            mantissa |= 0x800000u;
            int shift = 14 - halfExponent;
            uint rounded = mantissa >> shift;
            uint remainder = mantissa & ((1u << shift) - 1u);
            uint halfway = 1u << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0))
                rounded++;
            return (ushort)(sign | rounded);
        }

        uint halfMantissa = mantissa >> 13;
        uint normalRemainder = mantissa & 0x1fffu;
        if (normalRemainder > 0x1000u ||
            (normalRemainder == 0x1000u && (halfMantissa & 1u) != 0))
        {
            halfMantissa++;
            if (halfMantissa == 0x400u)
            {
                halfMantissa = 0;
                halfExponent++;
                if (halfExponent >= 31) return (ushort)(sign | 0x7c00u);
            }
        }
        return (ushort)(sign | ((uint)halfExponent << 10) | halfMantissa);
#endif
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Half FromBits(ushort raw)
    {
#if NET5_0_OR_GREATER
        return BitConverter.UInt16BitsToHalf(raw);
#else
        uint sign = ((uint)raw & 0x8000u) << 16;
        uint exponent = ((uint)raw >> 10) & 0x1fu;
        uint mantissa = (uint)raw & 0x3ffu;
        uint bits;
        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                int unbiasedExponent = -14;
                while ((mantissa & 0x400u) == 0)
                {
                    mantissa <<= 1;
                    unbiasedExponent--;
                }
                mantissa &= 0x3ffu;
                bits = sign | ((uint)(unbiasedExponent + 127) << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1fu)
        {
            bits = sign | 0x7f800000u | (mantissa << 13);
        }
        else
        {
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        }
        float single = *(float*)&bits;
        return (Half)single;
#endif
    }
}
