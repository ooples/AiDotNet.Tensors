// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The int32 range of the Vulkan, Metal and WebGPU Interpolate2D kernels' integer windows. Their Nearest and Area modes
/// compute source positions as y * Hin and window bounds in units of 1/Hout (products up to Hin * Hout, and the same
/// for widths, and an Area divisor of Hin * Win), because a shader float division is not correctly rounded; WGSL has
/// no 64-bit integers to widen them.
/// </summary>
internal static class IntegerInterpolationWindow
{
    private const int NearestMode = 0, AreaMode = 5;

    /// <summary>
    /// Throws <see cref="NotSupportedException"/>, the engine's signal to fall back to the CPU, when a Nearest or Area
    /// resize has an axis whose input and output sizes multiply past int.MaxValue, or an Area resize an input plane
    /// past it.
    /// </summary>
    internal static void EnsureInRange(int mode, int hin, int win, int hout, int wout)
    {
        if (mode != NearestMode && mode != AreaMode) return;
        if ((long)hin * hout > int.MaxValue || (long)win * wout > int.MaxValue
            || (mode == AreaMode && (long)hin * win > int.MaxValue))
            throw new NotSupportedException(
                $"Interpolate2D mode {mode}: {hin}x{win} -> {hout}x{wout} overflows the kernel's int32 window arithmetic.");
    }
}
