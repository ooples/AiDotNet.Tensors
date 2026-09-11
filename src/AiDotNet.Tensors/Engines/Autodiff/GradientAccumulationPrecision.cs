// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>Precision policy for fan-out gradient additions during backward.</summary>
public enum GradientAccumulationPrecision
{
    /// <summary>Use the active backward autocast precision for gradient additions.</summary>
    InheritBackwardPrecision = 0,

    /// <summary>Perform gradient additions in FP32 while leaving all other backward kernels autocast.</summary>
    Float32 = 1,
}

internal sealed class GradientAccumulationPrecisionScope : IDisposable
{
    private static readonly AsyncLocal<GradientAccumulationPrecisionScope?> CurrentScope = new();
    private readonly GradientAccumulationPrecisionScope? _previous;
    private bool _disposed;

    internal GradientAccumulationPrecisionScope(GradientAccumulationPrecision precision)
    {
        Precision = precision;
        _previous = CurrentScope.Value;
        CurrentScope.Value = this;
    }

    internal GradientAccumulationPrecision Precision { get; }

    internal static IDisposable? EnterFloat32AutocastForAddition()
    {
        if (CurrentScope.Value?.Precision != GradientAccumulationPrecision.Float32
            || !AutocastScope.IsEnabled
            || AutocastScope.ActivePrecision == PrecisionMode.Float32)
        {
            return null;
        }

        return new AutocastScope(PrecisionMode.Float32, AutocastScope.Current?.Policy);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CurrentScope.Value = _previous;
    }
}
