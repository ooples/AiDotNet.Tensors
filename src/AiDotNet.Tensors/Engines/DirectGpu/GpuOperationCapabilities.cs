// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>A family of device operations a backend either runs natively for an element type or does not.</summary>
public enum GpuOperationFamily
{
    /// <summary>Holding buffers of the type: allocation, typed transfer and device-to-device copy.</summary>
    Storage,
    /// <summary>Element-wise arithmetic and unary functions.</summary>
    Elementwise,
    /// <summary>Reductions to a scalar (sum, dot product).</summary>
    Reduction,
    /// <summary>Dense matrix multiplication.</summary>
    MatrixMultiply,
}

/// <summary>One row of a backend's capability table.</summary>
public readonly struct GpuOperationCapability : IEquatable<GpuOperationCapability>
{
    /// <summary>Creates a row.</summary>
    public GpuOperationCapability(GpuOperationFamily family, GpuElementType elementType, bool isNative)
    {
        Family = family;
        ElementType = elementType;
        IsNative = isNative;
    }

    /// <summary>The operation family.</summary>
    public GpuOperationFamily Family { get; }
    /// <summary>The element type the row is about.</summary>
    public GpuElementType ElementType { get; }
    /// <summary>Whether the backend computes the family in <see cref="ElementType"/> on the device. Where it does not,
    /// the engine runs the operation on the CPU in the tensor's own type; it never narrows the type to fit a kernel.</summary>
    public bool IsNative { get; }

    /// <inheritdoc />
    public bool Equals(GpuOperationCapability other)
        => Family == other.Family && ElementType == other.ElementType && IsNative == other.IsNative;
    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is GpuOperationCapability other && Equals(other);
    /// <inheritdoc />
    public override int GetHashCode() => ((int)Family * 397 ^ (int)ElementType) * 2 + (IsNative ? 1 : 0);
    /// <inheritdoc />
    public override string ToString() => $"{Family}<{ElementType}>: {(IsNative ? "native" : "CPU")}";
}

/// <summary>
/// What a backend runs natively, by operation family and element type: it answers "does this op stay on the device for
/// this type" for a caller deciding where to place data. It reads the same capability members the engine routes on.
/// </summary>
public static class GpuOperationCapabilities
{
    private static readonly GpuOperationFamily[] Families =
    {
        GpuOperationFamily.Storage, GpuOperationFamily.Elementwise, GpuOperationFamily.Reduction, GpuOperationFamily.MatrixMultiply,
    };

    private static readonly GpuElementType[] Types =
    {
        GpuElementType.Float32, GpuElementType.Float64, GpuElementType.Float16, GpuElementType.BFloat16, GpuElementType.Int8,
        GpuElementType.UInt8, GpuElementType.Int16, GpuElementType.Int32, GpuElementType.Int64, GpuElementType.Bool,
    };

    /// <summary>
    /// Whether <paramref name="backend"/> runs <paramref name="family"/> natively over <paramref name="elementType"/>
    /// buffers on its current device. A wrapper answers for the backend it wraps.
    /// </summary>
    public static bool SupportsOperation(this IDirectGpuBackend backend, GpuOperationFamily family, GpuElementType elementType)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (!Enum.IsDefined(typeof(GpuOperationFamily), family))
            throw new ArgumentOutOfRangeException(nameof(family), family, "Unknown operation family.");
        var physical = Unwrap(backend);
        if (elementType == GpuElementType.Unspecified || !physical.SupportsElementType(elementType)) return false;
        if (family == GpuOperationFamily.Storage) return true;
        return elementType switch
        {
            // The float kernels are the backend contract itself.
            GpuElementType.Float32 => true,
            GpuElementType.Float64 => family switch
            {
                GpuOperationFamily.Elementwise => physical is IGpuFloat64Elementwise { SupportsFloat64Elementwise: true },
                GpuOperationFamily.Reduction => physical is IGpuFloat64Reductions { SupportsFloat64Reductions: true },
                GpuOperationFamily.MatrixMultiply => physical is IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true },
                _ => false,
            },
            // Half, integer and bool buffers are held and moved, not yet computed on.
            _ => false,
        };
    }

    /// <summary>The whole table for <paramref name="backend"/>: every family for every element type, in a fixed order.</summary>
    public static IReadOnlyList<GpuOperationCapability> DescribeOperations(this IDirectGpuBackend backend)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        var rows = new List<GpuOperationCapability>(Families.Length * Types.Length);
        foreach (var type in Types)
            foreach (var family in Families)
                rows.Add(new GpuOperationCapability(family, type, backend.SupportsOperation(family, type)));
        return rows;
    }

    internal static IDirectGpuBackend Unwrap(IDirectGpuBackend backend)
    {
        while (backend is Gpu.DelegatingGpuBackend wrapper) backend = wrapper.InnerBackend;
        return backend;
    }
}
