// Copyright (c) AiDotNet. All rights reserved.
// The geometry a convolution node carries.
//
// Kept as a typed record rather than an int[] because these four numbers are not
// interchangeable and a transposed pair is invisible in an array. The index-map layer
// already turns them into affine expressions -- Window(spatial, tap, stride, pad) for the
// forward direction and TransposedWindow for the adjoint -- so this type exists only to
// carry them from the graph to that layer intact.

using System;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Geometry of a convolution node, stored in its attribute slot.</summary>
/// <param name="StrideHeight">Row step between output positions.</param>
/// <param name="StrideWidth">Column step between output positions.</param>
/// <param name="PadHeight">Rows of implicit zero padding on each side.</param>
/// <param name="PadWidth">Columns of implicit zero padding on each side.</param>
public sealed record CodegenConvAttributes(
    int StrideHeight, int StrideWidth, int PadHeight, int PadWidth)
{
    /// <summary>Unit stride with no padding.</summary>
    public static CodegenConvAttributes Valid { get; } = new(1, 1, 0, 0);

    /// <summary>Unit stride with one-pixel padding, which keeps a 3x3 output size.</summary>
    public static CodegenConvAttributes Same3x3 { get; } = new(1, 1, 1, 1);

    /// <summary>Same stride and padding on both axes.</summary>
    public static CodegenConvAttributes Uniform(int stride, int padding) =>
        new(stride, stride, padding, padding);

    /// <summary>
    /// Output extent along one spatial axis, or -1 when the geometry produces nothing.
    /// </summary>
    /// <param name="input">Input extent.</param>
    /// <param name="tap">Kernel extent along the same axis.</param>
    /// <param name="stride">Step between output positions.</param>
    /// <param name="padding">Implicit padding on each side.</param>
    public static int ForwardExtent(int input, int tap, int stride, int padding)
    {
        if (stride <= 0) return -1;

        // The span must be checked BEFORE the division. C# truncates toward zero, so a
        // kernel larger than its padded input gives a negative span that divides to 0 and
        // then reports one phantom output row: input 2, tap 5, pad 1, stride 2 is
        // (2 + 2 - 5) / 2 + 1 = -1/2 + 1 = 1, when nothing fits at all.
        int span = input + 2 * padding - tap;
        if (span < 0) return -1;

        return span / stride + 1;
    }

    /// <summary>Output extent of a transposed convolution along one spatial axis.</summary>
    /// <param name="input">Input extent.</param>
    /// <param name="tap">Kernel extent along the same axis.</param>
    /// <param name="stride">Upsampling factor.</param>
    /// <param name="padding">Implicit padding on each side.</param>
    public static int TransposedExtent(int input, int tap, int stride, int padding)
    {
        if (stride <= 0) return -1;
        int extent = (input - 1) * stride + tap - 2 * padding;
        return extent > 0 ? extent : -1;
    }

    /// <summary>Throws when the geometry could not produce a positive output.</summary>
    public void Validate()
    {
        if (StrideHeight <= 0 || StrideWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(StrideHeight),
                "Convolution stride must be positive; got " + StrideHeight + "x" + StrideWidth + ".");
        if (PadHeight < 0 || PadWidth < 0)
            throw new ArgumentOutOfRangeException(nameof(PadHeight),
                "Convolution padding cannot be negative; got " + PadHeight + "x" + PadWidth + ".");
    }
}
