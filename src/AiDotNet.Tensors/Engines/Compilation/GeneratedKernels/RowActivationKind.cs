// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;

/// <summary>
/// The activation a fused add/bias + activation step applies. This is the one definition of the dispatch protocol:
/// the fusion pattern, its plan codec, the generated CPU row kernel and the generated CUDA kernel all read it (the
/// CUDA source gets one <c>AIDN_ACT_*</c> define per value, built from this enum), and the integer values are the
/// stored plan format, so existing values must not be renumbered.
/// </summary>
internal enum RowActivationKind
{
    None = 0,
    Relu = 1,
    Gelu = 2,
    Swish = 3,
    Sigmoid = 4,
    Tanh = 5,
}

internal static class RowActivationKindProtocol
{
    /// <summary>A real activation (not <see cref="RowActivationKind.None"/>) with a defined value.</summary>
    internal static bool IsActivation(int value) =>
        value != (int)RowActivationKind.None && Enum.IsDefined(typeof(RowActivationKind), value);

    /// <summary><c>#define AIDN_ACT_RELU 1</c> and so on, one per enum value, for the generated CUDA source.</summary>
    internal static readonly string CudaDefines = BuildCudaDefines();

    private static string BuildCudaDefines()
    {
        var sb = new StringBuilder();
        foreach (RowActivationKind kind in Enum.GetValues(typeof(RowActivationKind)))
        {
            sb.Append("#define AIDN_ACT_").Append(kind.ToString().ToUpperInvariant()).Append(' ')
              .Append(((int)kind).ToString(System.Globalization.CultureInfo.InvariantCulture)).Append('\n');
        }
        return sb.ToString();
    }
}
