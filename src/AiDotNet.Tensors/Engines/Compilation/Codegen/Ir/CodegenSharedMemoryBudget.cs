// Copyright (c) AiDotNet. All rights reserved.

using System.Globalization;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>The portable static shared-memory contract for generated CUDA kernels.</summary>
internal static class CodegenSharedMemoryBudget
{
    internal const int DoubleBufferStages = 2;

    /// <summary>
    /// Static shared memory available without a device-specific opt-in attribute.
    /// </summary>
    internal const int MaximumStaticBytes = 48 * 1024;

    internal static bool Fits(long requestedBytes, out string reason)
    {
        if (requestedBytes <= MaximumStaticBytes)
        {
            reason = "eligible";
            return true;
        }

        reason = requestedBytes.ToString(CultureInfo.InvariantCulture) +
            " bytes of static shared memory exceed the " +
            MaximumStaticBytes.ToString(CultureInfo.InvariantCulture) + "-byte budget";
        return false;
    }
}
