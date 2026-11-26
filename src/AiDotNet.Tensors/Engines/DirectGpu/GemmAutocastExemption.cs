// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Marks a GEMM the library issues for an operation that is not a matrix product, such as a sum reduction or a
/// broadcast written as a product with a ones vector. Under PyTorch's autocast policy those operations run in FP32,
/// so while this thread holds the exemption its GEMMs ignore an open matmul autocast scope.
/// </summary>
/// <remarks>
/// Per thread: an exempt GEMM is issued synchronously by the operation that opened the exemption. Every backend
/// reads its autocast state on the calling thread before the GEMM launches. Open the exemption immediately around the
/// backend call, inside any deferred delegate, and not around code that runs later.
/// </remarks>
internal static class GemmAutocastExemption
{
    [System.ThreadStatic] private static int t_depth;

    /// <summary>Whether this thread's GEMMs ignore an open matmul autocast scope.</summary>
    internal static bool Active => t_depth > 0;

    /// <summary>Exempts this thread's GEMMs until the returned scope is disposed. Nests.</summary>
    internal static Scope Enter()
    {
        t_depth++;
        return new Scope(true);
    }

    /// <summary>Ends one level of <see cref="Enter"/>.</summary>
    internal readonly struct Scope : System.IDisposable
    {
        private readonly bool _entered;
        internal Scope(bool entered) => _entered = entered;

        public void Dispose()
        {
            if (_entered && t_depth > 0) t_depth--;
        }
    }
}
