using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The physical-ABI precondition every exact-shape direct-PTX kernel applies to
/// its tensor views before launch.
/// </summary>
/// <remarks>
/// This check was duplicated verbatim in every kernel of the layer, which meant
/// a change to the contract fields had to be repeated in each copy and could
/// silently drift between them. One implementation keeps the admission rule
/// identical across the family.
///
/// The rule is deliberately exact rather than "at least": an exact-shape kernel
/// bakes its extents into the emitted PTX, so a view that is merely large enough
/// is still the wrong allocation for that kernel.
/// </remarks>
internal static class DirectPtxAbiGuard
{
    /// <summary>
    /// Throws when <paramref name="view"/> does not exactly satisfy
    /// <paramref name="contract"/>.
    /// </summary>
    internal static void Require(
        DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
