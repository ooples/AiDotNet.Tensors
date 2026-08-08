namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Shared validation for the physical tensor ABI consumed by direct PTX kernels.</summary>
internal static class DirectPtxAbi
{
    internal static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
        => Require(view, contract, parameter, allowLargerView: false);

    internal static void RequireAtLeast(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
        => Require(view, contract, parameter, allowLargerView: true);

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter,
        bool allowLargerView)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            (allowLargerView
                ? view.ByteLength < contract.RequiredBytes
                : view.ByteLength != contract.RequiredBytes))
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
