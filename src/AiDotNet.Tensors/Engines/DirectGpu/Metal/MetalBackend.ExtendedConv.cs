namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #775: Metal implementations of the per-family extended-conv capability interfaces. Added incrementally
// (interface + kernels) so a partially-ported backend opts into exactly the families it can run; the
// engine routes the rest to the CPU. MSL kernels live in the ExtendedConv library
// (MetalExtendedConvKernels); this partial only wires the dispatch.
public sealed partial class MetalBackend : ITrilinearInterpolationKernels
{
    private const string ExtendedConvLibName = "ExtendedConv";

    public void TrilinearInterpolate(IGpuBuffer grid, IGpuBuffer positions, IGpuBuffer output,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(p * c);
        if (total <= 0) return;
        var pipeline = GetPipeline(ExtendedConvLibName, _extendedConvLibrary, "trilinear_interpolate");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)grid, 0);
        encoder.SetBuffer((MetalGpuBuffer)positions, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(d, 3); encoder.SetBytes(h, 4); encoder.SetBytes(w, 5);
        encoder.SetBytes(c, 6); encoder.SetBytes(p, 7); encoder.SetBytes(upperEps, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void TrilinearInterpolateBackward(IGpuBuffer gradOutput, IGpuBuffer positions, IGpuBuffer gradGrid,
        int d, int h, int w, int c, int p, float upperEps)
    {
        int total = checked(d * h * w * c);
        if (total <= 0) return;
        var pipeline = GetPipeline(ExtendedConvLibName, _extendedConvLibrary, "trilinear_interpolate_backward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)gradOutput, 0);
        encoder.SetBuffer((MetalGpuBuffer)positions, 1);
        encoder.SetBuffer((MetalGpuBuffer)gradGrid, 2);
        encoder.SetBytes(d, 3); encoder.SetBytes(h, 4); encoder.SetBytes(w, 5);
        encoder.SetBytes(c, 6); encoder.SetBytes(p, 7); encoder.SetBytes(upperEps, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
