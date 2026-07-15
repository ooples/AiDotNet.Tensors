namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #775: Metal implementations of the per-family extended-conv capability interfaces. Added incrementally
// (interface + kernels) so a partially-ported backend opts into exactly the families it can run; the
// engine routes the rest to the CPU. MSL kernels live in the ExtendedConv library
// (MetalExtendedConvKernels); this partial only wires the dispatch.
public sealed partial class MetalBackend : ITrilinearInterpolationKernels, IConvTranspose3DKernels, ISpiralConvKernels
{
    private const string ExtendedConvLibName = "ExtendedConv";

    public void SpiralConv(IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer biases, IGpuBuffer output, int v, int inC, int spiralLength, int outC)
    {
        int total = checked(v * outC);
        if (total <= 0) return;
        var pipeline = GetPipeline(ExtendedConvLibName, _extendedConvLibrary, "spiral_conv");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)vertexFeatures, 0);
        encoder.SetBuffer((MetalGpuBuffer)spiralIndices, 1);
        encoder.SetBuffer((MetalGpuBuffer)weights, 2);
        encoder.SetBuffer((MetalGpuBuffer)biases, 3);
        encoder.SetBuffer((MetalGpuBuffer)output, 4);
        encoder.SetBytes(v, 5); encoder.SetBytes(inC, 6); encoder.SetBytes(spiralLength, 7); encoder.SetBytes(outC, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void SpiralConvBackwardInput(IGpuBuffer gradOutput, IGpuBuffer spiralIndices, IGpuBuffer weights,
        IGpuBuffer gradVertexFeatures, int v, int inC, int spiralLength, int outC) =>
        DispatchSpiral4("spiral_conv_backward_input", checked(v * inC),
            gradOutput, spiralIndices, weights, gradVertexFeatures, v, inC, spiralLength, outC);

    public void SpiralConvBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer vertexFeatures, IGpuBuffer spiralIndices,
        IGpuBuffer gradWeights, int v, int inC, int spiralLength, int outC) =>
        DispatchSpiral4("spiral_conv_backward_weights", checked(outC * inC * spiralLength),
            gradOutput, vertexFeatures, spiralIndices, gradWeights, v, inC, spiralLength, outC);

    private void DispatchSpiral4(string name, int total,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, IGpuBuffer d,
        int v, int inC, int spiralLength, int outC)
    {
        if (total <= 0) return;
        var pipeline = GetPipeline(ExtendedConvLibName, _extendedConvLibrary, name);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)c, 2);
        encoder.SetBuffer((MetalGpuBuffer)d, 3);
        encoder.SetBytes(v, 4); encoder.SetBytes(inC, 5); encoder.SetBytes(spiralLength, 6); encoder.SetBytes(outC, 7);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ConvTranspose3D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        DispatchConv3DGrid("conv_transpose3d", checked(n * outC * outD * outH * outW),
            input, weights, output, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        DispatchConv3DGrid("conv_transpose3d_backward_input", checked(n * inC * iD * iH * iW),
            gradOutput, weights, gradInput, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    public void ConvTranspose3DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW) =>
        DispatchConv3DGrid("conv_transpose3d_backward_weights", checked(inC * outC * kD * kH * kW),
            gradOutput, input, gradWeights, n, inC, iD, iH, iW, outC, outD, outH, outW,
            kD, kH, kW, strideD, strideH, strideW, padD, padH, padW);

    private void DispatchConv3DGrid(string name, int total,
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c,
        int n, int inC, int iD, int iH, int iW, int outC, int outD, int outH, int outW,
        int kD, int kH, int kW, int strideD, int strideH, int strideW, int padD, int padH, int padW)
    {
        if (total <= 0) return;
        var pipeline = GetPipeline(ExtendedConvLibName, _extendedConvLibrary, name);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)a, 0);
        encoder.SetBuffer((MetalGpuBuffer)b, 1);
        encoder.SetBuffer((MetalGpuBuffer)c, 2);
        encoder.SetBytes(n, 3); encoder.SetBytes(inC, 4);
        encoder.SetBytes(iD, 5); encoder.SetBytes(iH, 6); encoder.SetBytes(iW, 7);
        encoder.SetBytes(outC, 8); encoder.SetBytes(outD, 9); encoder.SetBytes(outH, 10); encoder.SetBytes(outW, 11);
        encoder.SetBytes(kD, 12); encoder.SetBytes(kH, 13); encoder.SetBytes(kW, 14);
        encoder.SetBytes(strideD, 15); encoder.SetBytes(strideH, 16); encoder.SetBytes(strideW, 17);
        encoder.SetBytes(padD, 18); encoder.SetBytes(padH, 19); encoder.SetBytes(padW, 20);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

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
