// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend
{
    public void FusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer baseOutput,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        if (_context == null) throw new InvalidOperationException("OpenCL context not available");
        var k = _kernelCache["issue301_fused_lora_forward"];
        uint arg = 0;
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)baseOutput).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)loraA).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)loraB).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
        k.SetArg(arg++, batchSize);
        k.SetArg(arg++, inputFeatures);
        k.SetArg(arg++, rank);
        k.SetArg(arg++, outputFeatures);
        k.SetArg(arg++, scaling);
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        k.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
        _context.Finish();
    }

    public void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        if (_context == null) throw new InvalidOperationException("OpenCL context not available");
        var k = _kernelCache["issue301_fused_ddim_step"];
        uint arg = 0;
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)xT).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)epsilonTheta).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
        k.SetArg(arg++, size);
        k.SetArg(arg++, alphaBarT);
        k.SetArg(arg++, alphaBarTMinus1);
        if (size <= 0) return;
        k.Execute1D(size, CalculateOptimalWorkGroupSize1D(size));
        _context.Finish();
    }

    public void FusedSparseLinear(
        IGpuBuffer input,
        IGpuBuffer packedCsr,
        IGpuBuffer sparseValues,
        IGpuBuffer bias,
        IGpuBuffer output,
        int batchSize,
        int inputFeatures,
        int outputFeatures,
        int nnz,
        int hasBias,
        int activation)
    {
        if (_context == null) throw new InvalidOperationException("OpenCL context not available");
        var k = _kernelCache["issue301_fused_sparse_linear"];
        uint arg = 0;
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)input).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)packedCsr).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)sparseValues).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)bias).Buffer.Handle);
        k.SetArg(arg++, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
        k.SetArg(arg++, batchSize);
        k.SetArg(arg++, inputFeatures);
        k.SetArg(arg++, outputFeatures);
        k.SetArg(arg++, nnz);
        k.SetArg(arg++, hasBias);
        k.SetArg(arg++, activation);
        int total = batchSize * outputFeatures;
        if (total <= 0) return;
        k.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
        _context.Finish();
    }
}
