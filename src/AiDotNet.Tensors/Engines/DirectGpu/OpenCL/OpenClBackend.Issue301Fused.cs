// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend
{
    private void EnsureIssue301KernelsAvailable(string opName)
    {
        if (_context == null) throw new InvalidOperationException("OpenCL context not available");
        if (!_issue301KernelsAvailable)
            throw new NotSupportedException(
                $"OpenCL Issue #301 fused kernels are not available on this device — " +
                $"compile failed during backend initialization. {opName} cannot be dispatched. " +
                $"Fall back to the eager decomposed path.");
    }

    private DirectOpenClContext RequireContext(string opName)
    {
        EnsureIssue301KernelsAvailable(opName);
        // EnsureIssue301KernelsAvailable already validated _context is non-null;
        // this restates it locally so net471's nullable analysis is satisfied
        // without depending on [MemberNotNull] (which net471 cannot honour).
        return _context ?? throw new InvalidOperationException("OpenCL context not available");
    }

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
        var ctx = RequireContext(nameof(FusedLoRAForward));
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
        ctx.Finish();
    }

    public void FusedDDIMStep(
        IGpuBuffer xT,
        IGpuBuffer epsilonTheta,
        IGpuBuffer output,
        int size,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        var ctx = RequireContext(nameof(FusedDDIMStep));
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
        ctx.Finish();
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
        var ctx = RequireContext(nameof(FusedSparseLinear));
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
        ctx.Finish();
    }
}
