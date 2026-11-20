using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

public sealed partial class CuDnnConvolution
{
    internal bool TryConvBiasResidualGpu(in DirectGpu.ConvBiasResidualArgs args)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnConvolution));
        using var input = new CuDnnTensorDescriptor();
        using var output = new CuDnnTensorDescriptor();
        using var filter = new CuDnnFilterDescriptor();
        using var bias = new CuDnnTensorDescriptor();
        using var convolution = new CuDnnConvolutionDescriptor();
        using var activation = new CuDnnActivationDescriptor();
        input.Set4D(CuDnnNative.CudnnDataType.Float, args.Batch, args.InChannels, args.InHeight, args.InWidth);
        output.Set4D(CuDnnNative.CudnnDataType.Float, args.Batch, args.OutChannels, args.OutHeight, args.OutWidth);
        filter.Set4D(CuDnnNative.CudnnDataType.Float, args.OutChannels, args.InChannels, args.KernelH, args.KernelW);
        bias.Set4D(CuDnnNative.CudnnDataType.Float, 1, args.OutChannels, 1, 1);
        SetConvolution(convolution, args.PadH, args.PadW, args.StrideH, args.StrideW,
            args.DilationH, args.DilationW, CuDnnNative.CudnnMathType.Fma);
        activation.Set(CuDnnNative.CudnnActivationMode.Identity);
        // Identity activation supports only implicit-precomputed GEMM in this cuDNN API.
        const CuDnnNative.CudnnConvolutionFwdAlgo algorithm = CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitPrecompGemm;
        var status = CuDnnNative.cudnnGetConvolutionForwardWorkspaceSize(_context.Handle, input.Handle, filter.Handle,
            convolution.Handle, output.Handle, algorithm, out ulong bytes);
        if (status == CuDnnNative.CudnnStatus.NotSupported) return false;
        CuDnnContext.CheckStatus(status, "ConvBiasResidualWorkspace");
        lock (_context.StreamBindingLock)
        lock (_workspaceLock)
        {
            // The vendor store is optional: when even an exact-size workspace cannot be allocated, decline so the
            // caller's composition path (which needs no cuDNN workspace) runs instead of failing inference.
            if (bytes > 0 && !TryEnsureWorkspaceUnlocked(bytes, exactSize: false, out _)) return false;
            float alpha1 = 1f, alpha2 = 1f;
            status = ConvBiasResidualNative.Forward(_context.Handle, ref alpha1,
                input.Handle, args.Input.Handle, filter.Handle, args.Kernel.Handle,
                convolution.Handle, algorithm, bytes > 0 ? _workspace : IntPtr.Zero, bytes,
                ref alpha2, output.Handle, args.Residual.Handle, bias.Handle, args.Bias.Handle,
                activation.Handle, output.Handle, args.Output.Handle);
            if (status == CuDnnNative.CudnnStatus.NotSupported) return false;
            CuDnnContext.CheckStatus(status, "ConvolutionBiasResidualForward");
            DirectGpu.GpuLaunchProbe.OnLaunch();
            return true;
        }
    }

    private static class ConvBiasResidualNative
    {
        [DllImport("cudnn64_9.dll", EntryPoint = "cudnnConvolutionBiasActivationForward")]
        internal static extern CuDnnNative.CudnnStatus Forward(IntPtr handle, ref float alpha1,
            IntPtr inputDesc, IntPtr input, IntPtr filterDesc, IntPtr filter,
            IntPtr convolutionDesc, CuDnnNative.CudnnConvolutionFwdAlgo algorithm,
            IntPtr workspace, ulong workspaceBytes, ref float alpha2,
            IntPtr residualDesc, IntPtr residual, IntPtr biasDesc, IntPtr bias,
            IntPtr activationDesc, IntPtr outputDesc, IntPtr output);
    }
}
