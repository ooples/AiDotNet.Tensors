using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

public sealed partial class CuDnnConvolution
{
    internal bool TryGroupedConv2DForwardGpu(IntPtr input, IntPtr kernel, IntPtr output,
        in GroupedConv2DGeometry geometry, CuDnnNative.CudnnMathType mathType)
        => TryGroupedConvolutionGpu(0, input, kernel, output, geometry, false, mathType);

    internal bool TryGroupedConv2DBackwardDataGpu(IntPtr gradOutput, IntPtr kernel, IntPtr gradInput,
        in GroupedConv2DGeometry geometry, bool accumulate, CuDnnNative.CudnnMathType mathType)
        => TryGroupedConvolutionGpu(1, gradOutput, kernel, gradInput, geometry, accumulate, mathType);

    internal bool TryGroupedConv2DBackwardFilterGpu(IntPtr input, IntPtr gradOutput, IntPtr gradKernel,
        in GroupedConv2DGeometry geometry, bool accumulate, CuDnnNative.CudnnMathType mathType)
        => TryGroupedConvolutionGpu(2, input, gradOutput, gradKernel, geometry, accumulate, mathType);

    // The pointers are logical input/kernel/output for forward, dY/kernel/dX for data, and input/dY/dW for filter.
    // A false result is possible only before a convolution launch; execution errors cannot trigger an additive retry.
    private bool TryGroupedConvolutionGpu(int pass, IntPtr first, IntPtr second, IntPtr destination,
        in GroupedConv2DGeometry g, bool accumulate, CuDnnNative.CudnnMathType mathType)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnConvolution));
        using var inputDesc = new CuDnnTensorDescriptor();
        using var filterDesc = new CuDnnFilterDescriptor();
        using var convDesc = new CuDnnConvolutionDescriptor();
        using var outputDesc = new CuDnnTensorDescriptor();
        inputDesc.Set4D(CuDnnNative.CudnnDataType.Float, g.Batch, g.InChannels, g.InHeight, g.InWidth);
        filterDesc.Set4D(CuDnnNative.CudnnDataType.Float, g.OutChannels, g.InChannelsPerGroup, g.KernelH, g.KernelW);
        outputDesc.Set4D(CuDnnNative.CudnnDataType.Float, g.Batch, g.OutChannels, g.OutHeight, g.OutWidth);
        SetConvolution(convDesc, g.PadH, g.PadW, g.StrideH, g.StrideW, g.DilationH, g.DilationW, mathType);
        convDesc.SetGroupCount(g.Groups);

        lock (_context.StreamBindingLock)
        {
            var shapeStatus = CuDnnNative.cudnnGetConvolution2dForwardOutputDim(convDesc.Handle, inputDesc.Handle,
                filterDesc.Handle, out int n, out int c, out int h, out int w);
            if (shapeStatus != CuDnnNative.CudnnStatus.Success) return false;
            if (n != g.Batch || c != g.OutChannels || h != g.OutHeight || w != g.OutWidth)
                throw new InvalidOperationException("cuDNN grouped output geometry disagrees with the validated logical buffers.");

            // Groups, compact filter geometry, dtype and math policy all participate in the shared bounded cache.
            // Backward algorithms are queried against these complete descriptors on every call; no ungrouped key is reused.
            int algorithm = pass == 0
                ? (int)HeuristicForwardAlgorithm(inputDesc.Handle, filterDesc.Handle, convDesc.Handle, outputDesc.Handle,
                    mathType, (g.Batch, g.InChannels, g.InHeight, g.InWidth, g.OutChannels, g.KernelH, g.KernelW,
                        g.PadH, g.PadW, g.StrideH, g.StrideW, g.DilationH, g.DilationW, g.Groups,
                        (int)CuDnnNative.CudnnDataType.Float, (int)mathType))
                : pass == 1 ? (int)CuDnnNative.CudnnConvolutionBwdDataAlgo.Algo1
                : (int)CuDnnNative.CudnnConvolutionBwdFilterAlgo.Algo1;

            CuDnnNative.CudnnStatus Workspace(out ulong bytes) => pass switch
            {
                0 => CuDnnNative.cudnnGetConvolutionForwardWorkspaceSize(_context.Handle, inputDesc.Handle,
                    filterDesc.Handle, convDesc.Handle, outputDesc.Handle, (CuDnnNative.CudnnConvolutionFwdAlgo)algorithm, out bytes),
                1 => CuDnnNative.cudnnGetConvolutionBackwardDataWorkspaceSize(_context.Handle, filterDesc.Handle,
                    outputDesc.Handle, convDesc.Handle, inputDesc.Handle, (CuDnnNative.CudnnConvolutionBwdDataAlgo)algorithm, out bytes),
                _ => CuDnnNative.cudnnGetConvolutionBackwardFilterWorkspaceSize(_context.Handle, inputDesc.Handle,
                    outputDesc.Handle, convDesc.Handle, filterDesc.Handle, (CuDnnNative.CudnnConvolutionBwdFilterAlgo)algorithm, out bytes),
            };

            var query = Workspace(out ulong workspaceBytes);
            if (pass == 0 && (query != CuDnnNative.CudnnStatus.Success || workspaceBytes > HeuristicMaxWorkspace)
                && algorithm != (int)CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitGemm)
            {
                // A bounded second native query, not an execution fallback: destination has not been written.
                algorithm = (int)CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitGemm;
                query = Workspace(out workspaceBytes);
            }
            if (query != CuDnnNative.CudnnStatus.Success || workspaceBytes > HeuristicMaxWorkspace) return false;

            lock (_workspaceLock)
            {
                if (workspaceBytes > 0) EnsureWorkspaceUnlocked(workspaceBytes);
                IntPtr workspace = workspaceBytes == 0 ? IntPtr.Zero : _workspace;
                float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
                // beta0 overwrites the requested target, including its first compiled write; beta1 preserves prior consumers.
                var status = pass switch
                {
                    0 => CuDnnNative.cudnnConvolutionForward(_context.Handle, ref alpha, inputDesc.Handle, first,
                        filterDesc.Handle, second, convDesc.Handle, (CuDnnNative.CudnnConvolutionFwdAlgo)algorithm,
                        workspace, workspaceBytes, ref beta, outputDesc.Handle, destination),
                    1 => CuDnnNative.cudnnConvolutionBackwardData(_context.Handle, ref alpha, filterDesc.Handle, second,
                        outputDesc.Handle, first, convDesc.Handle, (CuDnnNative.CudnnConvolutionBwdDataAlgo)algorithm,
                        workspace, workspaceBytes, ref beta, inputDesc.Handle, destination),
                    _ => CuDnnNative.cudnnConvolutionBackwardFilter(_context.Handle, ref alpha, inputDesc.Handle, first,
                        outputDesc.Handle, second, convDesc.Handle, (CuDnnNative.CudnnConvolutionBwdFilterAlgo)algorithm,
                        workspace, workspaceBytes, ref beta, filterDesc.Handle, destination),
                };
                CuDnnContext.CheckStatus(status, "GroupedConvolutionPass" + pass);
                GpuLaunchProbe.OnLaunch();
                return true;
            }
        }
    }
}
