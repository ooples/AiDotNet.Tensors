namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IConvBiasResidualBackend
{
    private long _convBiasResidualVendorLaunchCount;
    internal long ConvBiasResidualVendorLaunchCount => System.Threading.Interlocked.Read(ref _convBiasResidualVendorLaunchCount);

    bool IConvBiasResidualBackend.TryConvBiasResidual(in ConvBiasResidualArgs args)
    {
        if (!args.TryCount(out int count)) return false;
        if (CudaDispatchPolicy.UseCudnnForConv && !CudaDispatchPolicy.AllowTF32
            && !CudaDispatchPolicy.ConvReducedPrecisionTF32
            && !IsStreamCapturing())
        {
            using var current = PushContext();
            EnsureCudnnConv();
            lock (_cudnnContext!.StreamBindingLock)
            {
                _cudnnContext.BindStream(_stream);
                if (_cudnnConv!.TryConvBiasResidualGpu(args))
                {
                    long launches = System.Threading.Interlocked.Increment(ref _convBiasResidualVendorLaunchCount);
                    if (launches == 1 && Environment.GetEnvironmentVariable("AIDOTNET_GPU_CONV_BIAS_RESIDUAL_TRACE") == "1")
                        Console.Error.WriteLine("[CONV-BIAS-RESIDUAL] CUDA cuDNN fused store succeeded; strict FP32 FMA.");
                    if (args.WriteActivation) LeakyRelu(args.Output, args.Activated, args.Alpha, count);
                    return true;
                }
            }
        }
        return ConvBiasResidualComposition.TryExecute(this, args);
    }
}
