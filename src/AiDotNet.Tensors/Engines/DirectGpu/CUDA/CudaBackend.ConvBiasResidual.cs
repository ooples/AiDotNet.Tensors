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
                    System.Threading.Interlocked.Increment(ref _convBiasResidualVendorLaunchCount);
                    if (args.WriteActivation) LeakyRelu(args.Output, args.Activated, args.Alpha, count);
                    return true;
                }
            }
        }
        return ConvBiasResidualComposition.TryExecute(this, args);
    }
}
