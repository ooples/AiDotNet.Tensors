using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// P/Invoke bindings for AMD MIOpen — optimized deep learning primitives for AMD GPUs.
/// Provides fused, auto-tuned kernels for convolution, normalization, pooling,
/// softmax, RNN/LSTM, and dropout that match or exceed cuDNN performance.
/// Install AiDotNet.Native.ROCm NuGet package to provide the native binary.
/// </summary>
internal static class MIOpenNativeBindings
{
    private const string MIOpenLibrary = "MIOpen";

    private static bool _checked;
    private static bool _available;
    private static readonly object _lock = new();

    public static bool IsAvailable
    {
        get
        {
            if (!_checked)
            {
                lock (_lock)
                {
                    if (!_checked)
                    {
                        try
                        {
#if NET5_0_OR_GREATER
                            if (NativeLibrary.TryLoad(MIOpenLibrary, out var handle))
                            {
                                NativeLibrary.Free(handle);
                                _available = true;
                            }
#else
                            var handle = LoadLibrary(MIOpenLibrary);
                            if (handle != IntPtr.Zero) { FreeLibrary(handle); _available = true; }
#endif
                        }
                        catch { _available = false; }
                        _checked = true;
                    }
                }
            }
            return _available;
        }
    }

#if !NET5_0_OR_GREATER
    [DllImport("kernel32", SetLastError = true)] private static extern IntPtr LoadLibrary(string lpFileName);
    [DllImport("kernel32")] private static extern bool FreeLibrary(IntPtr hModule);
#endif

    // =====================================================================
    // MIOpen handle management
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreate")]
    public static extern int Create(out IntPtr handle);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroy")]
    public static extern int Destroy(IntPtr handle);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenSetStream")]
    public static extern int SetStream(IntPtr handle, IntPtr stream);

    // =====================================================================
    // Tensor descriptor
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreateTensorDescriptor")]
    public static extern int CreateTensorDescriptor(out IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroyTensorDescriptor")]
    public static extern int DestroyTensorDescriptor(IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenSet4dTensorDescriptor")]
    public static extern int Set4dTensorDescriptor(
        IntPtr desc,
        int dataType,  // miopenFloat = 1
        int n, int c, int h, int w);

    // =====================================================================
    // Convolution
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreateConvolutionDescriptor")]
    public static extern int CreateConvolutionDescriptor(out IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroyConvolutionDescriptor")]
    public static extern int DestroyConvolutionDescriptor(IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenInitConvolutionDescriptor")]
    public static extern int InitConvolutionDescriptor(
        IntPtr desc,
        int mode,  // miopenConvolution = 0, miopenCrossCorrelation = 1
        int padH, int padW,
        int strideH, int strideW,
        int dilationH, int dilationW);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenFindConvolutionForwardAlgorithm")]
    public static extern int FindConvolutionForwardAlgorithm(
        IntPtr handle,
        IntPtr xDesc, IntPtr x,
        IntPtr wDesc, IntPtr w,
        IntPtr convDesc,
        IntPtr yDesc, IntPtr y,
        int requestedAlgoCount,
        out int returnedAlgoCount,
        IntPtr perfResults,
        IntPtr workSpace, UIntPtr workSpaceSize,
        [MarshalAs(UnmanagedType.Bool)] bool exhaustiveSearch);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenConvolutionForward")]
    public static extern int ConvolutionForward(
        IntPtr handle,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        IntPtr wDesc, IntPtr w,
        IntPtr convDesc,
        int algo,
        ref float beta,
        IntPtr yDesc, IntPtr y,
        IntPtr workSpace, UIntPtr workSpaceSize);

    // =====================================================================
    // Batch Normalization
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenBatchNormalizationForwardTraining")]
    public static extern int BatchNormForwardTraining(
        IntPtr handle,
        int mode,  // miopenBNPerActivation=0, miopenBNSpatial=1
        ref float alpha, ref float beta,
        IntPtr xDesc, IntPtr x,
        IntPtr yDesc, IntPtr y,
        IntPtr bnScaleBiasMeanVarDesc,
        IntPtr bnScale, IntPtr bnBias,
        double expAvgFactor,
        IntPtr resultRunningMean, IntPtr resultRunningVariance,
        double epsilon,
        IntPtr resultSaveMean, IntPtr resultSaveInvVariance);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenBatchNormalizationForwardInference")]
    public static extern int BatchNormForwardInference(
        IntPtr handle,
        int mode,
        ref float alpha, ref float beta,
        IntPtr xDesc, IntPtr x,
        IntPtr yDesc, IntPtr y,
        IntPtr bnScaleBiasMeanVarDesc,
        IntPtr bnScale, IntPtr bnBias,
        IntPtr estimatedMean, IntPtr estimatedVariance,
        double epsilon);

    // =====================================================================
    // Pooling
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreatePoolingDescriptor")]
    public static extern int CreatePoolingDescriptor(out IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroyPoolingDescriptor")]
    public static extern int DestroyPoolingDescriptor(IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenSet2dPoolingDescriptor")]
    public static extern int Set2dPoolingDescriptor(
        IntPtr desc,
        int mode,  // miopenPoolingMax=0, miopenPoolingAverage=1, miopenPoolingAverageInclusive=2
        int windowH, int windowW,
        int padH, int padW,
        int strideH, int strideW);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenPoolingForward")]
    public static extern int PoolingForward(
        IntPtr handle,
        IntPtr poolDesc,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y,
        [MarshalAs(UnmanagedType.Bool)] bool doBackward,
        IntPtr workSpace, UIntPtr workSpaceSize);

    // =====================================================================
    // Activation (ReLU, Sigmoid, Tanh, etc.)
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreateActivationDescriptor")]
    public static extern int CreateActivationDescriptor(out IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroyActivationDescriptor")]
    public static extern int DestroyActivationDescriptor(IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenSetActivationDescriptor")]
    public static extern int SetActivationDescriptor(
        IntPtr desc,
        int mode,  // 0=Passthru, 1=Logistic(sigmoid), 2=Tanh, 3=ReLU, 4=SoftReLU, 5=Abs, 6=Power, 7=ClippedReLU, 8=LeakyReLU, 9=ELU
        double alpha, double beta, double gamma);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenActivationForward")]
    public static extern int ActivationForward(
        IntPtr handle,
        IntPtr activDesc,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    // =====================================================================
    // Softmax
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenSoftmaxForward_V2")]
    public static extern int SoftmaxForward(
        IntPtr handle,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y,
        int algorithm,  // 0=FAST, 1=ACCURATE, 2=LOG
        int mode);      // 0=INSTANCE, 1=CHANNEL

    // =====================================================================
    // Dropout
    // =====================================================================

    [DllImport(MIOpenLibrary, EntryPoint = "miopenCreateDropoutDescriptor")]
    public static extern int CreateDropoutDescriptor(out IntPtr desc);

    [DllImport(MIOpenLibrary, EntryPoint = "miopenDestroyDropoutDescriptor")]
    public static extern int DestroyDropoutDescriptor(IntPtr desc);

    // =====================================================================
    // Constants
    // =====================================================================

    public const int StatusSuccess = 0;
    public const int DataTypeFloat = 1;
    public const int DataTypeHalf = 2;
    public const int DataTypeBFloat16 = 5;
    public const int ConvolutionMode = 0;
    public const int CrossCorrelationMode = 1;
    public const int BNPerActivation = 0;
    public const int BNSpatial = 1;
    public const int PoolingMax = 0;
    public const int PoolingAverage = 1;
    public const int ActivationReLU = 3;
    public const int ActivationSigmoid = 1;
    public const int ActivationTanh = 2;
    public const int SoftmaxFast = 0;
    public const int SoftmaxAccurate = 1;
}
