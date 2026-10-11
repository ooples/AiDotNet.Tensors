using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Mixed precision compilation pass — reduces compute precision for FP16-safe ops.
///
/// For each FP16-safe step, quantizes the input data in-place to Half precision
/// (fp32→Half→fp32 round-trip) before execution, then restores the original values
/// afterward. This reduces effective precision of intermediate computations.
///
/// Limitation: since compiled steps use pinned GCHandles to the original arrays,
/// the quantization must happen in-place. Original values are saved to thread-local
/// backup buffers and restored in a finally block to ensure exception safety.
///
/// Thread safety: backup buffers are ThreadStatic, so concurrent plan execution
/// on different threads is safe. Same-thread reentrancy is handled by the
/// PersistentParallelExecutor's reentrancy guard.
/// </summary>
internal sealed class MixedPrecisionPass : ICpuOptimizationPass
{
    public string Name => "MixedPrecision";

    public bool IsEnabled => TensorCodecOptions.Current.EnableMixedPrecision;

#if NET7_0_OR_GREATER
    // Thread-local backup buffers to avoid per-call allocation and ensure thread safety
    [ThreadStatic] private static float[][]? _backupBuffers;
#endif

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float)) return null;

#if NET7_0_OR_GREATER
        int fp16Count = 0;
        var classifications = new PrecisionClass[steps.Length];
        for (int i = 0; i < steps.Length; i++)
        {
            classifications[i] = ClassifyOp(steps[i].OpName);
            if (classifications[i] == PrecisionClass.FP16Safe) fp16Count++;
        }

        if (fp16Count < steps.Length * 0.3) return null;

        var result = new CompiledStep<T>[steps.Length];
        bool anyOptimized = false;

        for (int i = 0; i < steps.Length; i++)
        {
            if (classifications[i] != PrecisionClass.FP16Safe)
            {
                result[i] = steps[i];
                continue;
            }

            var capturedStep = steps[i];

            bool allContiguous = true;
            for (int j = 0; j < capturedStep.Inputs.Length; j++)
            {
                if (!capturedStep.Inputs[j].IsContiguous)
                {
                    allContiguous = false;
                    break;
                }
            }

            if (!allContiguous)
            {
                result[i] = steps[i];
                continue;
            }

            result[i] = new CompiledStep<T>(
                steps[i].OpName,
                (eng, output) =>
                {
                    // Per-input backup arrays (ThreadStatic for thread safety)
                    var backups = _backupBuffers;
                    int inputCount = capturedStep.Inputs.Length;
                    if (backups is null || backups.Length < inputCount)
                        _backupBuffers = backups = new float[inputCount][];

                    // Save originals and quantize in-place
                    for (int j = 0; j < inputCount; j++)
                    {
                        if (capturedStep.Inputs[j] is Tensor<float> floatInp)
                        {
                            var src = floatInp.Data.Span;
                            int len = src.Length;
                            if (backups[j] is null || backups[j].Length < len)
                                backups[j] = new float[len];
                            src.CopyTo(backups[j].AsSpan(0, len));
                            for (int k = 0; k < len; k++)
                                src[k] = (float)(Half)src[k];
                        }
                    }

                    try
                    {
                        capturedStep.Execute(eng, output);
                    }
                    finally
                    {
                        // Always restore original fp32 values
                        for (int j = 0; j < inputCount; j++)
                        {
                            if (capturedStep.Inputs[j] is Tensor<float> floatInp
                                && backups[j] is not null)
                            {
                                var src = floatInp.Data.Span;
                                int len = src.Length;
                                backups[j].AsSpan(0, len).CopyTo(src);
                            }
                        }
                    }
                },
                steps[i].OutputBuffer,
                steps[i].Inputs,
                steps[i].BackwardFn,
                steps[i].SavedState);
            anyOptimized = true;
        }

        return anyOptimized ? result : null;
#else
        return null;
#endif
    }

    internal static PrecisionClass ClassifyOp(string opName)
    {
        return opName switch
        {
            "TensorMatMul" or "BatchMatMul" or "Conv2D" or "DepthwiseConv2D"
                or "ConvTranspose2D" or "Conv3D" => PrecisionClass.FP16Safe,

            "TensorAdd" or "TensorSubtract" or "TensorMultiply" or "TensorDivide"
                or "ReLU" or "GELU" or "Swish" or "Mish" or "LeakyReLU" or "ELU"
                or "Sigmoid" or "Tanh" or "HardSwish" => PrecisionClass.FP16Safe,

            "ReduceSum" or "ReduceMean" or "ReduceMax" or "Softmax" or "LogSoftmax"
                or "BatchNorm" or "LayerNorm" or "GroupNorm" or "InstanceNorm"
                => PrecisionClass.FP32Required,

            "TensorCrossEntropyLoss" or "TensorBinaryCrossEntropy" or "TensorMSELoss"
                or "TensorL1Loss" or "TensorHuberLoss" => PrecisionClass.FP32Required,

            _ => PrecisionClass.FP32Required
        };
    }
}

internal enum PrecisionClass
{
    FP16Safe,
    FP32Required,
    CastBoundary
}
