using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Mixed precision compilation pass — currently disabled.
///
/// The in-place fp32→Half→fp32 round-trip approach does not actually reduce memory
/// bandwidth (data remains fp32 throughout) and corrupts shared input tensors.
/// A proper implementation would use separate Half-precision intermediate buffers
/// with Half-precision SIMD kernels. Kept as opt-in placeholder for future work.
/// </summary>
internal sealed class MixedPrecisionPass : ICpuOptimizationPass
{
    public string Name => "MixedPrecision";

    // Disabled: current implementation doesn't reduce bandwidth and corrupts shared data.
    // See class doc for details. Enable only after implementing real Half-precision buffers.
    public bool IsEnabled => false;

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

            // Only quantize inputs that are contiguous
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

            // Lazy-allocate backup buffers on first execution (avoids upfront allocation
            // for steps that may never run due to early exit or pruning)
            float[][]? backupBuffers = null;

            result[i] = new CompiledStep<T>(
                steps[i].OpName,
                (eng, output) =>
                {
                    // Lazy-init backup buffers on first call
                    if (backupBuffers is null)
                    {
                        backupBuffers = new float[capturedStep.Inputs.Length][];
                        for (int j = 0; j < capturedStep.Inputs.Length; j++)
                            backupBuffers[j] = new float[capturedStep.Inputs[j].Length];
                    }

                    // Quantize inputs to Half precision (save originals first)
                    for (int j = 0; j < capturedStep.Inputs.Length; j++)
                    {
                        if (capturedStep.Inputs[j] is Tensor<float> floatInp)
                        {
                            var src = floatInp.Data.Span;
                            var backup = backupBuffers[j];
                            int len = Math.Min(src.Length, backup.Length);
                            src.Slice(0, len).CopyTo(backup);
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
                        // Always restore fp32 values, even if Execute throws
                        for (int j = 0; j < capturedStep.Inputs.Length; j++)
                        {
                            if (capturedStep.Inputs[j] is Tensor<float> floatInp)
                            {
                                var src = floatInp.Data.Span;
                                var backup = backupBuffers[j];
                                int len = Math.Min(src.Length, backup.Length);
                                backup.AsSpan(0, len).CopyTo(src);
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
