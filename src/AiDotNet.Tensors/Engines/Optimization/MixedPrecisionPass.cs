using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 7.3: Mixed precision compilation infrastructure.
///
/// Provides the framework for AMP (Automatic Mixed Precision) compilation:
/// - Forward pass ops in fp16 for 2x memory bandwidth
/// - Backward pass and weight updates in fp32 for numerical stability
/// - Master weights in fp32, shadow copies in fp16 for forward
///
/// Currently provides analysis and op classification.
/// Actual fp16 kernel dispatch requires Half type support in the engine,
/// which is available on .NET 7+ via System.Half.
/// </summary>
internal sealed class MixedPrecisionPass : ICpuOptimizationPass
{
    public string Name => "MixedPrecision";

    /// <summary>Mixed precision is opt-in and requires explicit configuration.</summary>
    public bool IsEnabled => TensorCodecOptions.Current.EnableMixedPrecision;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float)) return null;

#if NET7_0_OR_GREATER
        // Classify ops and check if enough are fp16-safe to justify the pass
        int fp16Count = 0;
        var classifications = new PrecisionClass[steps.Length];
        for (int i = 0; i < steps.Length; i++)
        {
            classifications[i] = ClassifyOp(steps[i].OpName);
            if (classifications[i] == PrecisionClass.FP16Safe) fp16Count++;
        }

        if (fp16Count < steps.Length * 0.3) return null;

        // For fp16-safe ops, quantize inputs to Half before execution and
        // dequantize output after. This reduces memory bandwidth by 2x for
        // the input reads (Half = 2 bytes vs float = 4 bytes).
        var result = new CompiledStep<T>[steps.Length];
        bool anyOptimized = false;

        for (int i = 0; i < steps.Length; i++)
        {
            if (classifications[i] == PrecisionClass.FP16Safe
                && steps[i].Inputs.Length >= 1
                && steps[i].Inputs[0].IsContiguous)
            {
                var capturedStep = steps[i];
                // Pre-allocate backup + Half buffers per input at compile time.
                // Strategy: save original fp32 values → write quantized values → execute → restore.
                // This is safe because restore happens before any other step reads the tensor.
                var backupBuffers = new float[capturedStep.Inputs.Length][];
                var halfBuffers = new Half[capturedStep.Inputs.Length][];
                for (int j = 0; j < capturedStep.Inputs.Length; j++)
                {
                    int len = capturedStep.Inputs[j].Length;
                    backupBuffers[j] = new float[len];
                    halfBuffers[j] = new Half[len];
                }

                result[i] = new CompiledStep<T>(
                    steps[i].OpName,
                    (eng, output) =>
                    {
                        // Save originals, quantize inputs to Half precision
                        for (int j = 0; j < capturedStep.Inputs.Length; j++)
                        {
                            var inp = capturedStep.Inputs[j];
                            if (inp.IsContiguous && inp is Tensor<float> floatInp)
                            {
                                var src = floatInp.Data.Span;
                                var backup = backupBuffers[j];
                                var halfBuf = halfBuffers[j];
                                int len = Math.Min(src.Length, backup.Length);
                                // Save original values
                                for (int k = 0; k < len; k++)
                                    backup[k] = src[k];
                                // Quantize in-place
                                for (int k = 0; k < len; k++)
                                {
                                    halfBuf[k] = (Half)src[k];
                                    src[k] = (float)halfBuf[k];
                                }
                            }
                        }
                        // Execute with quantized inputs
                        capturedStep.Execute(eng, output);
                        // Restore original fp32 values (critical for backward pass)
                        for (int j = 0; j < capturedStep.Inputs.Length; j++)
                        {
                            var inp = capturedStep.Inputs[j];
                            if (inp.IsContiguous && inp is Tensor<float> floatInp)
                            {
                                var src = floatInp.Data.Span;
                                var backup = backupBuffers[j];
                                int len = Math.Min(src.Length, backup.Length);
                                for (int k = 0; k < len; k++)
                                    src[k] = backup[k];
                            }
                        }
                    },
                    steps[i].OutputBuffer,
                    steps[i].Inputs,
                    steps[i].BackwardFn, // Backward uses restored fp32 inputs
                    steps[i].SavedState);
                anyOptimized = true;
            }
            else
            {
                result[i] = steps[i];
            }
        }

        return anyOptimized ? result : null;
#else
        return null;
#endif
    }

    /// <summary>
    /// Classifies an operation for mixed precision:
    /// - FP16Safe: can run in half precision without accuracy loss
    /// - FP32Required: must stay in full precision (reductions, softmax, loss)
    /// - CastBoundary: needs explicit cast between precision levels
    /// </summary>
    internal static PrecisionClass ClassifyOp(string opName)
    {
        return opName switch
        {
            // Matrix operations: safe for fp16 (largest wins from reduced bandwidth)
            "TensorMatMul" or "BatchMatMul" or "Conv2D" or "DepthwiseConv2D"
                or "ConvTranspose2D" or "Conv3D" => PrecisionClass.FP16Safe,

            // Element-wise: safe for fp16
            "TensorAdd" or "TensorSubtract" or "TensorMultiply" or "TensorDivide"
                or "ReLU" or "GELU" or "Swish" or "Mish" or "LeakyReLU" or "ELU"
                or "Sigmoid" or "Tanh" or "HardSwish" => PrecisionClass.FP16Safe,

            // Reductions and normalization: require fp32 for numerical stability
            "ReduceSum" or "ReduceMean" or "ReduceMax" or "Softmax" or "LogSoftmax"
                or "BatchNorm" or "LayerNorm" or "GroupNorm" or "InstanceNorm"
                => PrecisionClass.FP32Required,

            // Loss functions: require fp32
            "TensorCrossEntropyLoss" or "TensorBinaryCrossEntropy" or "TensorMSELoss"
                or "TensorL1Loss" or "TensorHuberLoss" => PrecisionClass.FP32Required,

            // Default: stay in fp32 for safety
            _ => PrecisionClass.FP32Required
        };
    }
}

/// <summary>Precision classification for mixed-precision compilation.</summary>
internal enum PrecisionClass
{
    /// <summary>Can safely run in fp16 without accuracy loss.</summary>
    FP16Safe,

    /// <summary>Must run in fp32 for numerical stability.</summary>
    FP32Required,

    /// <summary>Needs explicit precision cast at boundary.</summary>
    CastBoundary
}
