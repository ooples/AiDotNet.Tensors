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
        if (!IsEnabled) return null;
        // Mixed precision only makes sense for float (downcast to Half for forward)
        if (typeof(T) != typeof(float)) return null;

#if NET7_0_OR_GREATER
        // Classify each op and find fp16-safe chains
        int fp16Count = 0;
        var classifications = new PrecisionClass[steps.Length];
        for (int i = 0; i < steps.Length; i++)
        {
            classifications[i] = ClassifyOp(steps[i].OpName);
            if (classifications[i] == PrecisionClass.FP16Safe) fp16Count++;
        }

        // Only apply if at least 30% of ops can run in fp16
        if (fp16Count < steps.Length * 0.3) return null;

        // For fp16-safe ops, wrap the execute delegate with fp32→fp16→execute→fp16→fp32 casts.
        // This simulates mixed precision by casting inputs to Half before computation
        // and casting outputs back to float. True fp16 kernel dispatch would be faster
        // but requires Half-typed tensors throughout.
        var result = new CompiledStep<T>[steps.Length];
        bool anyOptimized = false;

        for (int i = 0; i < steps.Length; i++)
        {
            if (classifications[i] == PrecisionClass.FP16Safe && steps[i].Inputs.Length >= 1)
            {
                // Wrap with precision cast: forward executes in reduced precision
                // by rounding inputs to Half precision first (simulates fp16 bandwidth)
                // Pre-allocate Half-rounded shadow buffers for each input at compile time.
                // The execute delegate copies fp32 inputs → shadow (rounded to Half precision)
                // → executes the op on the shadow inputs. Original inputs are NOT modified,
                // preserving backward pass correctness and shared tensor integrity.
                var capturedStep = steps[i];
                var shadowInputs = new Tensor<T>[steps[i].Inputs.Length];
                for (int si = 0; si < shadowInputs.Length; si++)
                    shadowInputs[si] = Helpers.TensorAllocator.RentUninitialized<T>(steps[i].Inputs[si]._shape);
                var capturedShadows = shadowInputs;

                result[i] = new CompiledStep<T>(
                    "MP_" + steps[i].OpName,
                    (eng, output) =>
                    {
                        // Copy inputs to shadow buffers with Half rounding (simulates fp16 bandwidth)
                        for (int si = 0; si < capturedStep.Inputs.Length; si++)
                        {
                            var inp = capturedStep.Inputs[si];
                            if (inp is Tensor<float> floatInp && capturedShadows[si] is Tensor<float> floatShadow
                                && floatInp.IsContiguous)
                            {
                                var src = floatInp.Data.Span;
                                var dst = floatShadow.Data.Span;
                                for (int j = 0; j < src.Length && j < dst.Length; j++)
                                    dst[j] = (float)(Half)src[j]; // fp32 → fp16 → fp32 round-trip
                            }
                        }
                        capturedStep.Execute(eng, output);
                    },
                    steps[i].OutputBuffer,
                    capturedShadows, // Use shadow inputs for this step
                    steps[i].BackwardFn, // Backward stays in fp32 on original inputs
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
        // System.Half not available before .NET 7 — no mixed precision support
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
