using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// TensorCodec: compile-once, replay-forever execution plans with
/// automatic graph recording, fusion, and optimization.
///
/// Usage for inference:
///   var plan = TensorCodec.CompileInference(() => {
///       var h = engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
///       return engine.FusedLinear(h, w2, b2, FusedActivationType.None);
///   });
///   var output = plan.Execute();  // zero-alloc replay
///
/// Usage for training:
///   var plan = TensorCodec.CompileTraining(() => {
///       var h = engine.FusedLinear(input, w1, b1, FusedActivationType.ReLU);
///       var output = engine.FusedLinear(h, w2, b2, FusedActivationType.None);
///       var loss = engine.ReduceSum(engine.TensorMultiply(
///           engine.TensorSubtract(output, target),
///           engine.TensorSubtract(output, target)), null);
///       return loss;
///   }, parameters: new[] { w1, w2 });
///   plan.Step();  // forward + backward, zero-alloc replay
///
/// Applies all TensorCodec optimizations automatically:
/// - Phase A: Spectral decomposition (inference only)
/// - Phase B: Dataflow fusion (multi-layer GEMM)
/// - Phase C: Algebraic backward simplification
/// - Specialized BLAS/SIMD delegates for all ops
/// </summary>
internal static class TensorCodec
{
    /// <summary>
    /// Compiles an inference computation into an optimized plan.
    /// The computation lambda is traced once via GraphMode, then compiled
    /// with all TensorCodec optimizations enabled.
    /// </summary>
    /// <param name="computation">The computation to compile. Called once for tracing.</param>
    /// <param name="options">Optional TensorCodec options. Defaults to all optimizations enabled.</param>
    /// <returns>A compiled inference plan that can be replayed with zero allocation.</returns>
    public static CompiledInferencePlan<T> CompileInference<T>(
        Action computation,
        TensorCodecOptions? options = null)
    {
        var opts = options ?? new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableAlgebraicBackward = true,
            EnableSpectralDecomposition = true
        };

        var prevOpts = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(opts);
        try
        {
            using var scope = GraphMode.Enable();
            computation();
            return scope.CompileInference<T>();
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prevOpts == TensorCodecOptions.Default ? null : prevOpts);
        }
    }

    /// <summary>
    /// Compiles a training computation into an optimized plan with automatic backward.
    /// The computation lambda is traced once via GraphMode, then compiled
    /// with all TensorCodec optimizations enabled. The plan includes forward + backward.
    /// </summary>
    /// <param name="computation">The computation to compile. Called once for tracing.</param>
    /// <param name="parameters">Parameters to compute gradients for.</param>
    /// <param name="options">Optional TensorCodec options.</param>
    /// <returns>A compiled training plan that can be replayed with near-zero allocation.</returns>
    public static CompiledTrainingPlan<T> CompileTraining<T>(
        Action computation,
        Tensor<T>[] parameters,
        TensorCodecOptions? options = null)
    {
        var opts = options ?? new TensorCodecOptions
        {
            EnableDataflowFusion = true,
            EnableAlgebraicBackward = true,
            EnableSpectralDecomposition = false // training needs exact gradients
        };

        var prevOpts = TensorCodecOptions.Current;
        TensorCodecOptions.SetCurrent(opts);
        try
        {
            using var scope = GraphMode.Enable();
            computation();
            return scope.CompileTraining(parameters);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(prevOpts == TensorCodecOptions.Default ? null : prevOpts);
        }
    }
}
