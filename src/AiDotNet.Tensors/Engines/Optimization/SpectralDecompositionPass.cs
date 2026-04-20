using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Fft;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase A: Spectral decomposition pass that analyzes weight matrices in MatMul
/// operations and replaces them with low-rank approximations via truncated SVD.
///
/// For W[M,N] with fast singular value decay (effective rank r ≪ min(M,N)):
///   y = x @ W  →  y = (x @ leftFactor[M,r]) @ rightFactor[r,N]
///
/// This reduces FLOP count from O(M*N) to O(M*r + r*N) per sample.
/// The approximation error is bounded by SpectralErrorTolerance.
///
/// Only applied to inference plans (frozen weights). Training plans need
/// exact gradients through the original weight matrix.
/// </summary>
internal sealed class SpectralDecompositionPass : ICpuOptimizationPass
{
    public string Name => "SpectralDecomposition";

    public bool IsEnabled => TensorCodecOptions.Current.EnableSpectralDecomposition;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float)) return null;

        var result = new CompiledStep<T>[steps.Length];
        bool anyOptimized = false;

        bool fftConvEnabled = TensorCodecOptions.Current.EnableFftConv;
        int fftKernelThreshold = TensorCodecOptions.Current.FftConvKernelThreshold;

        for (int i = 0; i < steps.Length; i++)
        {
            if (steps[i].OpType == OpType.TensorMatMul
                && steps[i].Inputs.Length == 2
                && steps[i].Inputs[0].Rank == 2
                && steps[i].Inputs[1].Rank == 2)
            {
                var optimized = TryDecomposeWeight(steps[i]);
                if (optimized != null)
                {
                    result[i] = optimized;
                    anyOptimized = true;
                    continue;
                }
            }
            if (fftConvEnabled
                && steps[i].OpType == OpType.Conv2D
                && steps[i].Inputs.Length >= 2
                && steps[i].Inputs[0].Rank == 4
                && steps[i].Inputs[1].Rank == 4)
            {
                var optimized = TryRewriteConv2DToFft(steps[i], fftKernelThreshold);
                if (optimized != null)
                {
                    result[i] = optimized;
                    anyOptimized = true;
                    continue;
                }
            }
            result[i] = steps[i];
        }

        return anyOptimized ? result : null;
    }

    /// <summary>
    /// Replace a Conv2D step with a FFT-based equivalent when the kernel is
    /// large enough that FFT convolution wins. Preconditions the pass assumes:
    ///   * stride = 1, dilation = 1, groups = 1, padding = 'same'.
    /// These hold for the typical large-kernel Conv2D call sites (diffusion
    /// models, UNet encoder blocks) but the pass cannot verify them from the
    /// delegate alone. Callers opt in via <see cref="TensorCodecOptions.EnableFftConv"/>
    /// to acknowledge this contract; other configurations must stay on the
    /// direct kernel or the rewrite silently produces wrong outputs.
    /// </summary>
    private static CompiledStep<T>? TryRewriteConv2DToFft<T>(CompiledStep<T> step, int kernelThreshold)
    {
        if (typeof(T) != typeof(float)) return null;

        var weight = step.Inputs[1];
        if (weight.Rank != 4) return null;
        int Kh = weight._shape[2];
        int Kw = weight._shape[3];
        if (Kh < kernelThreshold && Kw < kernelThreshold) return null;

        var capturedInput = step.Inputs[0];
        var capturedWeight = weight;
        // Bias is sometimes a third input (Conv2D with bias), sometimes baked
        // into a separate add step; we forward it only when present as input 2.
        Tensor<T>? capturedBias = step.Inputs.Length >= 3 ? step.Inputs[2] : null;

        return new CompiledStep<T>(
            "FftConv2D",
            (eng, output) =>
            {
                var fftInput = (Tensor<float>)(object)capturedInput;
                var fftWeight = (Tensor<float>)(object)capturedWeight;
                var fftBias = capturedBias is null ? null : (Tensor<float>)(object)capturedBias;
                var fftOut = FftConv.Conv2DSame(fftInput, fftWeight, fftBias);
                var fftData = fftOut.GetDataArray();
                var outArr = (float[])(object)output.GetDataArray();
                System.Array.Copy(fftData, outArr, fftData.Length);
            },
            step.OutputBuffer,
            step.Inputs,
            step.BackwardFn,  // Keep original backward — FFT-conv is forward-only optim.
            step.SavedState);
    }

    private static CompiledStep<T>? TryDecomposeWeight<T>(CompiledStep<T> step)
    {
        var weight = step.Inputs[1]; // W[K,N]
        if (weight.Rank != 2) return null;

        int m = weight._shape[0];
        int n = weight._shape[1];

        // Only worthwhile for matrices large enough
        if (m < 32 || n < 32) return null;

        var weightData = (float[])(object)weight.GetDataArray();
        var factors = TensorCodecOptimizer.TrySpectralDecompose(weightData, m, n);

        if (!factors.HasValue) return null;

        var input = step.Inputs[0];
        var capturedInput = input;
        var capturedFactors = factors.Value;

        return new CompiledStep<T>(
            "SpectralMatMul",
            (eng, output) =>
            {
                var inArr = (float[])(object)capturedInput.GetDataArray();
                var outArr = (float[])(object)output.GetDataArray();
                int rows = capturedInput._shape.Length >= 2 ? capturedInput._shape[0] : 1;
                int cols = capturedInput._shape.Length >= 2 ? capturedInput._shape[^1] : capturedInput._shape[0];

                // Workspace allocated per call — avoids mutable captured state that
                // would race if the same plan is executed on multiple threads.
                // SpectralMatMul reuses the workspace internally across the two GEMMs.
                int needed = rows * capturedFactors.Rank;
                var workspace = new float[needed];

                SvdDecomposition.SpectralMatMul(inArr, rows, cols, capturedFactors, outArr, workspace);
            },
            step.OutputBuffer,
            step.Inputs,
            step.BackwardFn,  // Keep original backward (spectral is inference-only optimization)
            step.SavedState);
    }
}
