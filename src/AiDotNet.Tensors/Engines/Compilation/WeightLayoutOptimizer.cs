using System.Buffers;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Weight layout optimization: packs weight matrices into cache-friendly formats
/// at compile time to avoid per-call packing overhead during execution.
///
/// Standard layout: row-major float[K, N] — BLAS-friendly but SIMD tile kernels
/// need panel format (blocks of Nr columns packed contiguously).
///
/// Pre-packing at compile time means:
/// - Zero per-call packing overhead (FusedMultiLayerGemm was packing W2 per call)
/// - Better cache utilization (packed format matches the SIMD tile access pattern)
/// - Amortized over thousands of training steps
/// </summary>
internal static class WeightLayoutOptimizer
{
    /// <summary>
    /// Packs a row-major weight matrix into panel format for SIMD GEMM.
    /// Panel format: Nr columns are stored contiguously for each row block.
    /// </summary>
    /// <param name="weights">Row-major weight matrix [rows, cols].</param>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="panelWidth">Panel width (Nr), typically 8 for AVX or 16 for AVX-512.</param>
    /// <returns>Pre-packed weight array in panel format.</returns>
    internal static float[] PackRowMajorToPanelFormat(float[] weights, int rows, int cols, int panelWidth = 8)
    {
        int numPanels = (cols + panelWidth - 1) / panelWidth;
        var packed = new float[rows * numPanels * panelWidth];

        for (int r = 0; r < rows; r++)
        {
            for (int panel = 0; panel < numPanels; panel++)
            {
                int packedOffset = (panel * rows + r) * panelWidth;
                int srcCol = panel * panelWidth;
                for (int j = 0; j < panelWidth && srcCol + j < cols; j++)
                    packed[packedOffset + j] = weights[r * cols + srcCol + j];
            }
        }

        return packed;
    }

}
