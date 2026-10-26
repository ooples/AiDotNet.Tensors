namespace AiDotNet.Tensors.Engines.DirectGpu;

using AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// The conv-as-GEMM product of the FP16 im2col conv path for a batch of images, written straight in NCHW. The im2col
/// buffer is <c>col[K, batch * pixels]</c> (image <c>b</c>'s columns at <c>b * pixels</c>), so one GEMM over the whole
/// batch would produce <c>[M][batch][pixels]</c>; this runs one GEMM per image, <c>out[b] = W[M,K] @ col[:, b]</c>,
/// into <c>out[b * M * pixels]</c>. A backend without it keeps a batch above one off the FP16 im2col path.
/// </summary>
internal interface IFp16ConvBatchedGemmBackend
{
    /// <summary><c>output[b][M][pixels] = weights[M,K] @ col[K, b * pixels + p]</c> for every image, FP16 operands, FP32
    /// accumulation, FP16 (<paramref name="halfOutput"/>) or FP32 output.</summary>
    void GemmFp16ConvBatched(IGpuBuffer weightsFp16, IGpuBuffer colFp16, IGpuBuffer output, int m, int pixels, int k,
        int batch, bool halfOutput);
}
