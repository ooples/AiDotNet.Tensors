namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the OpenCL port of CudaBackend.MatMulTransposedA.
    public sealed partial class OpenClBackend : ITransposedAGemm
    {
        public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new GpuCapabilityUnavailableException("OpenCL context not available");
            TransposedAGemmArgs.Validate(A, B, C, M, N, K);
            // CLBlast in place below its packed-kernel threshold, row stride M for A stored [K, M]; past it the library
            // copies and pads the operands and the output into buffers of its own (see MatMulTransposed: VGG-11's
            // classifier weight gradient, 4096 x 4096 x 64, held a 64 MB copy of the gradient at every step). There Aᵀ
            // comes from the pool, the size of A, an activation that is live anyway.
            if (UsesClBlastDirectKernel(M, N, K)
                && TryExecuteClBlastLibraryGemm(ClBlastNative.Transpose.Yes, ClBlastNative.Transpose.No, A, M, B, N, C, M, N, K, alpha, beta))
            {
                // CLBlast enqueues on the native queue, bypassing DirectOpenClKernel's dispatch hook.
                GpuLaunchProbe.OnLaunch();
                RecordGemmDispatch(OpenClGemmDispatchPath.ClBlastLibrary);
                return;
            }
            using var transposedA = AllocateBuffer(M * K);
            Transpose(A, transposedA, K, M);
            Gemm(transposedA, B, C, M, N, K, alpha, beta);
        }
    }
}
