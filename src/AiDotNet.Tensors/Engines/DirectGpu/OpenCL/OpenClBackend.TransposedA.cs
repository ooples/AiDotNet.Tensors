using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Row-major C[M,N] = alpha · A[K,M]ᵀ · B[K,N] + beta · C: the OpenCL port of CudaBackend.MatMulTransposedA.
    public sealed partial class OpenClBackend : ITransposedAGemm
    {
        public void MatMulTransposedA(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        {
            if (_context == null)
                throw new InvalidOperationException("OpenCL context not available");
            TransposedAGemmArgs.Validate(A, B, C, M, N, K);
            var bufferA = ((DirectOpenClGpuBuffer)A).Buffer;
            var bufferB = ((DirectOpenClGpuBuffer)B).Buffer;
            var bufferC = ((DirectOpenClGpuBuffer)C).Buffer;
            if (ClBlastNative.IsAvailable)
            {
                IntPtr queue = _context.CommandQueue;
                var memories = DirectOpenClSubmission.GetDirectSubmissionMemories(bufferA, bufferB, bufferC);
                ClBlastNative.StatusCode status;
                try
                {
                    lock (DirectOpenClSubmission.Gate)
                    {
                        using var waits = DirectOpenClSubmission.PrepareLocked(queue, memories);
                        waits.EnqueueBridgeMarker(queue);
                        // CLBlast row-major with transA = Yes: A is stored [K, M], so its row stride is M.
                        status = ClBlastNative.Sgemm(
                            ClBlastNative.Layout.RowMajor,
                            ClBlastNative.Transpose.Yes,
                            ClBlastNative.Transpose.No,
                            (UIntPtr)M, (UIntPtr)N, (UIntPtr)K,
                            alpha,
                            bufferA.Handle, UIntPtr.Zero, (UIntPtr)M,
                            bufferB.Handle, UIntPtr.Zero, (UIntPtr)N,
                            beta,
                            bufferC.Handle, UIntPtr.Zero, (UIntPtr)N,
                            ref queue,
                            IntPtr.Zero);
                        if (status == ClBlastNative.StatusCode.Success)
                            DirectOpenClSubmission.CommitLocked(queue, memories);
                    }
                }
                finally
                {
                    DirectOpenClSubmission.ReleaseDirectSubmissionMemories(memories);
                }
                if (status == ClBlastNative.StatusCode.Success)
                {
                    // CLBlast enqueues on the native queue, bypassing DirectOpenClKernel's dispatch hook.
                    GpuLaunchProbe.OnLaunch();
                    return;
                }
            }
            // CLBlast is optional: materialize Aᵀ [M,K] with the transpose kernel and run the on-device GEMM.
            using var transposedA = AllocateBuffer(M * K);
            Transpose(A, transposedA, K, M);
            Gemm(transposedA, B, C, M, N, K, alpha, beta);
        }
    }
}