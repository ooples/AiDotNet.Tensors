namespace AiDotNet.Tensors.Helpers
{
    /// <summary>
    /// Single process-wide gate serializing ALL native compute-library execution —
    /// OpenBLAS GEMM (<see cref="BlasProvider"/>) and oneDNN primitive execute
    /// (<see cref="OneDnnProvider"/>).
    /// </summary>
    /// <remarks>
    /// OpenBLAS and oneDNN each ship their own multi-threaded runtime (OpenMP / an
    /// internal thread pool). Executing a GEMM and a oneDNN primitive AT THE SAME TIME
    /// makes the two libraries contend over that shared threading runtime and its
    /// thread-local state, which intermittently corrupts it and faults the process with
    /// a native access violation (0xC0000005). A crash dump of the full xUnit suite
    /// caught exactly this: one thread inside <c>cblas_sgemm</c> while another was inside
    /// <c>dnnl_primitive_execute</c>. The two providers historically used separate locks
    /// (<c>_nativeGemmGate</c> and <c>_executeLock</c>), so they could overlap.
    ///
    /// Both providers now take THIS lock around their native execute, so the two native
    /// math libraries never run compute concurrently. In the common single-threaded
    /// training/inference loop, ops already run sequentially, so the lock is uncontended
    /// and effectively free; it only serializes genuinely concurrent native compute
    /// (independent tapes / parallel tests), which is exactly the unsafe scenario.
    /// </remarks>
    internal static class NativeComputeGate
    {
        /// <summary>The shared lock object. Acquire with <c>lock</c> / <c>Monitor</c>.</summary>
        internal static readonly object Instance = new object();
    }
}
