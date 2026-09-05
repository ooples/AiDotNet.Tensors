namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>Explicit per-call parallelism policy for internal SGEMM benchmarking.</summary>
internal enum SgemmExecutionMode
{
    /// <summary>Run the complete GEMM call without internal parallel dispatch.</summary>
    Sequential = 0,

    /// <summary>Allow the GEMM implementation to use its normal parallel thresholds.</summary>
    Parallel = 1
}
