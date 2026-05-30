using AiDotNet.Tensors.Engines.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>Per-variant bench shapes for the #379 specialized BLAS variants.</summary>
public static class SpecializedShapeCatalog
{
    public record TrsmShape(string Name, Side Side, Uplo Uplo, bool TransA, Diag Diag,
        int M, int N, bool Fp64, int Frequency, string Source);

    public static readonly TrsmShape[] Trsm =
    {
        new("Chol_Solve_64x1",   Side.Left, Uplo.Lower, false, Diag.NonUnit,  64,   1, true,  50, "workload:cholesky-solve"),
        new("Chol_Solve_256x1",  Side.Left, Uplo.Lower, false, Diag.NonUnit, 256,   1, true,  40, "workload:cholesky-solve"),
        new("QR_BackSub_256x64", Side.Left, Uplo.Upper, false, Diag.NonUnit, 256,  64, true,  30, "workload:qr-backsub"),
        new("Solve_MultiRhs_512x128", Side.Left, Uplo.Lower, false, Diag.NonUnit, 512, 128, true, 20, "workload:linsolve"),
        new("Chol_SolveT_256x1", Side.Left, Uplo.Lower, true,  Diag.NonUnit, 256,   1, true,  25, "workload:cholesky-solve-transpose"),
        new("Solve_FP32_512x64", Side.Left, Uplo.Lower, false, Diag.NonUnit, 512,  64, false, 15, "workload:linsolve-fp32"),
    };

    public record SyrkShape(string Name, Uplo Uplo, bool Trans, int N, int K, bool Fp64, int Frequency, string Source);

    public static readonly SyrkShape[] Syrk =
    {
        new("Cov_256x64",  Uplo.Lower, true,  256,  64, true, 30, "workload:covariance"),
        new("Cov_512x128", Uplo.Lower, true,  512, 128, true, 25, "workload:covariance"),
        new("Gram_128x768",Uplo.Lower, false, 128, 768, true, 20, "workload:gram-matrix"),
        new("Cov_FP32_256x64", Uplo.Lower, true, 256, 64, false, 15, "workload:covariance-fp32"),
    };

    public record SymmShape(string Name, Side Side, Uplo Uplo, int M, int N, bool Fp64, int Frequency, string Source);

    public static readonly SymmShape[] Symm =
    {
        new("Sym_Left_256x256",  Side.Left, Uplo.Lower, 256, 256, true, 20, "workload:optimization"),
        new("Sym_Left_512x128",  Side.Left, Uplo.Lower, 512, 128, true, 18, "workload:optimization"),
        new("Sym_Right_128x512", Side.Right, Uplo.Upper, 128, 512, true, 12, "workload:optimization"),
        new("Sym_FP32_256x256",  Side.Left, Uplo.Lower, 256, 256, false, 10, "workload:optimization-fp32"),
    };

    // SpMM: sparse A (rows×cols, given density) × dense B (cols×N). DensityPercent
    // is nnz / (rows·cols) × 100.
    public record SpMMShape(string Name, bool Csr, int Rows, int Cols, int N, double DensityPercent,
        bool Fp64, int Frequency, string Source);

    public static readonly SpMMShape[] SpMM =
    {
        new("GNN_Cora_2708x2708x64",   true, 2708, 2708,  64, 0.18, true, 30, "workload:gnn-cora"),
        new("GNN_PubMed_19717x500x128",true, 19717, 500, 128, 0.10, true, 20, "workload:gnn-pubmed"),
        new("RecSys_10000x10000x32",   true, 10000,10000, 32, 0.05, true, 15, "workload:recsys"),
        new("Dense_ish_512x512x64",    true, 512,  512,  64, 5.0,  true, 10, "workload:spmm-stress"),
        new("GNN_FP32_2708x2708x64",   true, 2708, 2708,  64, 0.18, false,12, "workload:gnn-cora-fp32"),
    };

    public record GbmvShape(string Name, bool Trans, int M, int N, int Kl, int Ku, bool Fp64, int Frequency, string Source);

    public static readonly GbmvShape[] Gbmv =
    {
        new("Tridiag_1024",   false, 1024, 1024, 1, 1, true, 20, "workload:tridiagonal-solver"),
        new("Pentadiag_2048", false, 2048, 2048, 2, 2, true, 15, "workload:pentadiagonal"),
        new("Tridiag_T_1024", true,  1024, 1024, 1, 1, true, 10, "workload:tridiagonal-transpose"),
        new("Banded_FP32_4096",false,4096, 4096, 3, 3, false, 8, "workload:banded-fp32"),
    };
}
