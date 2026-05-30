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
}
