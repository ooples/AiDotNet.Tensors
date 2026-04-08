using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Truncated SVD decomposition for weight matrix compression.
/// Decomposes W[M,N] = U[M,r] @ diag(σ[r]) @ V^T[r,N] where r &lt; min(M,N).
///
/// For inference: replaces one [M,N] matmul with two smaller matmuls:
///   y = x @ W  →  y = (x @ V_r) @ (Σ_r @ U_r^T)
/// where V_r is [N,r] and Σ_r@U_r^T is [r,M].
///
/// Uses one-sided Jacobi SVD for numerical stability on small-to-medium matrices
/// typical in neural network layers (up to ~1024 dimensions).
/// </summary>
internal static class SvdDecomposition
{
    /// <summary>
    /// Computes truncated SVD: W ≈ U_r @ diag(σ_r) @ V_r^T
    /// Returns the factored form ready for fast inference.
    /// </summary>
    /// <param name="matrix">Weight matrix [M,N] in row-major</param>
    /// <param name="m">Rows</param>
    /// <param name="n">Columns</param>
    /// <param name="maxRank">Maximum rank to retain (0 = auto-select by energy)</param>
    /// <param name="energyThreshold">Fraction of total energy to retain (e.g., 0.999 = 99.9%)</param>
    /// <returns>Spectral factors or null if decomposition not beneficial</returns>
    internal static SpectralFactors? Decompose(
        float[] matrix, int m, int n,
        int maxRank = 0,
        double energyThreshold = 0.9999)
    {
        int minDim = Math.Min(m, n);
        if (minDim < 2) return null;

        // Compute full SVD via one-sided Jacobi
        var sigma = new float[minDim];
        var u = new float[m * minDim];
        var vt = new float[minDim * n];

        JacobiSVD(matrix, m, n, u, sigma, vt);

        // Determine rank by energy threshold
        double totalEnergy = 0;
        for (int i = 0; i < minDim; i++)
            totalEnergy += (double)sigma[i] * sigma[i];

        if (totalEnergy < 1e-12) return null;

        int rank = minDim;
        if (maxRank > 0) rank = Math.Min(rank, maxRank);

        double retainedEnergy = 0;
        for (int i = 0; i < rank; i++)
        {
            retainedEnergy += (double)sigma[i] * sigma[i];
            if (retainedEnergy / totalEnergy >= energyThreshold)
            {
                rank = i + 1;
                break;
            }
        }

        // Check if rank reduction is beneficial (at least 2x reduction)
        if (rank >= minDim / 2) return null; // Not enough compression

        // Build left factor: (U_r @ diag(σ_r))^T = diag(σ_r) @ U_r^T → [r, M]
        // Actually for y = x @ W = x @ U @ Σ @ V^T:
        //   y = (x @ U_r @ diag(σ_r)) @ V_r^T
        //   leftFactor = U_r @ diag(σ_r): [M, r]
        //   rightFactor = V_r^T: [r, N]
        var leftFactor = new float[m * rank];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < rank; j++)
                leftFactor[i * rank + j] = u[i * minDim + j] * sigma[j];

        var rightFactor = new float[rank * n];
        Array.Copy(vt, 0, rightFactor, 0, rank * n);

        // Compute approximation error
        double error = 0;
        for (int i = rank; i < minDim; i++)
            error += (double)sigma[i] * sigma[i];
        error = Math.Sqrt(error / totalEnergy);

        return new SpectralFactors
        {
            LeftFactor = leftFactor,
            RightFactor = rightFactor,
            Rank = rank,
            OriginalM = m,
            OriginalN = n,
            ApproximationError = error,
            SingularValues = sigma
        };
    }

    /// <summary>
    /// One-sided Jacobi SVD: numerically stable, good for small-medium matrices.
    /// Computes W = U @ diag(σ) @ V^T.
    /// </summary>
    private static void JacobiSVD(float[] matrix, int m, int n, float[] u, float[] sigma, float[] vt)
    {
        int minDim = Math.Min(m, n);

        // Work on a copy
        var work = new float[m * n];
        Array.Copy(matrix, work, m * n);

        // Initialize V = I
        var v = new float[n * n];
        for (int i = 0; i < n; i++) v[i * n + i] = 1f;

        // Jacobi iterations: rotate columns of work to diagonalize W^T @ W
        int maxIter = 100;
        for (int iter = 0; iter < maxIter; iter++)
        {
            double offDiag = 0;

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    // Compute 2x2 subproblem: columns p and q
                    double app = 0, aqq = 0, apq = 0;
                    for (int i = 0; i < m; i++)
                    {
                        double wp = work[i * n + p];
                        double wq = work[i * n + q];
                        app += wp * wp;
                        aqq += wq * wq;
                        apq += wp * wq;
                    }

                    offDiag += apq * apq;

                    if (Math.Abs(apq) < 1e-12 * Math.Sqrt(app * aqq))
                        continue;

                    // Compute Jacobi rotation angle
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = Math.Sign(tau) / (Math.Abs(tau) + Math.Sqrt(1.0 + tau * tau));
                    double cos = 1.0 / Math.Sqrt(1.0 + t * t);
                    double sin = t * cos;

                    // Apply rotation to work columns p, q
                    for (int i = 0; i < m; i++)
                    {
                        double wp = work[i * n + p];
                        double wq = work[i * n + q];
                        work[i * n + p] = (float)(cos * wp - sin * wq);
                        work[i * n + q] = (float)(sin * wp + cos * wq);
                    }

                    // Apply rotation to V columns p, q
                    for (int i = 0; i < n; i++)
                    {
                        double vp = v[i * n + p];
                        double vq = v[i * n + q];
                        v[i * n + p] = (float)(cos * vp - sin * vq);
                        v[i * n + q] = (float)(sin * vp + cos * vq);
                    }
                }
            }

            if (offDiag < 1e-20) break;
        }

        // Extract singular values and U
        for (int j = 0; j < minDim; j++)
        {
            double norm = 0;
            for (int i = 0; i < m; i++)
                norm += (double)work[i * n + j] * work[i * n + j];
            norm = Math.Sqrt(norm);

            sigma[j] = (float)norm;

            if (norm > 1e-10)
            {
                for (int i = 0; i < m; i++)
                    u[i * minDim + j] = (float)(work[i * n + j] / norm);
            }
        }

        // Sort by descending singular value
        for (int i = 0; i < minDim - 1; i++)
        {
            int maxIdx = i;
            for (int j = i + 1; j < minDim; j++)
                if (sigma[j] > sigma[maxIdx]) maxIdx = j;

            if (maxIdx != i)
            {
                // Swap sigma
                (sigma[i], sigma[maxIdx]) = (sigma[maxIdx], sigma[i]);

                // Swap U columns
                for (int row = 0; row < m; row++)
                    (u[row * minDim + i], u[row * minDim + maxIdx]) = (u[row * minDim + maxIdx], u[row * minDim + i]);

                // Swap V columns (rows of V^T)
                for (int row = 0; row < n; row++)
                    (v[row * n + i], v[row * n + maxIdx]) = (v[row * n + maxIdx], v[row * n + i]);
            }
        }

        // V^T = transpose of V
        for (int i = 0; i < minDim; i++)
            for (int j = 0; j < n; j++)
                vt[i * n + j] = v[j * n + i];
    }

    /// <summary>
    /// Computes y = x @ W_approx using spectral factors.
    /// y = (x @ leftFactor) @ rightFactor
    /// where leftFactor = U_r @ diag(σ_r): [M,r] and rightFactor = V_r^T: [r,N]
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void SpectralMatMul(
        float[] x, int xRows, int xCols,
        SpectralFactors factors,
        float[] output,
        float[]? workspace = null)
    {
        int r = factors.Rank;
        int n = factors.OriginalN;

        // Reuse caller-provided workspace to avoid per-call allocation on hot paths.
        // Caller can pre-allocate: workspace = new float[xRows * factors.Rank]
        var temp = workspace != null && workspace.Length >= xRows * r
            ? workspace
            : new float[xRows * r];
        Array.Clear(temp, 0, xRows * r);
        if (!BlasProvider.TryGemm(xRows, r, xCols, x, 0, xCols, factors.LeftFactor, 0, r, temp, 0, r))
        {
            SimdGemm.Sgemm(x.AsSpan(0, xRows * xCols), factors.LeftFactor.AsSpan(0, xCols * r),
                temp.AsSpan(), xRows, xCols, r);
        }

        // Step 2: output = temp[xRows, r] @ rightFactor[r, N] → [xRows, N]
        if (!BlasProvider.TryGemm(xRows, n, r, temp, 0, r, factors.RightFactor, 0, n, output, 0, n))
        {
            SimdGemm.Sgemm(temp.AsSpan(0, xRows * r), factors.RightFactor.AsSpan(0, r * n),
                output.AsSpan(0, xRows * n), xRows, r, n);
        }
    }
}

/// <summary>Result of SVD decomposition for spectral acceleration.</summary>
internal struct SpectralFactors
{
    public float[] LeftFactor;   // U_r @ diag(σ_r): [M, r]
    public float[] RightFactor;  // V_r^T: [r, N]
    public int Rank;
    public int OriginalM;
    public int OriginalN;
    public double ApproximationError;
    public float[] SingularValues;
}
