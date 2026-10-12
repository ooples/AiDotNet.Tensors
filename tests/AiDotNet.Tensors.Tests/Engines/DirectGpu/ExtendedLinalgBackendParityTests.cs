#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Decompositions;
using AiDotNet.Tensors.LinearAlgebra.Solvers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IExtendedLinalgBackend"/> on every non-CUDA backend (#1142) against the managed references the engine
/// falls back to, so routing a solver to the device changes no result: same pivots, same signs, same singular-vector
/// order. Sizes cover the 4x4 the engine routes plus a smaller and the largest register-resident size, and SVD a tall
/// and a wide shape. CUDA's direct-PTX family is exact-4x4 on SM86 and has its own tests.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ExtendedLinalgBackendParityTests
{
    private const int Batch = 3;
    private const float Tolerance = 2e-3f;
    private static readonly int[] Sizes = { 3, 4, 8 };

    private static float[] Random(int count, int seed, float diagonalBoost = 0f, int n = 0, bool symmetric = false)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new float[count];
        for (int i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        if (n > 0)
        {
            for (int b = 0; b < count / (n * n); b++)
            {
                int o = b * n * n;
                for (int i = 0; i < n; i++)
                {
                    if (symmetric) for (int j = 0; j < i; j++) data[o + j * n + i] = data[o + i * n + j];
                    data[o + i * n + i] += diagonalBoost * (i % 2 == 0 ? 1 : -1);
                }
            }
        }
        return data;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float scale = Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance * scale,
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]}");
        }
    }

    private static IGpuBuffer UploadInts(IDirectGpuBackend backend, int[] values)
    {
        var bits = new float[values.Length];
        for (int i = 0; i < values.Length; i++) bits[i] = BitConverter.Int32BitsToSingle(values[i]);
        return backend.AllocateBuffer(bits);
    }

    private static int[] DownloadInts(IDirectGpuBackend backend, IGpuBuffer buffer)
    {
        float[] bits = backend.DownloadBuffer(buffer);
        var values = new int[bits.Length];
        for (int i = 0; i < bits.Length; i++) values[i] = BitConverter.SingleToInt32Bits(bits[i]);
        return values;
    }

    private static void Check(IDirectGpuBackend backend)
    {
        var lin = Assert.IsAssignableFrom<IExtendedLinalgBackend>(backend);
        foreach (int n in Sizes)
        {
            CheckLuSolve(backend, lin, n);
            CheckLdl(backend, lin, n);
            CheckSolveVector(backend, lin, n);
            CheckTriangular(backend, lin, n);
            CheckCholeskyBackward(backend, lin, n);
            CheckSolveBackward(backend, lin, n);
            CheckEigh(backend, lin, n);
            CheckSvd(backend, lin, n, n);
        }
        CheckSvd(backend, lin, 5, 3);
        CheckSvd(backend, lin, 3, 5);
        CheckQr(backend, lin);
    }

    private static void CheckLuSolve(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        var a = new Tensor<float>(Random(Batch * n * n, 11 + n, diagonalBoost: 0.5f, n: n), new[] { Batch, n, n });
        var (lu, pivots) = LuDecomposition.Factor(a);
        foreach (int columns in new[] { 1, 2 })
        {
            bool vector = columns == 1;
            int[] shape = vector ? new[] { Batch, n } : new[] { Batch, n, columns };
            var rhs = new Tensor<float>(Random(Batch * n * columns, 13 + n), shape);
            float[] expected = LuDecomposition.Solve(lu, pivots, rhs).GetDataArray();
            using var luBuf = backend.AllocateBuffer(lu.GetDataArray());
            using var pivBuf = UploadInts(backend, pivots.GetDataArray());
            using var rhsBuf = backend.AllocateBuffer(rhs.GetDataArray());
            using var outBuf = backend.AllocateBuffer(rhs.Length);
            lin.LinalgLuSolve(luBuf, pivBuf, rhsBuf, outBuf, Batch, n, columns, vector);
            AssertClose(expected, backend.DownloadBuffer(outBuf), $"LuSolve n={n} rhs={columns}");
        }
    }

    private static void CheckLdl(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        // Symmetric indefinite with diagonals of mixed magnitude, so the diagonal pivoting has to reorder.
        var a = new Tensor<float>(Random(Batch * n * n, 17 + n, diagonalBoost: 1.5f, n: n, symmetric: true), new[] { Batch, n, n });
        var rhs = new Tensor<float>(Random(Batch * n, 19 + n), new[] { Batch, n });
        foreach (bool upper in new[] { false, true })
        {
            var (ld, pivots) = LdlDecomposition.Factor(a, upper);
            using var aBuf = backend.AllocateBuffer(a.GetDataArray());
            using var ldBuf = backend.AllocateBuffer(a.Length);
            using var pivBuf = backend.AllocateBuffer(Batch * n);
            lin.LinalgLdlFactor(aBuf, ldBuf, pivBuf, Batch, n, upper);
            AssertClose(ld.GetDataArray(), backend.DownloadBuffer(ldBuf), $"LdlFactor n={n} upper={upper}");
            Assert.Equal(pivots.GetDataArray(), DownloadInts(backend, pivBuf));

            float[] expected = LdlDecomposition.Solve(ld, pivots, rhs, upper).GetDataArray();
            using var cpuLd = backend.AllocateBuffer(ld.GetDataArray());
            using var cpuPiv = UploadInts(backend, pivots.GetDataArray());
            using var rhsBuf = backend.AllocateBuffer(rhs.GetDataArray());
            using var outBuf = backend.AllocateBuffer(rhs.Length);
            lin.LinalgLdlSolve(cpuLd, cpuPiv, rhsBuf, outBuf, Batch, n, upper, rhsIsVector: true);
            AssertClose(expected, backend.DownloadBuffer(outBuf), $"LdlSolve n={n} upper={upper}");
        }
    }

    private static void CheckSolveVector(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        float[] data = Random(Batch * n * n, 23 + n, diagonalBoost: 0.5f, n: n);
        // The last matrix is singular (a zero column), so info must name its first zero pivot.
        for (int i = 0; i < n; i++) data[(Batch - 1) * n * n + i * n + 1] = 0f;
        var a = new Tensor<float>(data, new[] { Batch, n, n });
        var rhs = new Tensor<float>(Random(Batch * n, 29 + n), new[] { Batch, n });
        var (expected, info) = LinearSolvers.SolveEx(a, rhs);
        using var aBuf = backend.AllocateBuffer(a.GetDataArray());
        using var rhsBuf = backend.AllocateBuffer(rhs.GetDataArray());
        using var outBuf = backend.AllocateBuffer(rhs.Length);
        using var infoBuf = backend.AllocateBuffer(Batch);
        lin.LinalgSolveVector(aBuf, rhsBuf, outBuf, infoBuf, Batch, n);
        int[] gotInfo = DownloadInts(backend, infoBuf);
        Assert.Equal(info.GetDataArray(), gotInfo);
        Assert.NotEqual(0, gotInfo[Batch - 1]);
        float[] got = backend.DownloadBuffer(outBuf);
        AssertClose(expected.GetDataArray().AsSpan(0, (Batch - 1) * n).ToArray(), got.AsSpan(0, (Batch - 1) * n).ToArray(),
            $"Solve n={n}");
    }

    private static void CheckTriangular(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        var a = new Tensor<float>(Random(Batch * n * n, 31 + n, diagonalBoost: 2f, n: n), new[] { Batch, n, n });
        var rhs = new Tensor<float>(Random(Batch * n, 37 + n), new[] { Batch, n });
        foreach (bool upper in new[] { false, true })
            foreach (bool unit in new[] { false, true })
            {
                float[] expected = LinearSolvers.SolveTriangular(a, rhs, upper, unit).GetDataArray();
                using var aBuf = backend.AllocateBuffer(a.GetDataArray());
                using var rhsBuf = backend.AllocateBuffer(rhs.GetDataArray());
                using var outBuf = backend.AllocateBuffer(rhs.Length);
                lin.LinalgTriangularSolveVector(aBuf, rhsBuf, outBuf, Batch, n, upper, unit);
                AssertClose(expected, backend.DownloadBuffer(outBuf), $"SolveTriangular n={n} upper={upper} unit={unit}");
            }
    }

    private static void CheckCholeskyBackward(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        float[] l = Random(Batch * n * n, 41 + n);
        for (int b = 0; b < Batch; b++)
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++) l[b * n * n + i * n + j] = 0f;
                l[b * n * n + i * n + i] = 1.5f + 0.25f * i;
            }
        float[] g = Random(Batch * n * n, 43 + n);
        var expected = new float[l.Length];
        for (int b = 0; b < Batch; b++) CholeskyBackwardReference(l, g, expected, b * n * n, n);
        using var lBuf = backend.AllocateBuffer(l);
        using var gBuf = backend.AllocateBuffer(g);
        using var outBuf = backend.AllocateBuffer(l.Length);
        lin.LinalgCholeskyBackwardLower(lBuf, gBuf, outBuf, Batch, n);
        AssertClose(expected, backend.DownloadBuffer(outBuf), $"CholeskyBackward n={n}");
    }

    /// <summary>LinalgBackward.CholeskyBackward in double: 1/2 L^-T (Phi + Phi^T) L^-1, Phi = tril(L^T dL), diag halved.</summary>
    private static void CholeskyBackwardReference(float[] l, float[] g, float[] result, int o, int n)
    {
        var phi = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++)
            {
                double s = 0;
                for (int k = 0; k < n; k++) s += (double)l[o + k * n + i] * g[o + k * n + j];
                phi[i * n + j] = i == j ? 0.5 * s : s;
            }
        var sym = new double[n * n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) sym[i * n + j] = phi[i * n + j] + phi[j * n + i];
        var inv = new double[n * n];
        for (int c = 0; c < n; c++)
            for (int i = 0; i < n; i++)
            {
                double s = i == c ? 1 : 0;
                for (int k = 0; k < i; k++) s -= l[o + i * n + k] * inv[k * n + c];
                inv[i * n + c] = s / l[o + i * n + i];
            }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int r = 0; r < n; r++)
                    for (int k = 0; k < n; k++) s += inv[r * n + i] * sym[r * n + k] * inv[k * n + j];
                result[o + i * n + j] = (float)(0.5 * s);
            }
    }

    private static void CheckSolveBackward(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        var a = new Tensor<float>(Random(Batch * n * n, 47 + n, diagonalBoost: 0.5f, n: n), new[] { Batch, n, n });
        var x = new Tensor<float>(Random(Batch * n, 53 + n), new[] { Batch, n });
        var g = new Tensor<float>(Random(Batch * n, 59 + n), new[] { Batch, n });
        var aT = new Tensor<float>(new[] { Batch, n, n });
        float[] ad = a.GetDataArray(), atd = aT.GetDataArray();
        for (int b = 0; b < Batch; b++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) atd[b * n * n + i * n + j] = ad[b * n * n + j * n + i];
        float[] gradB = LinearSolvers.SolveEx(aT, g).Solution.GetDataArray();
        float[] xd = x.GetDataArray();
        var gradA = new float[Batch * n * n];
        for (int b = 0; b < Batch; b++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) gradA[b * n * n + i * n + j] = -gradB[b * n + i] * xd[b * n + j];
        using var aBuf = backend.AllocateBuffer(ad);
        using var xBuf = backend.AllocateBuffer(xd);
        using var gBuf = backend.AllocateBuffer(g.GetDataArray());
        using var gaBuf = backend.AllocateBuffer(gradA.Length);
        using var gbBuf = backend.AllocateBuffer(gradB.Length);
        lin.LinalgSolveBackwardVector(aBuf, xBuf, gBuf, gaBuf, gbBuf, Batch, n);
        AssertClose(gradB, backend.DownloadBuffer(gbBuf), $"SolveBackward gradB n={n}");
        AssertClose(gradA, backend.DownloadBuffer(gaBuf), $"SolveBackward gradA n={n}");
    }

    private static void CheckEigh(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int n)
    {
        // Not symmetric: each mode must read only its own triangle, as the managed reference does.
        var a = new Tensor<float>(Random(Batch * n * n, 61 + n), new[] { Batch, n, n });
        foreach (bool upper in new[] { false, true })
        {
            var (w, v) = EighDecomposition.Compute(a, upper);
            using var aBuf = backend.AllocateBuffer(a.GetDataArray());
            using var wBuf = backend.AllocateBuffer(Batch * n);
            using var vBuf = backend.AllocateBuffer(Batch * n * n);
            lin.LinalgEighSymmetric(aBuf, wBuf, vBuf, Batch, n, upper);
            AssertClose(w.GetDataArray(), backend.DownloadBuffer(wBuf), $"Eigh values n={n} upper={upper}");
            // Eigenvectors are unique up to sign: compare |<v_cpu, v_gpu>| per column.
            float[] ev = v.GetDataArray(), gv = backend.DownloadBuffer(vBuf);
            for (int b = 0; b < Batch; b++)
                for (int c = 0; c < n; c++)
                {
                    double dot = 0;
                    for (int r = 0; r < n; r++) dot += (double)ev[b * n * n + r * n + c] * gv[b * n * n + r * n + c];
                    Assert.True(Math.Abs(Math.Abs(dot) - 1) <= Tolerance,
                        $"Eigh vector n={n} upper={upper} batch {b} column {c}: |dot| = {Math.Abs(dot)}");
                }
        }
    }

    private static void CheckSvd(IDirectGpuBackend backend, IExtendedLinalgBackend lin, int m, int n)
    {
        int k = Math.Min(m, n);
        var a = new Tensor<float>(Random(Batch * m * n, 67 + m * 10 + n), new[] { Batch, m, n });
        var (u, s, vh) = SvdWrapper.Full(a, fullMatrices: false);
        using var aBuf = backend.AllocateBuffer(a.GetDataArray());
        using var uBuf = backend.AllocateBuffer(Batch * m * k);
        using var sBuf = backend.AllocateBuffer(Batch * k);
        using var vhBuf = backend.AllocateBuffer(Batch * k * n);
        lin.LinalgSvdReduced(aBuf, uBuf, sBuf, vhBuf, Batch, m, n);
        AssertClose(s.GetDataArray(), backend.DownloadBuffer(sBuf), $"Svd S {m}x{n}");
        AssertClose(u.GetDataArray(), backend.DownloadBuffer(uBuf), $"Svd U {m}x{n}");
        AssertClose(vh.GetDataArray(), backend.DownloadBuffer(vhBuf), $"Svd Vh {m}x{n}");
    }

    private static void CheckQr(IDirectGpuBackend backend, IExtendedLinalgBackend lin)
    {
        // R's diagonal signs are the Householder ones (LAPACK geqrf / torch.linalg.qr), not Gram-Schmidt's positive ones.
        foreach (var (m, n) in new[] { (4, 4), (6, 3), (3, 6), (16, 16) })
        {
            int k = Math.Min(m, n);
            var a = new Tensor<float>(Random(Batch * m * n, 71 + m * 10 + n), new[] { Batch, m, n });
            var (q, r) = QrDecomposition.Compute(a, "reduced");
            using var aBuf = backend.AllocateBuffer(a.GetDataArray());
            using var qBuf = backend.AllocateBuffer(Batch * m * k);
            using var rBuf = backend.AllocateBuffer(Batch * k * n);
            lin.LinalgQrReducedDirect(aBuf, qBuf, rBuf, Batch, m, n);
            AssertClose(q.GetDataArray(), backend.DownloadBuffer(qBuf), $"QR Q {m}x{n}");
            AssertClose(r.GetDataArray(), backend.DownloadBuffer(rBuf), $"QR R {m}x{n}");
        }
    }

    [SkippableFact]
    public void OpenCl_ExtendedLinalg_MatchesManagedReference()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Vulkan_ExtendedLinalg_MatchesManagedReference()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend);
    }

    [SkippableFact]
    public void Hip_ExtendedLinalg_MatchesManagedReference()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend);
    }
}
#endif
