namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Small dense solvers for <see cref="IExtendedLinalgBackend"/> (#1142): one thread per matrix, float32, matrices up
/// to <see cref="LaMaxN"/> on a side. Each kernel is a line-for-line port of the managed reference it replaces
/// (LuDecomposition, LdlDecomposition, LinearSolvers, LinalgBackward and SvdWrapper), so pivot order, signs and
/// singular-vector ordering match the CPU path the engine falls back to. Offsets are explicit rather than pointer
/// arithmetic so the OpenCL, GLSL, WGSL and MSL ports read the same.
/// </summary>
internal static partial class BatchExecutionCudaKernels
{
    public const int LaMaxN = 8;

    public const string LaSymmetrize = "la_symmetrize", LaLuSolve = "la_lu_solve", LaLdlFactor = "la_ldl_factor",
        LaLdlSolve = "la_ldl_solve", LaSolveVector = "la_solve_vector", LaTriangularSolveVector = "la_tri_solve_vector",
        LaCholeskyBackwardLower = "la_cholesky_backward_lower", LaSolveBackwardVector = "la_solve_backward_vector",
        LaSvdReduced = "la_svd_reduced", LaQrReduced = "la_qr_reduced";

    internal const string SolverSource = @"
#define LA_MAXN 8

// out = the symmetric matrix read from one triangle of a (upper != 0: the upper triangle).
extern ""C"" __global__ __launch_bounds__(256) void la_symmetrize(const float* a, float* outp, int batch, int n, int upper) {
    int i = bx_gid(); if (i >= batch * n * n) return;
    int o = (i / (n * n)) * n * n; int r = (i / n) % n; int c = i % n;
    bool keep = upper != 0 ? (r <= c) : (r >= c);
    outp[i] = keep ? a[i] : a[o + c * n + r];
}

// LuDecomposition.Solve: apply the getrf row swaps, then unit-lower forward and upper backward substitution.
extern ""C"" __global__ __launch_bounds__(256) void la_lu_solve(const float* lu, const int* piv, const float* rhs, float* x,
    int batch, int n, int nrhs) {
    int b = bx_gid(); if (b >= batch) return;
    int ao = b * n * n; int po = b * n; int xo = b * n * nrhs;
    for (int i = 0; i < n * nrhs; i++) x[xo + i] = rhs[xo + i];
    for (int i = 0; i < n; i++) {
        int p = piv[po + i];
        if (p != i && p >= 0 && p < n) {
            for (int c = 0; c < nrhs; c++) {
                float t = x[xo + i * nrhs + c]; x[xo + i * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t;
            }
        }
    }
    for (int c = 0; c < nrhs; c++) {
        for (int i = 1; i < n; i++) {
            float s = x[xo + i * nrhs + c];
            for (int j = 0; j < i; j++) s -= lu[ao + i * n + j] * x[xo + j * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int i = n - 1; i >= 0; i--) {
            float s = x[xo + i * nrhs + c];
            for (int j = i + 1; j < n; j++) s -= lu[ao + i * n + j] * x[xo + j * nrhs + c];
            x[xo + i * nrhs + c] = s / lu[ao + i * n + i];
        }
    }
}

// LdlDecomposition.Factor: symmetric diagonal pivoting on the largest remaining diagonal, factored in place in ld.
extern ""C"" __global__ __launch_bounds__(256) void la_ldl_factor(const float* a, float* ld, int* piv, int batch, int n, int upper) {
    int b = bx_gid(); if (b >= batch) return;
    int o = b * n * n; int po = b * n;
    for (int i = 0; i < n * n; i++) ld[o + i] = a[o + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabsf(ld[o + j * n + j]);
        for (int i = j + 1; i < n; i++) {
            float v = fabsf(ld[o + i * n + i]);
            if (v > best) { best = v; p = i; }
        }
        piv[po + j] = p;
        if (p != j) {
            for (int k = 0; k < n; k++) { float t = ld[o + j * n + k]; ld[o + j * n + k] = ld[o + p * n + k]; ld[o + p * n + k] = t; }
            for (int k = 0; k < n; k++) { float t = ld[o + k * n + j]; ld[o + k * n + j] = ld[o + k * n + p]; ld[o + k * n + p] = t; }
        }
        float d = ld[o + j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float lij = ld[o + i * n + j] / d;
            ld[o + i * n + j] = lij;
            for (int k = j + 1; k < n; k++) ld[o + i * n + k] -= lij * ld[o + j * n + k];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) ld[o + i * n + j] = 0.0f;
    if (upper != 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) { ld[o + j * n + i] = ld[o + i * n + j]; ld[o + i * n + j] = 0.0f; }
    }
}

// LdlDecomposition.Solve: permute, unit-L forward, divide by D (NaN on a zero pivot), L^T backward, unpermute.
extern ""C"" __global__ __launch_bounds__(256) void la_ldl_solve(const float* ld, const int* piv, const float* rhs, float* x,
    int batch, int n, int nrhs, int upper) {
    int b = bx_gid(); if (b >= batch) return;
    int o = b * n * n; int po = b * n; int xo = b * n * nrhs;
    for (int i = 0; i < n * nrhs; i++) x[xo + i] = rhs[xo + i];
    for (int c = 0; c < nrhs; c++) {
        for (int j = 0; j < n; j++) {
            int p = piv[po + j];
            if (p != j && p >= 0 && p < n) { float t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t; }
        }
        for (int i = 0; i < n; i++) {
            float s = x[xo + i * nrhs + c];
            for (int k = 0; k < i; k++) s -= (upper != 0 ? ld[o + k * n + i] : ld[o + i * n + k]) * x[xo + k * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int i = 0; i < n; i++) {
            float d = ld[o + i * n + i];
            x[xo + i * nrhs + c] = d == 0.0f ? __uint_as_float(0x7fc00000u) : x[xo + i * nrhs + c] / d;
        }
        for (int i = n - 1; i >= 0; i--) {
            float s = x[xo + i * nrhs + c];
            for (int k = i + 1; k < n; k++) s -= (upper != 0 ? ld[o + i * n + k] : ld[o + k * n + i]) * x[xo + k * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int j = n - 1; j >= 0; j--) {
            int p = piv[po + j];
            if (p != j && p >= 0 && p < n) { float t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t; }
        }
    }
}

// LinearSolvers.SolveEx for one vector: partial-pivot LU in registers, info = 1 + the first zero pivot (0: none).
extern ""C"" __global__ __launch_bounds__(256) void la_solve_vector(const float* a, const float* rhs, float* x, int* info,
    int batch, int n) {
    int b = bx_gid(); if (b >= batch) return;
    float w[LA_MAXN * LA_MAXN]; int pv[LA_MAXN];
    int o = b * n * n; int xo = b * n;
    for (int i = 0; i < n * n; i++) w[i] = a[o + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabsf(w[j * n + j]);
        for (int i = j + 1; i < n; i++) { float v = fabsf(w[i * n + j]); if (v > best) { best = v; p = i; } }
        pv[j] = p;
        if (p != j) for (int c = 0; c < n; c++) { float t = w[p * n + c]; w[p * n + c] = w[j * n + c]; w[j * n + c] = t; }
        float d = w[j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float f = w[i * n + j] / d;
            w[i * n + j] = f;
            for (int c = j + 1; c < n; c++) w[i * n + c] -= f * w[j * n + c];
        }
    }
    int status = 0;
    for (int k = 0; k < n; k++) if (w[k * n + k] == 0.0f) { status = k + 1; break; }
    info[b] = status;
    for (int i = 0; i < n; i++) x[xo + i] = rhs[xo + i];
    for (int i = 0; i < n; i++) { int p = pv[i]; if (p != i) { float t = x[xo + i]; x[xo + i] = x[xo + p]; x[xo + p] = t; } }
    for (int i = 1; i < n; i++) { float s = x[xo + i]; for (int j = 0; j < i; j++) s -= w[i * n + j] * x[xo + j]; x[xo + i] = s; }
    for (int i = n - 1; i >= 0; i--) {
        float s = x[xo + i];
        for (int j = i + 1; j < n; j++) s -= w[i * n + j] * x[xo + j];
        x[xo + i] = s / w[i * n + i];
    }
}

// LinearSolvers.SolveTriangular for one vector.
extern ""C"" __global__ __launch_bounds__(256) void la_tri_solve_vector(const float* a, const float* rhs, float* x,
    int batch, int n, int upper, int unitDiagonal) {
    int b = bx_gid(); if (b >= batch) return;
    int o = b * n * n; int xo = b * n;
    for (int i = 0; i < n; i++) x[xo + i] = rhs[xo + i];
    for (int st = 0; st < n; st++) {
        int i = upper != 0 ? n - 1 - st : st;
        float s = x[xo + i];
        if (upper != 0) { for (int j = i + 1; j < n; j++) s -= a[o + i * n + j] * x[xo + j]; }
        else { for (int j = 0; j < i; j++) s -= a[o + i * n + j] * x[xo + j]; }
        x[xo + i] = unitDiagonal != 0 ? s : s / a[o + i * n + i];
    }
}

// LinalgBackward.CholeskyBackward (lower): gradA = 1/2 L^-T (Phi + Phi^T) L^-1, Phi = tril(L^T gradL), diagonal halved.
extern ""C"" __global__ __launch_bounds__(256) void la_cholesky_backward_lower(const float* l, const float* g, float* ga,
    int batch, int n) {
    int b = bx_gid(); if (b >= batch) return;
    float t[LA_MAXN * LA_MAXN]; float inv[LA_MAXN * LA_MAXN]; float row[LA_MAXN];
    int o = b * n * n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += l[o + k * n + i] * g[o + k * n + j];
            t[i * n + j] = s;
        }
    // Phi zeroes the upper triangle and halves the diagonal, so Phi + Phi^T mirrors the lower triangle.
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) t[i * n + j] = t[j * n + i];
    for (int c = 0; c < n; c++)
        for (int i = 0; i < n; i++) {
            float s = i == c ? 1.0f : 0.0f;
            for (int k = 0; k < i; k++) s -= l[o + i * n + k] * inv[k * n + c];
            inv[i * n + c] = s / l[o + i * n + i];
        }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float s = 0.0f;
            for (int r = 0; r < n; r++) s += inv[r * n + i] * t[r * n + k];
            row[k] = s;
        }
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += row[k] * inv[k * n + j];
            ga[o + i * n + j] = 0.5f * s;
        }
    }
}

// LinalgBackward.SolveBackward for a vector rhs: gradB = A^-T gradX, gradA = -gradB x^T.
extern ""C"" __global__ __launch_bounds__(256) void la_solve_backward_vector(const float* a, const float* xs, const float* g,
    float* ga, float* gb, int batch, int n) {
    int b = bx_gid(); if (b >= batch) return;
    float w[LA_MAXN * LA_MAXN]; int pv[LA_MAXN]; float y[LA_MAXN];
    int o = b * n * n; int vo = b * n;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) w[i * n + j] = a[o + j * n + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabsf(w[j * n + j]);
        for (int i = j + 1; i < n; i++) { float v = fabsf(w[i * n + j]); if (v > best) { best = v; p = i; } }
        pv[j] = p;
        if (p != j) for (int c = 0; c < n; c++) { float t = w[p * n + c]; w[p * n + c] = w[j * n + c]; w[j * n + c] = t; }
        float d = w[j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float f = w[i * n + j] / d;
            w[i * n + j] = f;
            for (int c = j + 1; c < n; c++) w[i * n + c] -= f * w[j * n + c];
        }
    }
    for (int i = 0; i < n; i++) y[i] = g[vo + i];
    for (int i = 0; i < n; i++) { int p = pv[i]; if (p != i) { float t = y[i]; y[i] = y[p]; y[p] = t; } }
    for (int i = 1; i < n; i++) { float s = y[i]; for (int j = 0; j < i; j++) s -= w[i * n + j] * y[j]; y[i] = s; }
    for (int i = n - 1; i >= 0; i--) { float s = y[i]; for (int j = i + 1; j < n; j++) s -= w[i * n + j] * y[j]; y[i] = s / w[i * n + i]; }
    for (int i = 0; i < n; i++) {
        gb[vo + i] = y[i];
        for (int j = 0; j < n; j++) ga[o + i * n + j] = -y[i] * xs[vo + j];
    }
}

// QrDecomposition (reduced): Householder with alpha = -sign(x0) |x|, as LAPACK geqrf, so R's diagonal carries the
// same signs as the CPU and PyTorch. work holds, per matrix, the m x n working copy (reflector j in column j from
// row j down) followed by k betas and k diagonal entries of R.
extern ""C"" __global__ __launch_bounds__(256) void la_qr_reduced(const float* a, float* q, float* r, float* work,
    int batch, int m, int n) {
    int b = bx_gid(); if (b >= batch) return;
    int k = m < n ? m : n;
    int ao = b * m * n; int wo = b * (m * n + 2 * k); int bo = wo + m * n; int dg = bo + k;
    int qo = b * m * k; int ro = b * k * n;
    for (int i = 0; i < m * n; i++) work[wo + i] = a[ao + i];
    for (int j = 0; j < k; j++) {
        float norm2 = 0.0f;
        for (int i = j; i < m; i++) norm2 += work[wo + i * n + j] * work[wo + i * n + j];
        float norm = sqrtf(norm2);
        float x0 = work[wo + j * n + j];
        float alpha = x0 > 0.0f ? -norm : (x0 < 0.0f ? norm : 0.0f);
        if (x0 == 0.0f && alpha == 0.0f) alpha = norm;
        work[wo + j * n + j] = x0 - alpha;
        float vnorm2 = 0.0f;
        for (int i = j; i < m; i++) vnorm2 += work[wo + i * n + j] * work[wo + i * n + j];
        float beta = vnorm2 == 0.0f ? 0.0f : 2.0f / vnorm2;
        work[bo + j] = beta;
        work[dg + j] = alpha;
        for (int c = j + 1; c < n; c++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += work[wo + i * n + j] * work[wo + i * n + c];
            float sc = beta * dot;
            for (int i = j; i < m; i++) work[wo + i * n + c] -= sc * work[wo + i * n + j];
        }
    }
    for (int i = 0; i < k; i++)
        for (int c = 0; c < n; c++)
            r[ro + i * n + c] = c < i ? 0.0f : (c == i ? work[dg + i] : work[wo + i * n + c]);
    for (int i = 0; i < m; i++)
        for (int c = 0; c < k; c++) q[qo + i * k + c] = i == c ? 1.0f : 0.0f;
    for (int j = k - 1; j >= 0; j--) {
        float beta = work[bo + j];
        if (beta == 0.0f) continue;
        for (int c = 0; c < k; c++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += work[wo + i * n + j] * q[qo + i * k + c];
            float sc = beta * dot;
            for (int i = j; i < m; i++) q[qo + i * k + c] -= sc * work[wo + i * n + j];
        }
    }
}

// SvdWrapper.Full (reduced): Jacobi eigendecomposition of the smaller Gram matrix, singular values descending.
extern ""C"" __global__ __launch_bounds__(256) void la_svd_reduced(const float* a, float* u, float* s, float* vh,
    int batch, int m, int n) {
    int b = bx_gid(); if (b >= batch) return;
    float g[LA_MAXN * LA_MAXN]; float v[LA_MAXN * LA_MAXN]; float w[LA_MAXN]; int ord[LA_MAXN];
    bool tall = m >= n; int k = m < n ? m : n; int d = tall ? n : m;
    int ao = b * m * n; int uo = b * m * k; int so = b * k; int vo = b * k * n;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            float acc = 0.0f;
            if (tall) { for (int r = 0; r < m; r++) acc += a[ao + r * n + i] * a[ao + r * n + j]; }
            else { for (int r = 0; r < n; r++) acc += a[ao + i * n + r] * a[ao + j * n + r]; }
            g[i * d + j] = acc;
            v[i * d + j] = i == j ? 1.0f : 0.0f;
        }
    float scale = 0.0f;
    for (int i = 0; i < d * d; i++) scale += g[i] * g[i];
    scale = sqrtf(scale);
    for (int sweep = 0; sweep < 100; sweep++) {
        float off = 0.0f;
        for (int p = 0; p < d - 1; p++) for (int q = p + 1; q < d; q++) off += g[p * d + q] * g[p * d + q];
        off = sqrtf(off);
        if (off < 1e-12f || off <= 1e-7f * scale) break;
        for (int p = 0; p < d - 1; p++)
            for (int q = p + 1; q < d; q++) {
                float apq = g[p * d + q];
                if (fabsf(apq) < 1e-12f) continue;
                float theta = (g[q * d + q] - g[p * d + p]) / (2.0f * apq);
                float sg = theta > 0.0f ? 1.0f : (theta < 0.0f ? -1.0f : 0.0f);
                float t = theta == 0.0f ? 1.0f : sg / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float sn = t * c;
                for (int i = 0; i < d; i++) {
                    float aip = g[i * d + p]; float aiq = g[i * d + q];
                    g[i * d + p] = c * aip - sn * aiq; g[i * d + q] = sn * aip + c * aiq;
                }
                for (int i = 0; i < d; i++) {
                    float api = g[p * d + i]; float aqi = g[q * d + i];
                    g[p * d + i] = c * api - sn * aqi; g[q * d + i] = sn * api + c * aqi;
                }
                for (int i = 0; i < d; i++) {
                    float vip = v[i * d + p]; float viq = v[i * d + q];
                    v[i * d + p] = c * vip - sn * viq; v[i * d + q] = sn * vip + c * viq;
                }
            }
    }
    for (int i = 0; i < d; i++) { w[i] = g[i * d + i]; ord[i] = i; }
    for (int i = 1; i < d; i++) {
        int cur = ord[i]; int j = i - 1;
        while (j >= 0 && w[ord[j]] < w[cur]) { ord[j + 1] = ord[j]; j--; }
        ord[j + 1] = cur;
    }
    for (int j = 0; j < k; j++) {
        float sigma = sqrtf(fmaxf(0.0f, w[ord[j]]));
        s[so + j] = sigma;
        int e = ord[j];
        if (tall) {
            for (int c = 0; c < n; c++) vh[vo + j * n + c] = v[c * d + e];
            for (int r = 0; r < m; r++) {
                float acc = 0.0f;
                for (int c = 0; c < n; c++) acc += a[ao + r * n + c] * v[c * d + e];
                u[uo + r * k + j] = sigma > 1e-14f ? acc / sigma : 0.0f;
            }
        } else {
            for (int r = 0; r < m; r++) u[uo + r * k + j] = v[r * d + e];
            for (int c = 0; c < n; c++) {
                float acc = 0.0f;
                for (int r = 0; r < m; r++) acc += v[r * d + e] * a[ao + r * n + c];
                vh[vo + j * n + c] = sigma > 1e-14f ? acc / sigma : 0.0f;
            }
        }
    }
}
";
}
