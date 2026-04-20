// Copyright (c) AiDotNet. All rights reserved.
// GLSL compute kernels for the torch.linalg decomposition surface (#211 moat #2).
// One workgroup per batch slice, local_size = 256 threads. Matrices with n > 256
// fall back to CPU through DirectGpuTensorEngine.
//
// The kernels are written as separate shader sources per op, matching the
// VulkanParity210Kernels pattern. Each pipeline is compiled independently via
// glslang / shaderc at backend init; if compilation fails for a specific op,
// ILinalgBackend isn't advertised for this backend and the engine falls back.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    internal static class VulkanLinalgKernels
    {
        private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

        // ═══════════════════════════════════════════════════════════════════
        // CHOLESKY
        // ═══════════════════════════════════════════════════════════════════
        public static string Cholesky => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer L { float l[]; };
layout(set = 0, binding = 2) buffer Info { int info[]; };
layout(push_constant) uniform P { int batchCount; int n; int upper; };
shared int localInfo;

void main() {
    uint b = gl_WorkGroupID.x;
    if (int(b) >= batchCount) return;
    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;

    uint off = b * uint(n * n);
    for (uint idx = tid; idx < uint(n * n); idx += blockSize) l[off + idx] = a[off + idx];
    if (tid == 0u) localInfo = 0;
    barrier();

    for (int j = 0; j < n; ++j) {
        if (tid == 0u) {
            float diag = l[off + j * n + j];
            for (int k = 0; k < j; ++k) {
                float l_jk = upper != 0 ? l[off + k * n + j] : l[off + j * n + k];
                diag -= l_jk * l_jk;
            }
            if (diag <= 0.0) { localInfo = j + 1; l[off + j * n + j] = 0.0; }
            else l[off + j * n + j] = sqrt(diag);
        }
        barrier();
        if (localInfo != 0) break;

        float djj = l[off + j * n + j];
        if (djj == 0.0) break;

        for (int i = j + 1 + int(tid); i < n; i += int(blockSize)) {
            float sum = upper != 0 ? l[off + j * n + i] : l[off + i * n + j];
            for (int k = 0; k < j; ++k) {
                float l_ik = upper != 0 ? l[off + k * n + i] : l[off + i * n + k];
                float l_jk = upper != 0 ? l[off + k * n + j] : l[off + j * n + k];
                sum -= l_ik * l_jk;
            }
            if (upper != 0) l[off + j * n + i] = sum / djj;
            else            l[off + i * n + j] = sum / djj;
        }
        barrier();
    }
    for (uint idx = tid; idx < uint(n * n); idx += blockSize) {
        int i = int(idx) / n, j = int(idx) % n;
        bool shouldZero = (upper != 0) ? (i > j) : (i < j);
        if (shouldZero) l[off + idx] = 0.0;
    }
    if (tid == 0u) info[b] = localInfo;
}";

        // ═══════════════════════════════════════════════════════════════════
        // LU FACTOR
        // ═══════════════════════════════════════════════════════════════════
        public static string LuFactor => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer LU { float lu[]; };
layout(set = 0, binding = 2) buffer Pivots { int pivots[]; };
layout(push_constant) uniform P { int batchCount; int m; int n; };
shared int pivRow;

void main() {
    uint b = gl_WorkGroupID.x;
    if (int(b) >= batchCount) return;
    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;
    int k = min(m, n);

    uint matOff = b * uint(m * n);
    uint pivOff = b * uint(k);

    for (uint idx = tid; idx < uint(m * n); idx += blockSize) lu[matOff + idx] = a[matOff + idx];
    barrier();

    for (int j = 0; j < k; ++j) {
        if (tid == 0u) {
            int best = j;
            float bestVal = abs(lu[matOff + j * n + j]);
            for (int i = j + 1; i < m; ++i) {
                float v = abs(lu[matOff + i * n + j]);
                if (v > bestVal) { bestVal = v; best = i; }
            }
            pivRow = best;
            pivots[pivOff + j] = best;
        }
        barrier();

        int p = pivRow;
        if (p != j) {
            for (int c = int(tid); c < n; c += int(blockSize)) {
                float t = lu[matOff + p * n + c];
                lu[matOff + p * n + c] = lu[matOff + j * n + c];
                lu[matOff + j * n + c] = t;
            }
            barrier();
        }
        float pivVal = lu[matOff + j * n + j];
        if (pivVal == 0.0) continue;
        for (int i = j + 1 + int(tid); i < m; i += int(blockSize)) lu[matOff + i * n + j] /= pivVal;
        barrier();
        int trailRows = m - j - 1;
        int trailCols = n - j - 1;
        for (int idx2 = int(tid); idx2 < trailRows * trailCols; idx2 += int(blockSize)) {
            int di = idx2 / trailCols;
            int dc = idx2 % trailCols;
            int i = j + 1 + di;
            int c = j + 1 + dc;
            lu[matOff + i * n + c] -= lu[matOff + i * n + j] * lu[matOff + j * n + c];
        }
        barrier();
    }
}";

        // ═══════════════════════════════════════════════════════════════════
        // QR REDUCED (simplified: identity Q for trivial cases; full Householder
        // via in-place reflector application). Vulkan spec requires all control
        // flow to be well-defined so we use an iteration cap.
        // ═══════════════════════════════════════════════════════════════════
        public static string QrReduced => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer Q { float q[]; };
layout(set = 0, binding = 2) buffer R { float r[]; };
layout(push_constant) uniform P { int batchCount; int m; int n; };
shared float sNorm;
shared float sAlpha;
shared float sBeta;
shared float sV0;
shared float partials[256];

void main() {
    uint b = gl_WorkGroupID.x;
    if (int(b) >= batchCount) return;
    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;
    int k = min(m, n);

    uint aOff = b * uint(m * n);
    uint qOff = b * uint(m * k);
    uint rOff = b * uint(k * n);

    for (uint idx = tid; idx < uint(m * n); idx += blockSize) {
        int i = int(idx) / n;
        if (i < k) r[rOff + idx] = a[aOff + idx];
    }
    barrier();
    for (uint idx = tid; idx < uint(m * k); idx += blockSize) {
        int i = int(idx) / k, j = int(idx) % k;
        q[qOff + i * k + j] = (i == j) ? 1.0 : 0.0;
    }
    barrier();

    for (int j = 0; j < k; ++j) {
        float partial = 0.0;
        for (int i = j + int(tid); i < m; i += int(blockSize)) {
            float v = (i < k) ? r[rOff + i * n + j] : a[aOff + i * n + j];
            partial += v * v;
        }
        partials[tid] = partial;
        barrier();
        for (uint s = blockSize / 2u; s > 0u; s >>= 1) {
            if (tid < s) partials[tid] += partials[tid + s];
            barrier();
        }
        if (tid == 0u) sNorm = sqrt(partials[0]);
        barrier();
        if (sNorm == 0.0) continue;

        if (tid == 0u) {
            float x0 = (j < k) ? r[rOff + j * n + j] : a[aOff + j * n + j];
            sAlpha = (x0 >= 0.0) ? -sNorm : sNorm;
            sV0 = x0 - sAlpha;
            float vNorm2 = sV0 * sV0 + (sNorm * sNorm - x0 * x0);
            sBeta = (vNorm2 > 0.0) ? (2.0 / vNorm2) : 0.0;
        }
        barrier();
        if (sBeta == 0.0) continue;

        for (int c = j + int(tid); c < n; c += int(blockSize)) {
            float dot = sV0 * ((j < k) ? r[rOff + j * n + c] : a[aOff + j * n + c]);
            for (int i = 1; i + j < m; ++i) {
                float v_i = (j + i < k) ? r[rOff + (j + i) * n + j] : a[aOff + (j + i) * n + j];
                float a_ic = (j + i < k) ? r[rOff + (j + i) * n + c] : a[aOff + (j + i) * n + c];
                dot += v_i * a_ic;
            }
            float s = sBeta * dot;
            if (j < k) r[rOff + j * n + c] -= s * sV0;
            for (int i = 1; i + j < m; ++i) {
                float v_i = (j + i < k) ? r[rOff + (j + i) * n + j] : a[aOff + (j + i) * n + j];
                if (j + i < k) r[rOff + (j + i) * n + c] -= s * v_i;
            }
        }
        barrier();
        for (int ri = int(tid); ri < m; ri += int(blockSize)) {
            float dot = sV0 * q[qOff + ri * k + j];
            float s = sBeta * dot;
            q[qOff + ri * k + j] -= s * sV0;
        }
        barrier();
    }
    for (uint idx = tid; idx < uint(k * n); idx += blockSize) {
        int i = int(idx) / n, j = int(idx) % n;
        if (i > j && i < k) r[rOff + idx] = 0.0;
    }
}";

        // ═══════════════════════════════════════════════════════════════════
        // EIGH (symmetric cyclic Jacobi; n ≤ 64 for shared-memory budget)
        // ═══════════════════════════════════════════════════════════════════
        public static string Eigh => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer W { float w[]; };
layout(set = 0, binding = 2) buffer V { float v[]; };
layout(push_constant) uniform P { int batchCount; int n; };
shared float Ash[64 * 64];
shared float Vsh[64 * 64];
shared float c_sh, s_sh;

void main() {
    uint b = gl_WorkGroupID.x;
    if (int(b) >= batchCount) return;
    uint tid = gl_LocalInvocationID.x;
    uint blockSize = gl_WorkGroupSize.x;
    const float eps = 1e-7;
    const int maxSweeps = 64;

    uint matOff = b * uint(n * n);
    uint wOff = b * uint(n);

    for (uint idx = tid; idx < uint(n * n); idx += blockSize) {
        int i = int(idx) / n, j = int(idx) % n;
        Ash[idx] = (i <= j) ? a[matOff + i * n + j] : a[matOff + j * n + i];
        Vsh[idx] = (i == j) ? 1.0 : 0.0;
    }
    barrier();

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        float offSum = 0.0;
        for (int p = 0; p < n; ++p)
            for (int qq = p + 1; qq < n; ++qq)
                offSum += Ash[p * n + qq] * Ash[p * n + qq];
        if (sqrt(offSum) < eps) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int qq = p + 1; qq < n; ++qq) {
                if (tid == 0u) {
                    float apq = Ash[p * n + qq];
                    if (abs(apq) < eps) { c_sh = 1.0; s_sh = 0.0; }
                    else {
                        float app = Ash[p * n + p];
                        float aqq = Ash[qq * n + qq];
                        float theta = (aqq - app) / (2.0 * apq);
                        float t = (theta == 0.0) ? 1.0 : sign(theta) / (abs(theta) + sqrt(1.0 + theta * theta));
                        c_sh = 1.0 / sqrt(1.0 + t * t);
                        s_sh = t * c_sh;
                    }
                }
                barrier();
                if (s_sh == 0.0) continue;
                float cc = c_sh, ss = s_sh;
                for (int i = int(tid); i < n; i += int(blockSize)) {
                    float aip = Ash[i * n + p];
                    float aiq = Ash[i * n + qq];
                    Ash[i * n + p] = cc * aip - ss * aiq;
                    Ash[i * n + qq] = ss * aip + cc * aiq;
                }
                barrier();
                for (int j = int(tid); j < n; j += int(blockSize)) {
                    float apj = Ash[p * n + j];
                    float aqj = Ash[qq * n + j];
                    Ash[p * n + j] = cc * apj - ss * aqj;
                    Ash[qq * n + j] = ss * apj + cc * aqj;
                }
                barrier();
                for (int i = int(tid); i < n; i += int(blockSize)) {
                    float vip = Vsh[i * n + p];
                    float viq = Vsh[i * n + qq];
                    Vsh[i * n + p] = cc * vip - ss * viq;
                    Vsh[i * n + qq] = ss * vip + cc * viq;
                }
                barrier();
            }
        }
    }

    if (tid == 0u) {
        for (int i = 0; i < n; ++i) w[wOff + i] = Ash[i * n + i];
        for (int i = 0; i < n - 1; ++i) {
            int minIdx = i;
            for (int j = i + 1; j < n; ++j) if (w[wOff + j] < w[wOff + minIdx]) minIdx = j;
            if (minIdx != i) {
                float tmp = w[wOff + i]; w[wOff + i] = w[wOff + minIdx]; w[wOff + minIdx] = tmp;
                for (int r2 = 0; r2 < n; ++r2) {
                    float vv = Vsh[r2 * n + i]; Vsh[r2 * n + i] = Vsh[r2 * n + minIdx]; Vsh[r2 * n + minIdx] = vv;
                }
            }
        }
    }
    barrier();
    for (uint idx = tid; idx < uint(n * n); idx += blockSize) v[matOff + idx] = Vsh[idx];
}";

        public static (string Name, string Source)[] All => new[]
        {
            ("parity211_cholesky", Cholesky),
            ("parity211_lu_factor", LuFactor),
            ("parity211_qr_reduced", QrReduced),
            ("parity211_eigh", Eigh),
        };
    }
}
