namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shaders (Vulkan) for the parity-210 op surface.
/// Mirrors CudaParity210Kernels / HipParity210Kernels / MetalParity210Kernels
/// function-for-function. Each shader is compiled to SPIR-V at runtime via
/// <see cref="VulkanGlslCompiler"/> (libshaderc) and dispatched with
/// 1-D workgroups of local_size_x = 256.
/// </summary>
public static class VulkanParity210Kernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string TwoBuf = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer Out { float o[]; };
";

    private const string ThreeBuf = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Out { float o[]; };
";

    private const string IdxTake = @"
layout(set = 0, binding = 0) readonly buffer Src { float src[]; };
layout(set = 0, binding = 1) readonly buffer Idx { int idx[]; };
layout(set = 0, binding = 2) writeonly buffer Out { float o[]; };
";

    private const string IdxScatterReadWrite = @"
layout(set = 0, binding = 0) buffer Out { float o[]; };
layout(set = 0, binding = 1) readonly buffer Idx { int idx[]; };
layout(set = 0, binding = 2) readonly buffer Src { float src[]; };
";

    // =====================================================================
    // Movement
    // =====================================================================

    public static string Roll1D => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; int shift; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * axisSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int ax = tmp % axisSize;
    int outer = tmp / axisSize;
    int src = ((ax - shift) % axisSize + axisSize) % axisSize;
    o[gid] = a[(outer * axisSize + src) * innerSize + inner];
}";

    public static string FlipAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * axisSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int ax = tmp % axisSize;
    int outer = tmp / axisSize;
    int src = axisSize - 1 - ax;
    o[gid] = a[(outer * axisSize + src) * innerSize + inner];
}";

    public static string Triu => Header + TwoBuf + @"
layout(push_constant) uniform P { int batchSize; int rows; int cols; int diagonal; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = batchSize * rows * cols;
    if (gid >= total) return;
    int col = gid % cols;
    int tmp = gid / cols;
    int row = tmp % rows;
    o[gid] = ((col - row) >= diagonal) ? a[gid] : 0.0;
}";

    public static string Tril => Header + TwoBuf + @"
layout(push_constant) uniform P { int batchSize; int rows; int cols; int diagonal; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = batchSize * rows * cols;
    if (gid >= total) return;
    int col = gid % cols;
    int tmp = gid / cols;
    int row = tmp % rows;
    o[gid] = ((col - row) <= diagonal) ? a[gid] : 0.0;
}";

    public static string DiagEmbed => Header + TwoBuf + @"
layout(push_constant) uniform P { int batchSize; int diagLen; int matSize; int offset; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = batchSize * matSize * matSize;
    if (gid >= total) return;
    int col = gid % matSize;
    int tmp = gid / matSize;
    int row = tmp % matSize;
    int b = tmp / matSize;
    int diagRow = (offset >= 0) ? row : row + offset;
    int diagCol = (offset >= 0) ? col - offset : col;
    if (diagRow == diagCol && diagRow >= 0 && diagRow < diagLen)
        o[gid] = a[b * diagLen + diagRow];
    else
        o[gid] = 0.0;
}";

    // =====================================================================
    // Cumulative (one thread per (outer, inner) line)
    // =====================================================================

    public static string CumSumAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 0.0;
    for (int k = 0; k < axisSize; ++k) {
        acc += a[base_ + k * innerSize];
        o[base_ + k * innerSize] = acc;
    }
}";

    public static string CumProdAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 1.0;
    for (int k = 0; k < axisSize; ++k) {
        acc *= a[base_ + k * innerSize];
        o[base_ + k * innerSize] = acc;
    }
}";

    public static string CumMaxAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = -1.0e38;
    for (int k = 0; k < axisSize; ++k) {
        float v = a[base_ + k * innerSize];
        if (v > acc) acc = v;
        o[base_ + k * innerSize] = acc;
    }
}";

    public static string CumMinAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    float acc = 1.0e38;
    for (int k = 0; k < axisSize; ++k) {
        float v = a[base_ + k * innerSize];
        if (v < acc) acc = v;
        o[base_ + k * innerSize] = acc;
    }
}";

    public static string LogCumSumExpAxis => Header + TwoBuf + @"
layout(push_constant) uniform P { int outerSize; int axisSize; int innerSize; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int outer = gid / innerSize;
    int base_ = outer * axisSize * innerSize + inner;
    if (axisSize <= 0) return;
    // Bootstrap from the first element so a leading -Inf doesn't produce NaN.
    float m = a[base_];
    float s = 1.0;
    o[base_] = m;
    for (int k = 1; k < axisSize; ++k) {
        float x = a[base_ + k * innerSize];
        if (x > m) { s = s * exp(m - x) + 1.0; m = x; }
        else { s += exp(x - m); }
        o[base_ + k * innerSize] = m + log(s);
    }
}";

    // =====================================================================
    // Indexing
    // =====================================================================

    public static string TakeLinear => Header + IdxTake + @"
layout(push_constant) uniform P { int outSize; int inputLinearLen; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= outSize) return;
    int pos = idx[gid];
    o[gid] = (pos >= 0 && pos < inputLinearLen) ? src[pos] : 0.0;
}";

    public static string TakeAlongDim => Header + IdxTake + @"
layout(push_constant) uniform P { int outerSize; int idxAxis; int innerSize; int srcAxis; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * idxAxis * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxAxis;
    int outer = tmp / idxAxis;
    int target = idx[gid];
    int srcIdx = (outer * srcAxis + target) * innerSize + inner;
    o[gid] = (target >= 0 && target < srcAxis) ? src[srcIdx] : 0.0;
}";

    // IndexAdd requires atomic float add. Vulkan 1.2+ / VK_EXT_shader_atomic_float.
    public static string IndexAdd => @"#version 450
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer Out { float o[]; };
layout(set = 0, binding = 1) readonly buffer Idx { int idx[]; };
layout(set = 0, binding = 2) readonly buffer Src { float src[]; };
layout(push_constant) uniform P { int outerSize; int dstAxis; int innerSize; int idxLen; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = idx[i];
    if (target < 0 || target >= dstAxis) return;
    int dstPos = (outer * dstAxis + target) * innerSize + inner;
    int srcPos = (outer * idxLen + i) * innerSize + inner;
    atomicAdd(o[dstPos], src[srcPos]);
}
";

    public static string IndexCopy => Header + IdxScatterReadWrite + @"
layout(push_constant) uniform P { int outerSize; int dstAxis; int innerSize; int idxLen; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = idx[i];
    if (target < 0 || target >= dstAxis) return;
    o[(outer * dstAxis + target) * innerSize + inner] =
        src[(outer * idxLen + i) * innerSize + inner];
}";

    public static string IndexFill => Header + @"
layout(set = 0, binding = 0) buffer Out { float o[]; };
layout(set = 0, binding = 1) readonly buffer Idx { int idx[]; };
layout(push_constant) uniform P { int outerSize; int dstAxis; int innerSize; int idxLen; float fillValue; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = outerSize * idxLen * innerSize;
    if (gid >= total) return;
    int inner = gid % innerSize;
    int tmp = gid / innerSize;
    int i = tmp % idxLen;
    int outer = tmp / idxLen;
    int target = idx[i];
    if (target < 0 || target >= dstAxis) return;
    o[(outer * dstAxis + target) * innerSize + inner] = fillValue;
}";

    public static string MaskedScatter => Header + @"
layout(set = 0, binding = 0) buffer Out { float o[]; };
layout(set = 0, binding = 1) readonly buffer Mask { int mask[]; };     // 0/1 per element
layout(set = 0, binding = 2) readonly buffer Prefix { int prefix[]; };
layout(set = 0, binding = 3) readonly buffer Src { float src[]; };
layout(push_constant) uniform P { int total; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= total) return;
    if (mask[gid] != 0) o[gid] = src[prefix[gid]];
}";

    // =====================================================================
    // Element-wise binary special
    // =====================================================================

    public static string Hypot => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float av = a[gid]; float bv = b[gid];
    o[gid] = sqrt(av*av + bv*bv);
}";

    public static string Copysign => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float mag = abs(a[gid]);
    o[gid] = (b[gid] < 0.0) ? -mag : mag;
}";

    public static string Fmod => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float bv = b[gid];
    if (bv == 0.0) { o[gid] = 0.0; return; }
    float av = a[gid];
    float q = trunc(av / bv);
    o[gid] = av - q * bv;
}";

    public static string Remainder => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float av = a[gid]; float bv = b[gid];
    if (bv == 0.0) { o[gid] = 0.0; return; }
    float q = floor(av / bv);
    o[gid] = av - q * bv;
}";

    public static string FloatPower => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = pow(a[gid], b[gid]);
}";

    public static string LogAddExp => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float av = a[gid]; float bv = b[gid];
    float m = max(av, bv);
    float s = min(av, bv);
    o[gid] = m + log(1.0 + exp(s - m));
}";

    public static string LogAddExp2 => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float av = a[gid]; float bv = b[gid];
    float m = max(av, bv);
    float s = min(av, bv);
    o[gid] = m + log2(1.0 + exp2(s - m));
}";

    public static string Xlogy => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float xv = a[gid];
    o[gid] = (xv == 0.0) ? 0.0 : xv * log(b[gid]);
}";

    public static string Xlog1py => Header + ThreeBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float xv = a[gid];
    o[gid] = (xv == 0.0) ? 0.0 : xv * log(1.0 + b[gid]);
}";

    // =====================================================================
    // Element-wise unary special
    // =====================================================================

    // GLSL lacks erfc/erfinv/i0/i1/lgamma/digamma — we implement them in the
    // shader body directly using polynomial approximations. Only exp / log /
    // sqrt / abs / pow are standard.

    private const string I0I1Helpers = @"
float p210_i0(float x) {
    float ax = abs(x);
    if (ax < 3.75) {
        float y = (x / 3.75); y = y * y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    } else {
        float y = 3.75 / ax;
        float ans = 0.39894228 + y * (0.01328592 + y * (0.00225319
                + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        return (exp(ax) / sqrt(ax)) * ans;
    }
}
float p210_i1(float x) {
    float ax = abs(x);
    float ans;
    if (ax < 3.75) {
        float y = (x / 3.75); y = y * y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
    } else {
        float y = 3.75 / ax;
        ans = 0.39894228 + y * (-0.03988024 + y * (-0.00362018
                + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967
                + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059)))))));
        ans *= (exp(ax) / sqrt(ax));
    }
    return (x < 0.0) ? -ans : ans;
}
// erf using Abramowitz-Stegun 7.1.26 (6-digit): accurate to ~1.5e-7.
float p210_erf(float x) {
    float sign_ = (x < 0.0) ? -1.0 : 1.0;
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float y = 1.0 - (((((
            1.061405429 * t - 1.453152027) * t
            + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t) * exp(-ax*ax);
    return sign_ * y;
}
";

    public static string Erfc => Header + I0I1Helpers + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = 1.0 - p210_erf(a[gid]);
}";

    public static string Erfinv => Header + I0I1Helpers + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float y = a[gid];
    if (y >= 1.0) { o[gid] = 1.0e38; return; }
    if (y <= -1.0) { o[gid] = -1.0e38; return; }
    float ln = log(1.0 - y * y);
    float aC = 0.147;
    float t = 2.0 / (3.14159265358979 * aC) + ln * 0.5;
    float xs = sign(y) * sqrt(sqrt(t * t - ln / aC) - t);
    for (int k = 0; k < 2; ++k) {
        float e = p210_erf(xs);
        float df = 2.0 / sqrt(3.14159265358979) * exp(-xs * xs);
        xs -= (e - y) / df;
    }
    o[gid] = xs;
}";

    // lgamma via Lanczos approximation (single-precision, 8-term).
    private const string LgammaHelper = @"
float p210_lgamma(float x) {
    // Reflection for x < 0.5
    if (x < 0.5) {
        return log(3.14159265358979 / abs(sin(3.14159265358979 * x))) - p210_lgamma(1.0 - x);
    }
    float g = 7.0;
    float t = x - 1.0;
    float hi = t + g + 0.5;
    float sum = 0.99999999999980993;
    float lc[8];
    lc[0] = 676.5203681218851;
    lc[1] = -1259.1392167224028;
    lc[2] = 771.32342877765313;
    lc[3] = -176.61502916214059;
    lc[4] = 12.507343278686905;
    lc[5] = -0.13857109526572012;
    lc[6] = 9.9843695780195716e-6;
    lc[7] = 1.5056327351493116e-7;
    for (int i = 0; i < 8; ++i) sum += lc[i] / (t + float(i + 1));
    return 0.5 * log(2.0 * 3.14159265358979) + (t + 0.5) * log(hi) - hi + log(sum);
}
";

    public static string LgammaApprox => Header + LgammaHelper + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = p210_lgamma(a[gid]);
}";

    public static string Digamma => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = a[gid];
    float result = 0.0;
    for (int k = 0; k < 64 && x < 6.0; ++k) { result -= 1.0 / x; x += 1.0; }
    float inv = 1.0 / x;
    float inv2 = inv * inv;
    result += log(x) - 0.5 * inv
            - inv2 * ((1.0/12.0)
              - inv2 * ((1.0/120.0)
                - inv2 * (1.0/252.0)));
    o[gid] = result;
}";

    public static string I0 => Header + I0I1Helpers + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = p210_i0(a[gid]);
}";

    public static string I1 => Header + I0I1Helpers + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = p210_i1(a[gid]);
}";

    // Overflow-safe form — see the CUDA/Metal counterparts for full commentary.
    public static string I0e => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = a[gid];
    float ax = abs(x);
    float ans;
    if (ax < 3.75) {
        float y = (x / 3.75); y = y * y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
        ans = exp(-ax) * ans;
    } else {
        float y = 3.75 / ax;
        ans = 0.39894228 + y * (0.01328592 + y * (0.00225319
                + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
                + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))));
        ans = ans / sqrt(ax);
    }
    o[gid] = ans;
}";

    public static string I1e => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = a[gid];
    float ax = abs(x);
    float ans;
    if (ax < 3.75) {
        float y = (x / 3.75); y = y * y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
        ans = exp(-ax) * ans;
    } else {
        float y = 3.75 / ax;
        ans = 0.39894228 + y * (-0.03988024 + y * (-0.00362018
                + y * (0.00163801 + y * (-0.01031555 + y * (0.02282967
                + y * (-0.02895312 + y * (0.01787654 + y * -0.00420059)))))));
        ans = ans / sqrt(ax);
    }
    o[gid] = (x < 0.0) ? -ans : ans;
}";

    // =====================================================================
    // Predicates + numeric hygiene
    // =====================================================================

    public static string IsFinite => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = a[gid];
    // GLSL isinf returns bool; we also guard NaN.
    bool f = !isinf(x) && !isnan(x);
    o[gid] = f ? 1.0 : 0.0;
}";

    public static string IsNan => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = isnan(a[gid]) ? 1.0 : 0.0;
}";

    public static string IsInf => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    o[gid] = isinf(a[gid]) ? 1.0 : 0.0;
}";

    public static string NanToNum => Header + TwoBuf + @"
layout(push_constant) uniform P { int size; float nanVal; float posInfVal; float negInfVal; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = a[gid];
    if (isnan(x)) o[gid] = nanVal;
    else if (isinf(x)) o[gid] = (x > 0.0) ? posInfVal : negInfVal;
    else o[gid] = x;
}";

    // =====================================================================
    // Pairwise
    // =====================================================================

    public static string CosineSimilarityLast => Header + ThreeBuf + @"
layout(push_constant) uniform P { int n; int d; float eps; };
void main() {
    int row = int(gl_GlobalInvocationID.x);
    if (row >= n) return;
    float dot = 0.0; float na = 0.0; float nb = 0.0;
    int base_ = row * d;
    for (int k = 0; k < d; ++k) {
        float av = a[base_ + k]; float bv = b[base_ + k];
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    float denom = max(sqrt(na * nb), eps);
    o[row] = dot / denom;
}";

    public static string CdistL2 => Header + ThreeBuf + @"
layout(push_constant) uniform P { int n; int m; int d; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int total = n * m;
    if (gid >= total) return;
    int j = gid % m;
    int i = gid / m;
    float acc = 0.0;
    for (int k = 0; k < d; ++k) {
        float v = a[i * d + k] - b[j * d + k];
        acc += v * v;
    }
    o[gid] = sqrt(acc);
}";

    // =====================================================================
    // Clamp
    // =====================================================================

    public static string ClampMinMax => Header + @"
layout(set = 0, binding = 0) readonly buffer Src { float s[]; };
layout(set = 0, binding = 1) readonly buffer Lo { float lo[]; };
layout(set = 0, binding = 2) readonly buffer Hi { float hi[]; };
layout(set = 0, binding = 3) writeonly buffer Out { float o[]; };
layout(push_constant) uniform P { int size; int hasLo; int hasHi; };
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= size) return;
    float x = s[gid];
    if (hasLo != 0) { float l = lo[gid]; if (x < l) x = l; }
    if (hasHi != 0) { float h = hi[gid]; if (x > h) x = h; }
    o[gid] = x;
}";
}
