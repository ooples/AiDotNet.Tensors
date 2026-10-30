// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL ports of <see cref="BatchExecutionCudaKernels"/> for <see cref="IGpuBatchExecution"/> on Vulkan (#1142).
/// Same kernels, same mode numbering and the same counter hash, so every backend produces the same values.
/// Buffers bind in declaration order; push constants are the kernel's scalars in order, 4 bytes each, with a
/// 64-bit seed split into its low and high words.
/// </summary>
internal static class VulkanBatchExecutionKernels
{
    private const string Common = @"
uint bx_hash(uint counter, uint key) {
    uint hi; uint lo;
    umulExtended(counter, 0xCD9E8D57u, hi, lo);
    counter = hi ^ key ^ lo;
    umulExtended(counter, 0xCD9E8D57u, hi, lo);
    return hi ^ (key + 1u) ^ lo;
}
float bx_inf() { return uintBitsToFloat(0x7f800000u); }
float bx_unit(uint h) { return float(h >> 8) * (1.0 / 16777216.0); }
int bx_gid() { return int(gl_GlobalInvocationID.x); }
";

    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(2);

    // GLSL keywords and reserved words a C parameter name can collide with. Declared before Solvers, whose initialiser
    // reads it.
    private static readonly HashSet<string> GlslReservedWords = new(StringComparer.Ordinal)
    {
        "input", "output", "layout", "filter", "sample", "common", "partition", "active", "buffer", "shared", "coherent",
        "patch", "precise", "smooth", "flat", "uniform", "varying", "attribute", "in", "out", "inout", "texture", "image",
        "resource", "interface", "half", "fixed", "cast", "union", "enum", "class", "default", "readonly", "writeonly",
        "restrict", "volatile", "lowp", "mediump", "highp", "precision", "invariant", "discard", "subroutine", "centroid",
        "noperspective", "superp", "namespace", "using", "template", "this", "goto", "inline", "noinline", "public",
        "static", "extern", "external", "long", "short", "sizeof", "typedef", "asm", "unsigned",
    };

    /// <summary>
    /// GLSL for the IExtendedLinalgBackend solvers, translated mechanically from the shared CUDA C (#1142) so Vulkan
    /// runs the same code as HIP, OpenCL and Metal: each pointer parameter becomes a storage buffer of the same name
    /// (read-only when const), each scalar a push constant in order, and the float builtins their GLSL names.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> Solvers = TranslateSolvers(
        BatchExecutionCudaKernels.SolverSource + BatchExecutionCudaKernels.AttentionSource + BatchExecutionCudaKernels.LstmSource + BatchExecutionCudaKernels.GruSource
        + BatchExecutionCudaKernels.DecodeLinearSource + BatchExecutionCudaKernels.DirectConvSource);

    private static Dictionary<string, string> TranslateSolvers(string cuda)
    {
        // The math names map on the whole source, so the #define bodies copied verbatim below are translated too.
        cuda = Regex.Replace(cuda, @"\b(sqrt|fabs|fmax|exp|log|tanh)f\(",
            m => m.Groups[1].Value == "fabs" ? "abs(" : m.Groups[1].Value == "fmax" ? "max(" : m.Groups[1].Value + "(",
            RegexOptions.None, RegexTimeout);
        cuda = cuda.Replace("__int_as_float(", "intBitsToFloat(");
        // A C cast of a name, (float)D, is float(D) in GLSL.
        cuda = Regex.Replace(cuda, @"\((float|int)\)\s*(\w+)", "$1($2)", RegexOptions.None, RegexTimeout);
        var defines = new StringBuilder();
        foreach (Match d in Regex.Matches(cuda, @"^#define [^\n]*\n", RegexOptions.Multiline, RegexTimeout)) defines.Append(d.Value);
        var kernels = Regex.Matches(cuda, @"extern ""C"" __global__ __launch_bounds__\(256\) void (\w+)\(([^)]*)\) \{",
            RegexOptions.None, RegexTimeout);
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int k = 0; k < kernels.Count; k++)
        {
            Match kernel = kernels[k];
            int end = k + 1 < kernels.Count ? kernels[k + 1].Index : cuda.Length;
            string body = cuda.Substring(kernel.Index + kernel.Length, end - kernel.Index - kernel.Length)
                .Replace("__uint_as_float(", "uintBitsToFloat(");
            body = Regex.Replace(body, @"\b(sqrt|fabs|fmax|exp|log|tanh)f\(",
                m => m.Groups[1].Value == "fabs" ? "abs(" : m.Groups[1].Value == "fmax" ? "max(" : m.Groups[1].Value + "(",
                RegexOptions.None, RegexTimeout);
            // A parameter named with a GLSL keyword (input, output, layout) is renamed in the declaration and the body.
            string parameters = kernel.Groups[2].Value;
            foreach (Match name in Regex.Matches(parameters, @"(\w+)\s*(?:,|$)", RegexOptions.None, RegexTimeout))
            {
                string identifier = name.Groups[1].Value;
                if (!GlslReservedWords.Contains(identifier)) continue;
                string pattern = @"\b" + identifier + @"\b";
                parameters = Regex.Replace(parameters, pattern, identifier + "_", RegexOptions.None, RegexTimeout);
                body = Regex.Replace(body, pattern, identifier + "_", RegexOptions.None, RegexTimeout);
            }
            var buffers = new List<string>();
            var push = new StringBuilder();
            foreach (string raw in parameters.Split(','))
            {
                string p = Regex.Replace(raw.Trim(), @"\s+", " ", RegexOptions.None, RegexTimeout);
                Match pointer = Regex.Match(p, @"^(const )?(float|int)\* (\w+)$", RegexOptions.None, RegexTimeout);
                // A buffer a solver writes is declared volatile. These kernels factor in place: they write an element
                // and read it back through a different index expression later in the same invocation (a pivot swap,
                // then the next column's pivot search). NVIDIA's Vulkan compiler reused a value read before the swap:
                // la_ldl_factor compared the next pivot candidate against the previous pivot and chose a different
                // pivot from every other backend (RTX 3080; the same source is right on CUDA and OpenCL, and with
                // shaderc's optimizer off the result was the same, so it is the driver). volatile makes every read
                // go to memory.
                if (pointer.Success)
                    buffers.Add((pointer.Groups[1].Success ? "r:" : "rwv:") + pointer.Groups[3].Value + ":" + pointer.Groups[2].Value);
                else if (Regex.IsMatch(p, @"^int \w+$", RegexOptions.None, RegexTimeout))
                    push.Append(p).Append("; ");
                else
                    throw new InvalidOperationException($"Untranslatable kernel parameter '{p}' in {kernel.Groups[1].Value}.");
            }
            result[kernel.Groups[1].Value] = Shader(buffers.ToArray(), push.ToString().TrimEnd(),
                defines + "void main() {" + body);
        }
        return result;
    }
    /// <summary>
    /// Buffer specs are "r:name", "w:name", "rw:name" or "rwv:name" (read-write and volatile), bound to bindings
    /// 0..n-1 in order.
    /// </summary>
    private static string Shader(string[] buffers, string push, string body)
    {
        var sb = new StringBuilder("#version 450\nlayout(local_size_x = 256) in;\n");
        for (int i = 0; i < buffers.Length; i++)
        {
            string[] parts = buffers[i].Split(':');
            string access = parts[0] == "r" ? "readonly " : parts[0] == "w" ? "writeonly " : parts[0] == "rwv" ? "volatile " : "";
            sb.Append("layout(set = 0, binding = ").Append(i).Append(") ").Append(access)
              .Append("buffer Buf").Append(i).Append(" { ").Append(parts.Length > 2 ? parts[2] : "float").Append(" ")
              .Append(parts[1]).Append("[]; };\n");
        }
        sb.Append("layout(push_constant) uniform Params { ").Append(push).Append(" };\n");
        sb.Append(Common).Append(body);
        return sb.ToString();
    }

    public static readonly string FullReduce = Shader(new[] { "r:inp", "w:outp" }, "int size; int mode; float maxVal;", @"
shared float scratch[256];
void main() {
    int lid = int(gl_LocalInvocationID.x);
    float acc = mode == 2 ? 1.0 : (mode == 4 ? bx_inf() : 0.0);
    for (int i = lid; i < size; i += 256) {
        float v = inp[i];
        if (mode <= 1) acc += v * v;
        else if (mode == 2) acc *= v;
        else if (mode == 3) acc = max(acc, abs(v));
        else if (mode == 4) acc = min(acc, abs(v));
        else acc += exp(v - maxVal);
    }
    scratch[lid] = acc;
    barrier();
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            float a = scratch[lid]; float b = scratch[lid + s];
            scratch[lid] = mode == 2 ? a * b : (mode == 3 ? max(a, b) : (mode == 4 ? min(a, b) : a + b));
        }
        barrier();
    }
    if (lid == 0) {
        float r = scratch[0];
        outp[0] = mode == 1 ? sqrt(r) : (mode == 5 ? maxVal + log(r) : r);
    }
}");

    public static readonly string RowReduce = Shader(new[] { "r:inp", "w:outp" }, "int outerSize; int innerSize; int mode;", @"
void main() {
    int o = bx_gid(); if (o >= outerSize) return;
    int base = o * innerSize;
    float r;
    if (mode == 2) { r = 1.0; for (int j = 0; j < innerSize; j++) r *= inp[base + j]; }
    else if (mode == 3) { r = 0.0; for (int j = 0; j < innerSize; j++) r += inp[base + j] * inp[base + j]; r = sqrt(r); }
    else if (mode == 4) {
        float m = -bx_inf(); for (int j = 0; j < innerSize; j++) m = max(m, inp[base + j]);
        float s = 0.0; for (int j = 0; j < innerSize; j++) s += exp(inp[base + j] - m);
        r = m + log(s);
    } else {
        float sum = 0.0; for (int j = 0; j < innerSize; j++) sum += inp[base + j];
        float mean = sum / float(innerSize);
        float sq = 0.0; for (int j = 0; j < innerSize; j++) { float d = inp[base + j] - mean; sq += d * d; }
        float v = sq / float(innerSize);
        r = mode == 0 ? v : (mode == 1 ? sqrt(v) : log(v + 1e-8));
    }
    outp[o] = r;
}");

    public static readonly string RowMap = Shader(new[] { "r:inp", "rw:outp" }, "int outerSize; int innerSize; int mode;", @"
void main() {
    int o = bx_gid(); if (o >= outerSize) return;
    int base = o * innerSize;
    if (mode == 0) {
        float acc = 0.0; for (int j = 0; j < innerSize; j++) { acc += inp[base + j]; outp[base + j] = acc; }
    } else if (mode == 1) {
        float sq = 0.0; for (int j = 0; j < innerSize; j++) sq += inp[base + j] * inp[base + j];
        float inv = 1.0 / (sqrt(sq) + 1e-12);
        for (int j = 0; j < innerSize; j++) outp[base + j] = inp[base + j] * inv;
    } else if (mode == 2) {
        float m = -bx_inf(); for (int j = 0; j < innerSize; j++) m = max(m, inp[base + j]);
        float s = 0.0; for (int j = 0; j < innerSize; j++) s += exp(inp[base + j] - m);
        float l = log(s);
        for (int j = 0; j < innerSize; j++) outp[base + j] = inp[base + j] - m - l;
    } else if (mode == 3) {
        float s = 0.0;
        for (int j = 0; j < innerSize; j++) { float x = inp[base + j]; float t = max(1.0 + x + 0.5 * x * x, 1e-10); outp[base + j] = t; s += t; }
        float inv = 1.0 / s;
        for (int j = 0; j < innerSize; j++) outp[base + j] *= inv;
    } else if (mode == 4) {
        float sq = 0.0; for (int j = 0; j < innerSize; j++) sq += inp[base + j] * inp[base + j];
        float n = sqrt(sq + 1e-12);
        float m = -bx_inf(); for (int j = 0; j < innerSize; j++) m = max(m, inp[base + j] / n);
        float s = 0.0; for (int j = 0; j < innerSize; j++) { float e = exp(inp[base + j] / n - m); outp[base + j] = e; s += e; }
        float inv = 1.0 / (s + 1e-10);
        for (int j = 0; j < innerSize; j++) outp[base + j] *= inv;
    } else {
        float supportSum = 0.0; int supportCount = 0;
        for (int j = 0; j < innerSize; j++) {
            float z = inp[base + j]; int k = 0; float s = 0.0;
            for (int i = 0; i < innerSize; i++) { float v = inp[base + i]; if (v >= z) { k++; s += v; } }
            if (1.0 + float(k) * z > s) { supportSum += z; supportCount++; }
        }
        float tau = (supportSum - 1.0) / float(supportCount);
        for (int j = 0; j < innerSize; j++) outp[base + j] = max(inp[base + j] - tau, 0.0);
    }
}");

    public static readonly string Scalar = Shader(new[] { "r:inp", "w:outp" }, "float scalar; int size; int mode;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    outp[i] = mode == 0 ? scalar - inp[i] : inp[i] / scalar;
}");

    public static readonly string Unary = Shader(new[] { "r:inp", "w:outp" }, "int size; int mode;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    float v = inp[i];
    outp[i] = mode == 0 ? inversesqrt(v + 1e-12) : (mode == 1 ? (v > 0.0 ? 1.0 : 0.0) : (mode == 2 ? v * (1.0 - v) : 1.0 - v * v));
}");

    public static readonly string SinCos = Shader(new[] { "r:inp", "w:sinOut", "w:cosOut" }, "int size;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    sinOut[i] = sin(inp[i]);
    cosOut[i] = cos(inp[i]);
}");

    public static readonly string ReduceBackward = Shader(new[] { "r:gradOut", "r:inp", "r:stat", "r:stat2", "w:gradIn" },
        "int outerSize; int reduceSize; int mode;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * reduceSize) return;
    int o = i / reduceSize;
    float g = gradOut[o];
    if (mode == 0) gradIn[i] = g;
    else if (mode == 1) gradIn[i] = inp[i] == stat[o] ? g : 0.0;
    else if (mode == 2) gradIn[i] = g * 2.0 * (inp[i] - stat[o]) / float(reduceSize);
    else gradIn[i] = g / (stat2[o] + 1e-8) * 2.0 * (inp[i] - stat[o]) / float(reduceSize);
}");

    public static readonly string BroadcastFirst = Shader(new[] { "r:a", "r:b", "w:outp" }, "int outerSize; int innerSize; int mode;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * innerSize) return;
    float x = a[i / innerSize];
    outp[i] = mode == 0 ? x + b[i] : x * b[i];
}");

    public static readonly string GatedBackward = Shader(new[] { "r:gradOut", "r:inp", "w:gradIn" }, "int outerSize; int halfDim; int mode;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * halfDim) return;
    int o = i / halfDim; int d = i % halfDim; int full = halfDim * 2;
    float v = inp[o * full + d]; float g = inp[o * full + halfDim + d]; float go = gradOut[i];
    float act; float dact;
    if (mode == 0) { float s = 1.0 / (1.0 + exp(-g)); act = s; dact = s * (1.0 - s); }
    else if (mode == 1) {
        float t = tanh(0.7978845608 * (g + 0.044715 * g * g * g));
        act = 0.5 * g * (1.0 + t);
        dact = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * g * g);
    }
    else if (mode == 2) { act = max(g, 0.0); dact = g > 0.0 ? 1.0 : 0.0; }
    else { float s = 1.0 / (1.0 + exp(-g)); act = g * s; dact = s + g * s * (1.0 - s); }
    gradIn[o * full + d] = go * act;
    gradIn[o * full + halfDim + d] = go * v * dact;
}");

    public static readonly string ConcatAxis = Shader(new[] { "r:a", "r:b", "w:outp" }, "int outerSize; int aInner; int bInner;", @"
void main() {
    int total = aInner + bInner;
    int i = bx_gid(); if (i >= outerSize * total) return;
    int o = i / total; int j = i % total;
    outp[i] = j < aInner ? a[o * aInner + j] : b[o * bInner + j - aInner];
}");

    public static readonly string SliceLastAxis = Shader(new[] { "r:inp", "w:outp" }, "int outerSize; int inputInner; int start; int sliceSize;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * sliceSize) return;
    outp[i] = inp[(i / sliceSize) * inputInner + start + i % sliceSize];
}");

    public static readonly string SetSliceLastAxis = Shader(new[] { "rw:outp", "r:vals" }, "int outerSize; int outputInner; int start; int sliceSize;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * sliceSize) return;
    outp[(i / sliceSize) * outputInner + start + i % sliceSize] = vals[i];
}");

    public static readonly string SliceAxis = Shader(new[] { "r:inp", "w:outp" }, "int outerSize; int axisSize; int stride; int index;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * stride) return;
    outp[i] = inp[((i / stride) * axisSize + index) * stride + i % stride];
}");

    public static readonly string SetSliceAxis = Shader(new[] { "rw:outp", "r:vals" }, "int outerSize; int axisSize; int stride; int index;", @"
void main() {
    int i = bx_gid(); if (i >= outerSize * stride) return;
    outp[((i / stride) * axisSize + index) * stride + i % stride] = vals[i];
}");

    public static readonly string Stack2 = Shader(new[] { "r:a", "r:b", "w:outp" }, "int size;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    outp[2 * i] = a[i];
    outp[2 * i + 1] = b[i];
}");

    public static readonly string Window2D = Shader(new[] { "r:inp", "w:outp" },
        "int batch; int channels; int inH; int inW; int outH; int outW; int h0; int w0; float padValue; int mode;", @"
void main() {
    int i = bx_gid();
    if (mode == 0) {
        if (i >= batch * channels * outH * outW) return;
        int w = i % outW - w0; int h = i / outW % outH - h0; int bc = i / (outW * outH);
        outp[i] = (h >= 0 && h < inH && w >= 0 && w < inW) ? inp[(bc * inH + h) * inW + w] : padValue;
    } else if (mode == 1) {
        if (i >= batch * channels * inH * inW) return;
        int w = i % inW; int h = i / inW % inH; int bc = i / (inW * inH);
        outp[i] = inp[(bc * outH + h + h0) * outW + w + w0];
    } else if (mode == 2) {
        if (i >= batch * channels * outH * outW) return;
        int w = i % outW; int h = i / outW % outH; int bc = i / (outW * outH);
        outp[i] = inp[(bc * inH + h + h0) * inW + w + w0];
    } else {
        if (i >= batch * channels * inH * inW) return;
        int w = i % inW - w0; int h = i / inW % inH - h0; int bc = i / (inW * inH);
        outp[i] = (h >= 0 && h < outH && w >= 0 && w < outW) ? inp[(bc * outH + h) * outW + w] : 0.0;
    }
}");

    public static readonly string TileLastAxis = Shader(new[] { "r:inp", "w:outp" }, "int outerSize; int innerSize; int repeats;", @"
void main() {
    int tiled = innerSize * repeats;
    int i = bx_gid(); if (i >= outerSize * tiled) return;
    outp[i] = inp[(i / tiled) * innerSize + (i % tiled) % innerSize];
}");

    public static readonly string RepeatElements = Shader(new[] { "r:inp", "w:outp" }, "int outerSize; int innerSize; int repeats;", @"
void main() {
    int tiled = innerSize * repeats;
    int i = bx_gid(); if (i >= outerSize * tiled) return;
    outp[i] = inp[(i / tiled) * innerSize + (i % tiled) / repeats];
}");

    public static readonly string Linspace = Shader(new[] { "w:outp" }, "float start; float step; int size;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    outp[i] = start + step * float(i);
}");

    public static readonly string Diag = Shader(new[] { "r:inp", "w:outp" }, "int n;", @"
void main() {
    int i = bx_gid(); if (i >= n * n) return;
    outp[i] = (i / n == i % n) ? inp[i / n] : 0.0;
}");

    public static readonly string ExtractDiag = Shader(new[] { "r:inp", "w:outp" }, "int n; int cols;", @"
void main() {
    int i = bx_gid(); if (i >= n || i >= cols) return;
    outp[i] = inp[i * cols + i];
}");

    public static readonly string TriangularMask = Shader(new[] { "w:outp" }, "int rows; int cols; int diagonal; float maskValue;", @"
void main() {
    int i = bx_gid(); if (i >= rows * cols) return;
    outp[i] = (i % cols > i / cols + diagonal) ? maskValue : 0.0;
}");

    public static readonly string IndexSelect = Shader(new[] { "r:inp", "r:indices", "w:outp" }, "int numIndices; int innerSize;", @"
void main() {
    int i = bx_gid(); if (i >= numIndices * innerSize) return;
    outp[i] = inp[int(indices[i / innerSize]) * innerSize + i % innerSize];
}");

    public static readonly string RowLoss = Shader(new[] { "r:pred", "r:targ", "w:loss" }, "int batchSize; int numClasses; int mode;", @"
void main() {
    int b = bx_gid(); if (b >= batchSize) return;
    int base = b * numClasses;
    float r = 0.0;
    if (mode == 0) {
        float m = -bx_inf(); for (int c = 0; c < numClasses; c++) m = max(m, pred[base + c]);
        float s = 0.0; for (int c = 0; c < numClasses; c++) s += exp(pred[base + c] - m);
        float lse = m + log(s);
        for (int c = 0; c < numClasses; c++) if (targ[base + c] > 0.0) r -= targ[base + c] * (pred[base + c] - lse);
    } else {
        for (int c = 0; c < numClasses; c++) { float d = pred[base + c] - targ[base + c]; r += d * d; }
        r /= float(numClasses);
    }
    loss[b] = r;
}");

    public static readonly string DropoutMask = Shader(new[] { "w:mask" }, "int size; float keepProb; uint seedLo; uint seedHi;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    float u = bx_unit(bx_hash(uint(i), seedLo ^ seedHi));
    float invKeep = keepProb > 1e-7 ? 1.0 / keepProb : 0.0;
    mask[i] = u < keepProb ? invKeep : 0.0;
}");

    public static readonly string GaussianNoise = Shader(new[] { "w:outp" }, "int size; float mean; float stdDev; uint seedLo; uint seedHi;", @"
void main() {
    int i = bx_gid(); if (i >= size) return;
    uint key = seedLo ^ seedHi;
    float u1 = bx_unit(bx_hash(uint(2 * i), key)) + (0.5 / 16777216.0);
    float u2 = bx_unit(bx_hash(uint(2 * i + 1), key + 2u));
    outp[i] = mean + stdDev * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}");

    public static readonly string GumbelSoftmax = Shader(new[] { "r:logits", "rw:outp" },
        "int outerSize; int innerSize; float temperature; uint seedLo; uint seedHi;", @"
void main() {
    int o = bx_gid(); if (o >= outerSize) return;
    uint key = seedLo ^ seedHi;
    float t = max(temperature, 1e-7);
    int base = o * innerSize;
    float m = -bx_inf();
    for (int j = 0; j < innerSize; j++) {
        float u = bx_unit(bx_hash(uint(base + j), key)) + (0.5 / 16777216.0);
        float v = (logits[base + j] - log(-log(u))) / t;
        outp[base + j] = v; m = max(m, v);
    }
    float s = 0.0;
    for (int j = 0; j < innerSize; j++) { float e = exp(outp[base + j] - m); outp[base + j] = e; s += e; }
    float inv = 1.0 / (s + 1e-10);
    for (int j = 0; j < innerSize; j++) outp[base + j] *= inv;
}");

    public static readonly string BatchOuterProduct = Shader(new[] { "r:a", "r:b", "w:outp" }, "int batchSize; int M; int N;", @"
void main() {
    int i = bx_gid(); if (i >= batchSize * M * N) return;
    int mn = M * N; int bt = i / mn; int r = i % mn;
    outp[i] = a[bt * M + r / N] * b[bt * N + r % N];
}");

    public static readonly string Meshgrid2D = Shader(new[] { "r:source0", "r:source1", "w:output0", "w:output1" }, "int n0; int n1; int xy;", @"
void main() {
    int i = bx_gid(); if (i >= n0 * n1) return;
    int cols = xy != 0 ? n0 : n1; int r = i / cols; int c = i % cols;
    output0[i] = source0[xy != 0 ? c : r];
    output1[i] = source1[xy != 0 ? r : c];
}");
}
