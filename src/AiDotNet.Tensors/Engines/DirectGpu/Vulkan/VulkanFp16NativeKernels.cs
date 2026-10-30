// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Vulkan GLSL FP16-NATIVE op kernels (Tensors #558): GELU / ReLU / residual-add over half-stored
// activations. Each kernel reads its activation DIRECTLY from a packed-half buffer (two halves per
// 32-bit word — the layout VulkanBackend.ConvertToFp16 produces), up-casts to FP32 in-register with the
// core unpackHalf2x16 built-in for the math, and writes a packed-half result with packHalf2x16 — no
// separate FP32 buffer, so there is no convert transient (the failure mode that made buffer-compression
// RAISE peak VRAM). Compiled to SPIR-V at runtime via libshaderc.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// FP16-native elementwise / activation compute shaders for the Vulkan backend. One invocation processes
/// one 32-bit word = two half elements (<c>unpackHalf2x16</c> → FP32 math → <c>packHalf2x16</c>); the
/// backend dispatches <c>ceil(N/2)</c> threads. Push constant carries the element count N.
/// </summary>
internal static class VulkanFp16NativeKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    // GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))), tanh arg clamped to +/-20 (overflow guard).
    private const string GeluFn = @"
float op(float x) {
    float z = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    z = clamp(z, -20.0, 20.0);
    return 0.5 * x * (1.0 + tanh(z));
}
";

    private const string ReluFn = @"
float op(float x) { return max(x, 0.0); }
";

    private static string UnaryBody(string fn) => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) writeonly buffer Ob { uint Outp[]; };
layout(push_constant) uniform PC { uint N; };
" + fn + @"
void main() {
    uint w = gl_GlobalInvocationID.x;
    uint numWords = (N + 1u) >> 1;
    if (w >= numWords) return;
    vec2 v = unpackHalf2x16(Inp[w]);
    Outp[w] = packHalf2x16(vec2(op(v.x), op(v.y)));
}";

    /// <summary>FP16-native GELU: packed-half in/out, FP32 math.</summary>
    public static string Gelu => UnaryBody(GeluFn);

    /// <summary>FP16-native ReLU: packed-half in/out, FP32 math.</summary>
    public static string Relu => UnaryBody(ReluFn);

    /// <summary>FP16-native residual add: out = a + b (same shape), packed-half in/out, FP32 accumulate.</summary>
    public static string Add => Header + @"
layout(set=0,binding=0) readonly buffer Ab { uint A[]; };
layout(set=0,binding=1) readonly buffer Bb { uint B[]; };
layout(set=0,binding=2) writeonly buffer Ob { uint Outp[]; };
layout(push_constant) uniform PC { uint N; };
void main() {
    uint w = gl_GlobalInvocationID.x;
    uint numWords = (N + 1u) >> 1;
    if (w >= numWords) return;
    vec2 a = unpackHalf2x16(A[w]);
    vec2 b = unpackHalf2x16(B[w]);
    Outp[w] = packHalf2x16(a + b);
}";

    // Row softmax over the last axis: ONE WORK-GROUP PER ROW (gl_WorkGroupID.x == row), processed PER WORD
    // (two half elements per uint). FP32 max/sum reductions in shared memory, packed-half in/out. The packed
    // layout requires EVEN cols so each row is word-aligned and per-word writes never race across rows — the
    // dispatch rejects odd cols. Counterpart of the CUDA fp16_softmax_native.
    public static string Softmax => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) writeonly buffer Ob { uint Outp[]; };
layout(push_constant) uniform PC { uint rows; uint cols; };
shared float sdata[256];
void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint bs = gl_WorkGroupSize.x;
    uint words = cols >> 1u;
    uint base = row * words;
    float m = -3.4e38;
    for (uint i = tid; i < words; i += bs) { vec2 v = unpackHalf2x16(Inp[base + i]); m = max(m, max(v.x, v.y)); }
    sdata[tid] = m; barrier();
    for (uint s = bs >> 1u; s > 0u; s >>= 1u) { if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]); barrier(); }
    float rowmax = sdata[0]; barrier();
    float sum = 0.0;
    for (uint i = tid; i < words; i += bs) { vec2 v = unpackHalf2x16(Inp[base + i]); sum += exp(v.x - rowmax) + exp(v.y - rowmax); }
    sdata[tid] = sum; barrier();
    for (uint s = bs >> 1u; s > 0u; s >>= 1u) { if (tid < s) sdata[tid] += sdata[tid + s]; barrier(); }
    float inv = 1.0 / sdata[0]; barrier();
    for (uint i = tid; i < words; i += bs) {
        vec2 v = unpackHalf2x16(Inp[base + i]);
        Outp[base + i] = packHalf2x16(vec2(exp(v.x - rowmax) * inv, exp(v.y - rowmax) * inv));
    }
}";

    // Row layernorm over the last axis with packed-half gamma/beta: ONE WORK-GROUP PER ROW, per word. FP32
    // mean/var reductions in shared memory, packed-half in/out; writes per-row FP32 mean + variance (always
    // — the dispatch supplies temporaries when the caller passes none). Population variance (/cols), eps
    // inside inversesqrt. EVEN cols required (packed layout). Counterpart of the CUDA fp16_layernorm_native.
    public static string LayerNorm => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) readonly buffer Gb { uint Gamma[]; };
layout(set=0,binding=2) readonly buffer Bb { uint Beta[]; };
layout(set=0,binding=3) writeonly buffer Ob { uint Outp[]; };
layout(set=0,binding=4) writeonly buffer Mb { float MeanOut[]; };
layout(set=0,binding=5) writeonly buffer Vb { float VarOut[]; };
layout(push_constant) uniform PC { uint rows; uint cols; float eps; };
shared float sdata[256];
void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint bs = gl_WorkGroupSize.x;
    uint words = cols >> 1u;
    uint base = row * words;
    float s = 0.0;
    for (uint i = tid; i < words; i += bs) { vec2 v = unpackHalf2x16(Inp[base + i]); s += v.x + v.y; }
    sdata[tid] = s; barrier();
    for (uint st = bs >> 1u; st > 0u; st >>= 1u) { if (tid < st) sdata[tid] += sdata[tid + st]; barrier(); }
    float mean = sdata[0] / float(cols); barrier();
    float vv = 0.0;
    for (uint i = tid; i < words; i += bs) { vec2 v = unpackHalf2x16(Inp[base + i]); float d0 = v.x - mean; float d1 = v.y - mean; vv += d0 * d0 + d1 * d1; }
    sdata[tid] = vv; barrier();
    for (uint st = bs >> 1u; st > 0u; st >>= 1u) { if (tid < st) sdata[tid] += sdata[tid + st]; barrier(); }
    float var = sdata[0] / float(cols); barrier();
    float invstd = inversesqrt(var + eps);
    if (tid == 0u) { MeanOut[row] = mean; VarOut[row] = var; }
    for (uint i = tid; i < words; i += bs) {
        vec2 v = unpackHalf2x16(Inp[base + i]);
        vec2 g = unpackHalf2x16(Gamma[i]);
        vec2 b = unpackHalf2x16(Beta[i]);
        float n0 = (v.x - mean) * invstd;
        float n1 = (v.y - mean) * invstd;
        Outp[base + i] = packHalf2x16(vec2(n0 * g.x + b.x, n1 * g.y + b.y));
    }
}";

    // ---- Any column count (#1230). The kernels above give a row its own words, which only works when a row is a
    // whole number of words, that is an even column count. With an odd count the last half of one row and the
    // first half of the next share a word, so a row cannot be written by its own work-group without racing its
    // neighbour. These split the op in two: a per-row reduction that writes the row's statistics as FP32, and an
    // apply pass with ONE INVOCATION PER OUTPUT WORD, which looks up the row of each of its two elements. Every
    // word is written by exactly one invocation, whatever the column count. Elements are addressed by their half
    // index e = row * cols + column: word e / 2, low half when e is even.
    private const string HalfAt = @"
float half_at(uint e) { vec2 v = unpackHalf2x16(Inp[e >> 1u]); return (e & 1u) == 0u ? v.x : v.y; }
";

    // stats[2 * row] = the row's maximum, stats[2 * row + 1] = 1 / sum(exp(x - max)). One work-group per row.
    public static string SoftmaxRowStats => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) writeonly buffer Sb { float Stats[]; };
layout(push_constant) uniform PC { uint rows; uint cols; };
shared float sdata[256];
" + HalfAt + @"
void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint bs = gl_WorkGroupSize.x;
    uint base = row * cols;
    float m = -3.4e38;
    for (uint i = tid; i < cols; i += bs) m = max(m, half_at(base + i));
    sdata[tid] = m; barrier();
    for (uint s = bs >> 1u; s > 0u; s >>= 1u) { if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]); barrier(); }
    float rowmax = sdata[0]; barrier();
    float sum = 0.0;
    for (uint i = tid; i < cols; i += bs) sum += exp(half_at(base + i) - rowmax);
    sdata[tid] = sum; barrier();
    for (uint s = bs >> 1u; s > 0u; s >>= 1u) { if (tid < s) sdata[tid] += sdata[tid + s]; barrier(); }
    if (tid == 0u) { Stats[2u * row] = rowmax; Stats[2u * row + 1u] = 1.0 / sdata[0]; }
}";

    public static string SoftmaxApply => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) readonly buffer Sb { float Stats[]; };
layout(set=0,binding=2) writeonly buffer Ob { uint Outp[]; };
layout(push_constant) uniform PC { uint rows; uint cols; };
" + HalfAt + @"
float value(uint e, uint total) {
    if (e >= total) return 0.0;          // the unused half of a final word
    uint row = e / cols;
    return exp(half_at(e) - Stats[2u * row]) * Stats[2u * row + 1u];
}
void main() {
    uint total = rows * cols;
    uint w = gl_GlobalInvocationID.x;
    if (w >= (total + 1u) >> 1u) return;
    float v0 = value(2u * w, total);
    float v1 = value(2u * w + 1u, total);
    Outp[w] = packHalf2x16(vec2(v0, v1));
}";

    // MeanOut[row] and VarOut[row] (population variance), which the apply pass then reads. One work-group per row.
    public static string LayerNormRowStats => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) writeonly buffer Mb { float MeanOut[]; };
layout(set=0,binding=2) writeonly buffer Vb { float VarOut[]; };
layout(push_constant) uniform PC { uint rows; uint cols; };
shared float sdata[256];
" + HalfAt + @"
void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint bs = gl_WorkGroupSize.x;
    uint base = row * cols;
    float s = 0.0;
    for (uint i = tid; i < cols; i += bs) s += half_at(base + i);
    sdata[tid] = s; barrier();
    for (uint st = bs >> 1u; st > 0u; st >>= 1u) { if (tid < st) sdata[tid] += sdata[tid + st]; barrier(); }
    float mean = sdata[0] / float(cols); barrier();
    float vv = 0.0;
    for (uint i = tid; i < cols; i += bs) { float d = half_at(base + i) - mean; vv += d * d; }
    sdata[tid] = vv; barrier();
    for (uint st = bs >> 1u; st > 0u; st >>= 1u) { if (tid < st) sdata[tid] += sdata[tid + st]; barrier(); }
    if (tid == 0u) { MeanOut[row] = mean; VarOut[row] = sdata[0] / float(cols); }
}";

    public static string LayerNormApply => Header + @"
layout(set=0,binding=0) readonly buffer Ib { uint Inp[]; };
layout(set=0,binding=1) readonly buffer Gb { uint Gamma[]; };
layout(set=0,binding=2) readonly buffer Bb { uint Beta[]; };
layout(set=0,binding=3) readonly buffer Mb { float Mean[]; };
layout(set=0,binding=4) readonly buffer Vb { float Var[]; };
layout(set=0,binding=5) writeonly buffer Ob { uint Outp[]; };
layout(push_constant) uniform PC { uint rows; uint cols; float eps; };
" + HalfAt + @"
float gamma_at(uint c) { vec2 v = unpackHalf2x16(Gamma[c >> 1u]); return (c & 1u) == 0u ? v.x : v.y; }
float beta_at(uint c) { vec2 v = unpackHalf2x16(Beta[c >> 1u]); return (c & 1u) == 0u ? v.x : v.y; }
float value(uint e, uint total) {
    if (e >= total) return 0.0;          // the unused half of a final word
    uint row = e / cols;
    uint c = e - row * cols;
    return (half_at(e) - Mean[row]) * inversesqrt(Var[row] + eps) * gamma_at(c) + beta_at(c);
}
void main() {
    uint total = rows * cols;
    uint w = gl_GlobalInvocationID.x;
    if (w >= (total + 1u) >> 1u) return;
    float v0 = value(2u * w, total);
    float v1 = value(2u * w + 1u, total);
    Outp[w] = packHalf2x16(vec2(v0, v1));
}";
}
