// Copyright (c) AiDotNet. All rights reserved.
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
}
