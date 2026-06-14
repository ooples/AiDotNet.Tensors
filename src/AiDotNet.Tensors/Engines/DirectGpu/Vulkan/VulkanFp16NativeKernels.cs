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
}
