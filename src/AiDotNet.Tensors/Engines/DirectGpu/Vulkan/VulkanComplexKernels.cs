// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GLSL compute shaders for native Complex<T> tensor operations.
// These are compiled to SPIR-V at runtime via shaderc.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanComplexKernels
{
    public static string ComplexMultiply => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer AR { float aReal[]; };
layout(set = 0, binding = 1) readonly buffer AI { float aImag[]; };
layout(set = 0, binding = 2) readonly buffer BR { float bReal[]; };
layout(set = 0, binding = 3) readonly buffer BI { float bImag[]; };
layout(set = 0, binding = 4) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 5) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float ar = aReal[idx], ai = aImag[idx];
    float br = bReal[idx], bi = bImag[idx];
    outReal[idx] = ar * br - ai * bi;
    outImag[idx] = ar * bi + ai * br;
}";

    public static string ComplexConjugate => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IR { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer II { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 3) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx];
    outImag[idx] = -inImag[idx];
}";

    public static string ComplexMagnitude => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IR { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer II { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OM { float outMag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMag[idx] = sqrt(re * re + im * im);
}";

    public static string ComplexMagnitudeSquared => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IR { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer II { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OMS { float outMagSq[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    outMagSq[idx] = re * re + im * im;
}";

    public static string ComplexPhase => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IR { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer II { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OP { float outPhase[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    outPhase[idx] = atan(inImag[idx], inReal[idx]);
}";

    public static string ComplexFromPolar => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer MG { float mag[]; };
layout(set = 0, binding = 1) readonly buffer PH { float phase[]; };
layout(set = 0, binding = 2) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 3) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float m = mag[idx], p = phase[idx];
    outReal[idx] = m * cos(p);
    outImag[idx] = m * sin(p);
}";

    public static string ComplexScale => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer IR { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer II { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 3) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; float scalar; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    outReal[idx] = inReal[idx] * scalar;
    outImag[idx] = inImag[idx] * scalar;
}";

    public static string ComplexAdd => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer AR { float aReal[]; };
layout(set = 0, binding = 1) readonly buffer AI { float aImag[]; };
layout(set = 0, binding = 2) readonly buffer BR { float bReal[]; };
layout(set = 0, binding = 3) readonly buffer BI { float bImag[]; };
layout(set = 0, binding = 4) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 5) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    outReal[idx] = aReal[idx] + bReal[idx];
    outImag[idx] = aImag[idx] + bImag[idx];
}";

    public static string ComplexCrossSpectral => @"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer XR { float xReal[]; };
layout(set = 0, binding = 1) readonly buffer XI { float xImag[]; };
layout(set = 0, binding = 2) readonly buffer YR { float yReal[]; };
layout(set = 0, binding = 3) readonly buffer YI { float yImag[]; };
layout(set = 0, binding = 4) writeonly buffer OR { float outReal[]; };
layout(set = 0, binding = 5) writeonly buffer OI { float outImag[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float xr = xReal[idx], xi = xImag[idx];
    float yr = yReal[idx], yi = yImag[idx];
    outReal[idx] = xr * yr + xi * yi;
    outImag[idx] = xi * yr - xr * yi;
}";
}
