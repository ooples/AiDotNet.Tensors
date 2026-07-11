namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL gated activation kernels: GLU, GeGLU, ReGLU, SwiGLU + derivatives.
/// </summary>
public static class GatedActivationKernels
{
    public static string GetSource()
    {
        return @"
__kernel void glu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    output[idx] = value * (1.0f / (1.0f + exp(-gate)));
}
__kernel void glu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    float sig = 1.0f / (1.0f + exp(-gate)); float grad = grad_output[idx];
    grad_input[outer * fullDim + d] = grad * sig;
    grad_input[outer * fullDim + halfDim + d] = grad * value * sig * (1.0f - sig);
}
// #775: the gated variants must apply the activation to the GATE (second half b) and multiply by
// the value (first half a): out = a * act(b) — the same convention as glu_forward (a*sigmoid(b))
// and the CpuEngine reference. These kernels had it BACKWARDS (act(a)*b), so GPU GeGLU/ReGLU/SwiGLU
// diverged wildly from CPU (the op-parity scaffold caught it: e.g. ReGLU CPU 0 vs GPU -0.73). Fixed
// forward AND the paired backward (grad wrt a = grad*act(b); grad wrt b = grad*a*act'(b)).
__kernel void geglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    float g3 = gate * gate * gate;
    float gelu = 0.5f * gate * (1.0f + tanh(0.7978845608f * (gate + 0.044715f * g3)));
    output[idx] = value * gelu;
}
__kernel void geglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    float g3 = gate * gate * gate; float inner = 0.7978845608f * (gate + 0.044715f * g3);
    float th = tanh(inner); float gelu = 0.5f * gate * (1.0f + th);
    float gelu_d = 0.5f * (1.0f + th) + 0.5f * gate * (1.0f - th * th) * 0.7978845608f * (1.0f + 0.134145f * gate * gate);
    grad_input[outer * fullDim + d] = grad * gelu;                  // d/d value
    grad_input[outer * fullDim + halfDim + d] = grad * value * gelu_d; // d/d gate
}
__kernel void reglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    output[idx] = value * fmax(gate, 0.0f);
}
__kernel void reglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    grad_input[outer * fullDim + d] = grad * fmax(gate, 0.0f);                        // d/d value
    grad_input[outer * fullDim + halfDim + d] = grad * value * ((gate > 0.0f) ? 1.0f : 0.0f); // d/d gate
}
__kernel void swiglu_forward(__global const float* input, __global float* output, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d];
    float sig = 1.0f / (1.0f + exp(-gate));
    output[idx] = value * (gate * sig);
}
__kernel void swiglu_backward(__global const float* grad_output, __global const float* input, __global float* grad_input, int outerSize, int halfDim) {
    int idx = get_global_id(0); int total = outerSize * halfDim; if (idx >= total) return;
    int outer = idx / halfDim; int d = idx % halfDim; int fullDim = halfDim * 2;
    float value = input[outer * fullDim + d]; float gate = input[outer * fullDim + halfDim + d]; float grad = grad_output[idx];
    float sig = 1.0f / (1.0f + exp(-gate)); float swish = gate * sig;
    float swish_d = sig + gate * sig * (1.0f - sig);
    grad_input[outer * fullDim + d] = grad * swish;                 // d/d value
    grad_input[outer * fullDim + halfDim + d] = grad * value * swish_d; // d/d gate
}
__kernel void relu_derivative(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
}
__kernel void sigmoid_derivative(__global const float* sig_out, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float s = sig_out[idx]; output[idx] = s * (1.0f - s);
}
__kernel void tanh_derivative(__global const float* tanh_out, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float t = tanh_out[idx]; output[idx] = 1.0f - t * t;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "glu_forward", "glu_backward", "geglu_forward", "geglu_backward",
            "reglu_forward", "reglu_backward", "swiglu_forward", "swiglu_backward",
            "relu_derivative", "sigmoid_derivative", "tanh_derivative"
        };
    }
}
