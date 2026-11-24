// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Add then activation in one pass (AddRelu / AddSigmoid / AddGelu, the residual join of a ResNet block): the same
// arithmetic as the element-wise add followed by relu, sigmoid and gelu_op, without writing and re-reading the sum.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    private const string AddActivationModuleName = "AddActivation";

    // ElementWiseSource's bindings, so DispatchBinaryOpAsync launches it.
    private const string AddActivationSource = @"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF: f32 = 0.044715;

// tanh with its relative precision near 0, where gelu's 0.5 x (1 + tanh(u)) needs it: WGSL bounds tanh only as its exp
// form, which near 0 loses most digits (gelu(-0.105) came out 7e-5 off). A series below 0.25 (next term under 1e-8
// relative), 1 - 2 / (exp(2|u|) + 1) above, where the cancellation costs at most 2 bits.
fn tanh_accurate(u: f32) -> f32 {
    let a = abs(u);
    if (a < 0.25) {
        let z = u * u;
        return u + u * z * (-1.0 / 3.0 + z * (2.0 / 15.0 + z * (-17.0 / 315.0 + z * (62.0 / 2835.0))));
    }
    let t = 1.0 - 2.0 / (exp(2.0 * a) + 1.0);
    return select(t, -t, u < 0.0);
}

@compute @workgroup_size(256)
fn add_relu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        // relu's contract: a NaN sum stays NaN (tested on the bits; WGSL may assume no NaN), -0 becomes +0.
        let s = A[idx] + B[idx];
        C[idx] = select(select(0.0, s, s > 0.0), s, (bitcast<u32>(s) & 0x7fffffffu) > 0x7f800000u);
    }
}

@compute @workgroup_size(256)
fn add_sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        C[idx] = 1.0 / (1.0 + exp(-(A[idx] + B[idx])));
    }
}

@compute @workgroup_size(256)
fn add_gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.size) {
        let x = A[idx] + B[idx];
        C[idx] = 0.5 * x * (1.0 + tanh_accurate(SQRT_2_OVER_PI * (x + GELU_COEF * x * x * x)));
    }
}
";

    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => DispatchBinaryOpAsync(AddActivationModuleName, AddActivationSource, "add_relu", A, B, C, size).GetAwaiter().GetResult();

    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => DispatchBinaryOpAsync(AddActivationModuleName, AddActivationSource, "add_sigmoid", A, B, C, size).GetAwaiter().GetResult();

    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => DispatchBinaryOpAsync(AddActivationModuleName, AddActivationSource, "add_gelu", A, B, C, size).GetAwaiter().GetResult();
}
#endif
