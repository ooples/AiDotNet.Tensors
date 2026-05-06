// Copyright (c) AiDotNet. All rights reserved.

#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static class WebGpuIssue301FusedKernels
{
    private const string ActivationHelper = @"
fn issue301_apply_activation(xIn : f32, activation : i32) -> f32 {
    var x = xIn;
    if (activation == 1) { return max(x, 0.0); }
    if (activation == 2) {
        let k = sqrt(2.0 / 3.14159265358979323846);
        return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)));
    }
    if (activation == 3) { return 1.0 / (1.0 + exp(-x)); }
    if (activation == 4) { return tanh(x); }
    if (activation == 5) { return select(0.01 * x, x, x > 0.0); }
    if (activation == 6) {
        let sigmoid = 1.0 / (1.0 + exp(-x));
        return x * sigmoid;
    }
    return x;
}
";

    public const string LoRAForward = @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> baseOutput : array<f32>;
@group(0) @binding(2) var<storage, read> loraA : array<f32>;
@group(0) @binding(3) var<storage, read> loraB : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;
struct P { batchSize : i32, inputFeatures : i32, rank : i32, outputFeatures : i32, scaling : f32, pad0 : f32, pad1 : f32, pad2 : f32 };
@group(0) @binding(5) var<uniform> p : P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.outputFeatures;
    if (gid >= total) { return; }

    let b = gid / p.outputFeatures;
    let o = gid - b * p.outputFeatures;
    var delta : f32 = 0.0;

    for (var r : i32 = 0; r < p.rank; r = r + 1) {
        var hidden : f32 = 0.0;
        for (var i : i32 = 0; i < p.inputFeatures; i = i + 1) {
            hidden = hidden + input_[b * p.inputFeatures + i] * loraA[i * p.rank + r];
        }
        delta = delta + hidden * loraB[r * p.outputFeatures + o];
    }

    output[gid] = baseOutput[gid] + p.scaling * delta;
}
";

    public const string DDIMStep = @"
@group(0) @binding(0) var<storage, read> xT : array<f32>;
@group(0) @binding(1) var<storage, read> epsilonTheta : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
struct P { size : i32, alphaBarT : f32, alphaBarTMinus1 : f32, pad : f32 };
@group(0) @binding(3) var<uniform> p : P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid >= p.size) { return; }

    let eps = epsilonTheta[gid];
    let x0Pred = (xT[gid] - sqrt(max(0.0, 1.0 - p.alphaBarT)) * eps) / sqrt(p.alphaBarT);
    output[gid] = sqrt(p.alphaBarTMinus1) * x0Pred + sqrt(max(0.0, 1.0 - p.alphaBarTMinus1)) * eps;
}
";

    public static string SparseLinear => ActivationHelper + @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> packedCsr : array<f32>;
@group(0) @binding(2) var<storage, read> sparseValues : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;
struct P { batchSize : i32, inputFeatures : i32, outputFeatures : i32, nnz : i32, hasBias : i32, activation : i32, pad0 : i32, pad1 : i32 };
@group(0) @binding(5) var<uniform> p : P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.batchSize * p.outputFeatures;
    if (gid >= total) { return; }

    let b = gid / p.outputFeatures;
    let o = gid - b * p.outputFeatures;
    let rowStart = bitcast<i32>(packedCsr[o]);
    let rowEnd = bitcast<i32>(packedCsr[o + 1]);
    let colBase = p.outputFeatures + 1;
    var sum : f32 = select(0.0, bias[o], p.hasBias != 0);

    for (var idx : i32 = rowStart; idx < rowEnd; idx = idx + 1) {
        let col = bitcast<i32>(packedCsr[colBase + idx]);
        sum = sum + input_[b * p.inputFeatures + col] * sparseValues[idx];
    }

    output[gid] = issue301_apply_activation(sum, p.activation);
}
";
}
#endif
