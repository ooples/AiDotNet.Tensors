#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    // Nested initialization avoids a partial-file static field ordering dependency
    // when ByKernelName is constructed.
    private static class WeightNormShaders
    {
        internal static readonly string Forward = Shader(new[] { "r:v", "r:g", "rw:w" },
            "rows: i32, columns: i32, epsilon: f32", @"
var<workgroup> normPart: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) lane3: vec3<u32>) {
    let row = i32(group.x); let lane = i32(lane3.x);
    // Every dispatched workgroup is one valid row: the barrier is reached uniformly.
    var sum = 0.0;
    for (var j = lane; j < p.columns; j += 256) { let x = v[row * p.columns + j]; sum += x * x; }
    normPart[lane] = sum; workgroupBarrier();
    for (var s = 128; s > 0; s = s / 2) {
        if (lane < s) { normPart[lane] += normPart[lane + s]; }
        workgroupBarrier();
    }
    let scale = g[row] * (1.0 / sqrt(normPart[0] + p.epsilon));
    for (var j = lane; j < p.columns; j += 256) { w[row * p.columns + j] = v[row * p.columns + j] * scale; }
}
");
        internal static readonly string Stats = Shader(new[] { "r:h", "r:v", "rw:stats" },
            "rows: i32, columns: i32, epsilon: f32", @"
var<workgroup> normPart: array<f32, 256>;
var<workgroup> dotPart: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) lane3: vec3<u32>) {
    let row = i32(group.x); let lane = i32(lane3.x);
    var sum = 0.0; var dot = 0.0;
    for (var j = lane; j < p.columns; j += 256) {
        let x = v[row * p.columns + j]; sum += x * x; dot += h[row * p.columns + j] * x;
    }
    normPart[lane] = sum; dotPart[lane] = dot; workgroupBarrier();
    for (var s = 128; s > 0; s = s / 2) {
        if (lane < s) { normPart[lane] += normPart[lane + s]; dotPart[lane] += dotPart[lane + s]; }
        workgroupBarrier();
    }
    if (lane == 0) { stats[2 * row] = 1.0 / sqrt(normPart[0] + p.epsilon); stats[2 * row + 1] = dotPart[0]; }
}
");
        internal static readonly string DirectionGrad = Shader(new[] { "r:h", "r:v", "r:g", "r:stats", "rw:dv" },
            "rows: i32, columns: i32, accumulate: i32", Main + @"
    if (i >= p.rows * p.columns) { return; }
    let row = i / p.columns; let inv = stats[2 * row];
    let value = (g[row] * inv) * (h[i] - v[i] * (stats[2 * row + 1] * inv * inv));
    if (p.accumulate != 0) { dv[i] += value; } else { dv[i] = value; }
}
");
        internal static readonly string GainGrad = Shader(new[] { "r:stats", "rw:dg" },
            "rows: i32, accumulate: i32", Main + @"
    if (i >= p.rows) { return; }
    let value = stats[2 * i + 1] * stats[2 * i];
    if (p.accumulate != 0) { dg[i] += value; } else { dg[i] = value; }
}
");
    }
}
#endif

