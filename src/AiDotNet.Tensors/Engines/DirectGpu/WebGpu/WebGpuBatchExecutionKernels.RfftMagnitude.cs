// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    private static class RfftMagnitudeShaders
    {
        internal static readonly string Magnitude = Shader(new[] { "r:spectrum", "rw:output" },
            "rows: i32, n: i32, spectrumStride: i32, epsilon: f32", @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x); let bins = p.n / 2 + 1; if (i >= p.rows * bins) { return; }
    let source = 2 * ((i / bins) * p.spectrumStride + i % bins);
    let re = spectrum[source]; let im = spectrum[source + 1];
    output[i] = sqrt((re * re + im * im) + p.epsilon);
}
");
        internal static readonly string Seed = Shader(new[] { "r:spectrum", "r:upstream", "rw:seed" },
            "rows: i32, n: i32, spectrumStride: i32, epsilon: f32", @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x); if (i >= p.rows * p.n) { return; }
    let k = i % p.n; let bins = p.n / 2 + 1;
    var re = 0.0; var im = 0.0;
    if (k < bins) {
        let source = 2 * ((i / p.n) * p.spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        let scale = upstream[(i / p.n) * bins + k] / sqrt((re * re + im * im) + p.epsilon);
        re = re * scale; im = im * scale;
    }
    seed[2 * i] = re; seed[2 * i + 1] = im;
}
");
        internal static readonly string Gradient = Shader(new[] { "r:complexValues", "rw:output" },
            "count: i32, scale: f32, accumulate: i32", @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x); if (i >= p.count) { return; }
    let value = complexValues[2 * i] * p.scale;
    if (p.accumulate != 0) { output[i] = output[i] + value; } else { output[i] = value; }
}
");
    }
}
#endif
