// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

internal static partial class WebGpuBatchExecutionKernels
{
    private static class RfftMagnitudeShaders
    {
        private const string Norm = @"
fn bx_rfft_norm(re: f32, im: f32, epsilon: f32) -> f32 {
    var peak = max(abs(re), abs(im));
    if (peak > 1e19 || (peak > 0.0 && peak < 1e-18) || epsilon > 1e37) {
        peak = max(peak, sqrt(epsilon));
        let scaled = vec3<f32>(re, im, sqrt(epsilon)) / peak;
        return peak * sqrt(dot(scaled, scaled));
    }
    return sqrt((re * re + im * im) + epsilon);
}
";
        internal static readonly string Magnitude = Shader(new[] { "r:spectrum", "rw:output" },
            "rows: i32, n: i32, spectrumStride: i32, epsilon: f32", Norm + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x); let bins = p.n / 2 + 1; if (i >= p.rows * bins) { return; }
    let source = 2 * ((i / bins) * p.spectrumStride + i % bins);
    let re = spectrum[source]; let im = spectrum[source + 1];
    output[i] = bx_rfft_norm(re, im, p.epsilon);
}
");
        internal static readonly string Seed = Shader(new[] { "r:spectrum", "r:upstream", "rw:seed" },
            "rows: i32, n: i32, spectrumStride: i32, epsilon: f32", Norm + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = i32(id.x); if (i >= p.rows * p.n) { return; }
    let k = i % p.n; let bins = p.n / 2 + 1;
    var re = 0.0; var im = 0.0;
    if (k < bins) {
        let source = 2 * ((i / p.n) * p.spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        let magnitude = bx_rfft_norm(re, im, p.epsilon);
        let grad = upstream[(i / p.n) * bins + k];
        re = (re / magnitude) * grad; im = (im / magnitude) * grad;
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
