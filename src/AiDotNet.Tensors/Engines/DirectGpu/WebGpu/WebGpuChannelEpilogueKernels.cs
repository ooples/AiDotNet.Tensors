#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of ChannelEpilogueKernelTemplate, restated operation for operation. Each kernel is its own module: the
// storage buffers in the template's binding order (the written one at 1), then the ints in a uniform struct padded to 16
// bytes. NaN is tested by its bits (WGSL lets an implementation assume v == v). WGSL does not promise that a multiply and
// an add stay separately rounded, so where an implementation contracts them the values may differ from the separate passes
// in the last bit.
internal static class WebGpuChannelEpilogueKernels
{
    private const string ParamsDecl =
        "struct Params { count: i32, channels: i32, spatial: i32, mode: i32, slopeBits: i32, offset: i32, pad0: i32, pad1: i32, }";

    private static readonly string ForwardSource = @"
@group(0) @binding(0) var<storage, read> zin: array<f32>;
@group(0) @binding(1) var<storage, read_write> yout: array<f32>;
@group(0) @binding(2) var<storage, read> bia: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> shf: array<f32>;
" + ParamsDecl + @"
@group(0) @binding(5) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn " + ChannelEpilogueKernelTemplate.Forward + @"(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    if (gid < pm.count) {
        let i = pm.offset + gid;
        let c = (i / pm.spatial) % pm.channels;
        var v = zin[i];
        if ((pm.mode & 1) != 0) { v = v + bia[c]; }
        if ((pm.mode & 2) != 0) { v = v * scl[c]; }
        if ((pm.mode & 4) != 0) { v = v + shf[c]; }
        let act = (pm.mode >> 3u) & 3;
        if (act == 1) { v = select(select(0.0, v, v > 0.0), v, (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u); }
        else if (act == 2) { v = select(bitcast<f32>(pm.slopeBits) * v, v, v > 0.0); }
        yout[i] = v;
    }
}
";

    private static readonly string BackwardSource = @"
@group(0) @binding(0) var<storage, read> dyin: array<f32>;
@group(0) @binding(1) var<storage, read_write> dzout: array<f32>;
@group(0) @binding(2) var<storage, read> yin: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
" + ParamsDecl + @"
@group(0) @binding(4) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn " + ChannelEpilogueKernelTemplate.Backward + @"(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    if (gid < pm.count) {
        let i = pm.offset + gid;
        let c = (i / pm.spatial) % pm.channels;
        var g = dyin[i];
        let act = (pm.mode >> 3u) & 3;
        if (act == 1) { g = select(0.0, g, yin[i] > 0.0); }
        else if (act == 2) { g = select(bitcast<f32>(pm.slopeBits) * g, g, yin[i] > 0.0); }
        if ((pm.mode & 2) != 0) { g = g * scl[c]; }
        dzout[i] = g;
    }
}
";

    private static readonly string LayoutSource = @"
@group(0) @binding(0) var<storage, read> lin: array<f32>;
@group(0) @binding(1) var<storage, read_write> lout: array<f32>;
" + ParamsDecl + @"
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn " + ChannelEpilogueKernelTemplate.Layout + @"(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    if (gid < pm.count) {
        let i = pm.offset + gid;
        var src: i32;
        if ((pm.mode & 1) != 0) {
            let c = i % pm.channels;
            let t = i / pm.channels;
            let s = t % pm.spatial;
            src = ((t / pm.spatial) * pm.channels + c) * pm.spatial + s;
        } else {
            let s = i % pm.spatial;
            let t = i / pm.spatial;
            let c = t % pm.channels;
            src = ((t / pm.channels) * pm.spatial + s) * pm.channels + c;
        }
        lout[i] = lin[src];
    }
}
";

    /// <summary>The WGSL module of one <see cref="ChannelEpilogueKernelTemplate"/> kernel.</summary>
    internal static string SourceFor(string name) => name == ChannelEpilogueKernelTemplate.Forward ? ForwardSource
        : name == ChannelEpilogueKernelTemplate.Backward ? BackwardSource
        : name == ChannelEpilogueKernelTemplate.Layout ? LayoutSource
        : throw new System.Collections.Generic.KeyNotFoundException(name);

    /// <summary>The uniform words every kernel's Params struct holds (its six ints, padded to 16 bytes).</summary>
    internal const int UniformWords = 8;
}
#endif
