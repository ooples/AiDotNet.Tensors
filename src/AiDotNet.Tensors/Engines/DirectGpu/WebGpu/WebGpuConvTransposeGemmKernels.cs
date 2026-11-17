// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of ConvTransposeGemmKernelTemplate: the batch regroup and the overlap-add gather of the single-GEMM
// transposed-convolution forward, restated operation for operation. Each kernel is its own module: the storage buffers in
// the template's binding order (the written one at 1), then the ints in a uniform struct padded to 16 bytes.
internal static class WebGpuConvTransposeGemmKernels
{
    private const string RegroupSource = @"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> xt: array<f32>;
struct Params { nb: i32, C: i32, L: i32, srcOffset: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn NAME(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    let total = pm.nb * pm.C * pm.L;
    if (gid < total) {
        let l = gid % pm.L;
        let r = gid / pm.L;
        let c = r % pm.C;
        let b = r / pm.C;
        xt[c * (pm.nb * pm.L) + b * pm.L + l] = src[pm.srcOffset + gid];
    }
}
";

    private const string OverlapAddSource = @"
@group(0) @binding(0) var<storage, read> col: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct Params { nb: i32, OC: i32, OH: i32, OW: i32, KH: i32, KW: i32, SH: i32, SW: i32, PH: i32, PW: i32, IH: i32,
    IW: i32, dstOffset: i32, pad0: i32, pad1: i32, pad2: i32, }
@group(0) @binding(2) var<uniform> pm: Params;
@compute @workgroup_size(256)
fn NAME(@builtin(global_invocation_id) gidv: vec3<u32>) {
    let gid = i32(gidv.x);
    let total = pm.nb * pm.OC * pm.OH * pm.OW;
    if (gid < total) {
        let ow = gid % pm.OW;
        var r = gid / pm.OW;
        let oh = r % pm.OH;
        r = r / pm.OH;
        let c = r % pm.OC;
        let b = r / pm.OC;
        let n = pm.nb * pm.IH * pm.IW;
        var sum = 0.0;
        for (var kh = (oh + pm.PH) % pm.SH; kh < pm.KH; kh = kh + pm.SH) {
            let ih = (oh + pm.PH - kh) / pm.SH;
            if (ih >= 0 && ih < pm.IH) {
                for (var kw = (ow + pm.PW) % pm.SW; kw < pm.KW; kw = kw + pm.SW) {
                    let iw = (ow + pm.PW - kw) / pm.SW;
                    if (iw >= 0 && iw < pm.IW) {
                        sum = sum + col[((c * pm.KH + kh) * pm.KW + kw) * n + (b * pm.IH + ih) * pm.IW + iw];
                    }
                }
            }
        }
        dst[pm.dstOffset + gid] = sum;
    }
}
";

    private static readonly Dictionary<string, string> Sources = new()
    {
        [ConvTransposeGemmKernelTemplate.Regroup] = RegroupSource.Replace("NAME", ConvTransposeGemmKernelTemplate.Regroup),
        [ConvTransposeGemmKernelTemplate.OverlapAdd] =
            OverlapAddSource.Replace("NAME", ConvTransposeGemmKernelTemplate.OverlapAdd),
    };

    /// <summary>The WGSL module of one <see cref="ConvTransposeGemmKernelTemplate"/> kernel.</summary>
    internal static string SourceFor(string name) => Sources[name];

    /// <summary>The uniform words a kernel's Params struct holds (its ints, padded to 16 bytes).</summary>
    internal static int UniformWords(string name) => name == ConvTransposeGemmKernelTemplate.Regroup ? 4 : 16;
}
