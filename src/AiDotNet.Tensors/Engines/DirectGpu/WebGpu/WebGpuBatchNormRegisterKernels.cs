namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// The WGSL instance of BatchNormRegisterKernelTemplate (one 256-wide workgroup per channel, the slice in private arrays,
// mode bits for the fused forms), restated operation for operation. Each kernel is its own module: the storage buffers
// in the template's binding order, the ints in a uniform struct after them (epsilon and 1/count as float bits). Every
// workgroupBarrier sits in uniform control flow: the loops' bounds are constants.
internal static class WebGpuBatchNormRegisterKernels
{
    private const string Params = @"
struct Params { batch: i32, channels: i32, spatial: i32, epsBits: i32, invCountBits: i32, mode: i32, pad0: i32, pad1: i32, }
var<workgroup> sm: array<f32, 256>;

fn group_sum(tid: u32, value: f32) -> f32 {
    sm[tid] = value;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { sm[tid] = sm[tid] + sm[tid + st]; }
        workgroupBarrier();
    }
    let r = sm[0];
    workgroupBarrier();
    return r;
}
";

    private const string Forward = @"
@group(0) @binding(0) var<storage, read> xin: array<f32>;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read_write> yout: array<f32>;
@group(0) @binding(3) var<storage, read> prm: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;
@group(0) @binding(5) var<uniform> pm: Params;
" + Params + @"
@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lidx;
    let t = i32(lidx);
    let c = i32(wid.x);
    let n = pm.batch * pm.spatial;
    let eps = bitcast<f32>(pm.epsBits);
    let invCount = bitcast<f32>(pm.invCountBits);
    var v: array<f32, KPT>;
    var s1 = 0.0;
    for (var k = 0; k < KPT; k = k + 1) {
        let e = t + k * 256;
        v[k] = 0.0;
        if (e < n) {
            let b = e / pm.spatial;
            v[k] = xin[(b * pm.channels + c) * pm.spatial + (e - b * pm.spatial)];
            s1 = s1 + v[k];
        }
    }
    let mean = group_sum(tid, s1) * invCount;
    var s2 = 0.0;
    for (var k = 0; k < KPT; k = k + 1) {
        if (t + k * 256 < n) {
            let d = v[k] - mean;
            s2 = s2 + d * d;
        }
    }
    let varc = group_sum(tid, s2) * invCount;
    let inv = inverseSqrt(varc + eps);
    if (t == 0) { stats[c] = mean; stats[pm.channels + c] = varc; }
    let g = prm[c];
    let bt = prm[pm.channels + c];
    for (var k = 0; k < KPT; k = k + 1) {
        let e = t + k * 256;
        if (e < n) {
            let b = e / pm.spatial;
            let idx = (b * pm.channels + c) * pm.spatial + (e - b * pm.spatial);
            var y = fma(g, (v[k] - mean) * inv, bt);
            if ((pm.mode & 2) != 0) { y = y + res[idx]; }
            if ((pm.mode & 1) != 0) { y = max(y, 0.0); }
            yout[idx] = y;
        }
    }
}
";

    private const string Backward = @"
@group(0) @binding(0) var<storage, read> dyin: array<f32>;
@group(0) @binding(1) var<storage, read> msk: array<f32>;
@group(0) @binding(2) var<storage, read_write> dres: array<f32>;
@group(0) @binding(3) var<storage, read> xin: array<f32>;
@group(0) @binding(4) var<storage, read> prm: array<f32>;
@group(0) @binding(5) var<storage, read_write> dxout: array<f32>;
@group(0) @binding(6) var<storage, read_write> dprm: array<f32>;
@group(0) @binding(7) var<storage, read> dy2: array<f32>;
@group(0) @binding(8) var<uniform> pm: Params;
" + Params + @"
@compute @workgroup_size(256)
fn NAME(@builtin(local_invocation_index) lidx: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lidx;
    let t = i32(lidx);
    let c = i32(wid.x);
    let n = pm.batch * pm.spatial;
    let eps = bitcast<f32>(pm.epsBits);
    let invCount = bitcast<f32>(pm.invCountBits);
    let g = prm[c];
    let bt = prm[pm.channels + c];
    let mean = prm[2 * pm.channels + c];
    let inv = inverseSqrt(prm[3 * pm.channels + c] + eps);
    var xv: array<f32, KPT>;
    var dv: array<f32, KPT>;
    var sdy = 0.0;
    var sdyx = 0.0;
    for (var k = 0; k < KPT; k = k + 1) {
        let e = t + k * 256;
        xv[k] = 0.0;
        dv[k] = 0.0;
        if (e < n) {
            let b = e / pm.spatial;
            let idx = (b * pm.channels + c) * pm.spatial + (e - b * pm.spatial);
            let x = xin[idx];
            var d = dyin[idx];
            if ((pm.mode & 8) != 0) { d = d + dy2[idx]; }
            if ((pm.mode & 4) != 0) {
                if (!(fma(g, (x - mean) * inv, bt) > 0.0)) { d = 0.0; }
            } else if ((pm.mode & 1) != 0) {
                if (!(msk[idx] > 0.0)) { d = 0.0; }
            }
            if ((pm.mode & 2) != 0) { dres[idx] = d; }
            xv[k] = x;
            dv[k] = d;
            sdy = sdy + d;
            sdyx = sdyx + d * ((x - mean) * inv);
        }
    }
    let sumDy = group_sum(tid, sdy);
    let sumDyXhat = group_sum(tid, sdyx);
    if (t == 0) { dprm[c] = sumDyXhat; dprm[pm.channels + c] = sumDy; }
    let sumDxhat = g * sumDy;
    let sumDxhatXhat = g * sumDyXhat;
    for (var k = 0; k < KPT; k = k + 1) {
        let e = t + k * 256;
        if (e < n) {
            let b = e / pm.spatial;
            let idx = (b * pm.channels + c) * pm.spatial + (e - b * pm.spatial);
            dxout[idx] = inv * (dv[k] * g - invCount * (sumDxhat + ((xv[k] - mean) * inv) * sumDxhatXhat));
        }
    }
}
";

    /// <summary>The module source for a <see cref="BatchNormRegisterKernelTemplate"/> kernel name.</summary>
    internal static string SourceFor(string name)
    {
        foreach (int k in BatchNormRegisterKernelTemplate.PerThread)
        {
            string kpt = k.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (name == BatchNormRegisterKernelTemplate.ForwardName(k)) return Forward.Replace("NAME", name).Replace("KPT", kpt);
            if (name == BatchNormRegisterKernelTemplate.BackwardName(k)) return Backward.Replace("NAME", name).Replace("KPT", kpt);
        }
        throw new System.ArgumentException($"No WGSL BatchNorm kernel named '{name}'.", nameof(name));
    }
}
