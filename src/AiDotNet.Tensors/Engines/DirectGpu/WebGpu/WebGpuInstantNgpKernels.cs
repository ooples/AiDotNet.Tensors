#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public static class WebGpuInstantNgpKernels
{
    private const string Common = @"
struct P {
    numPoints: i32,
    resolution: i32,
    tableSize: i32,
    featuresPerLevel: i32,
    levelOffset: i32,
    outputStride: i32,
};
@group(0) @binding(3) var<uniform> p : P;

fn instant_ngp_hash(x: i32, y: i32, z: i32) -> u32 {
    let hx = bitcast<u32>(x) * 73856093u;
    let hy = bitcast<u32>(y) * 19349663u;
    let hz = bitcast<u32>(z) * 83492791u;
    return (hx ^ hy ^ hz) % u32(p.tableSize);
}

fn instant_ngp_clamp_position(value: f32) -> f32 {
    if (value <= 0.0) { return 0.0; }
    if (value >= 0.999999) { return 0.999999; }
    return value;
}
";

    public static string Forward => @"
@group(0) @binding(0) var<storage, read> positions : array<f32>;
@group(0) @binding(1) var<storage, read> hashTable : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
" + Common + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.numPoints * p.featuresPerLevel;
    if (gid >= total) { return; }
    let n = gid / p.featuresPerLevel;
    let f = gid - n * p.featuresPerLevel;
    let gx = instant_ngp_clamp_position(positions[n * 3]) * f32(p.resolution);
    let gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * f32(p.resolution);
    let gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * f32(p.resolution);
    let x0 = i32(floor(gx)); let y0 = i32(floor(gy)); let z0 = i32(floor(gz));
    let x1 = x0 + 1; let y1 = y0 + 1; let z1 = z0 + 1;
    let fx = gx - f32(x0); let fy = gy - f32(y0); let fz = gz - f32(z0);
    let ix = 1.0 - fx; let iy = 1.0 - fy; let iz = 1.0 - fz;
    let h000 = i32(instant_ngp_hash(x0, y0, z0));
    let h001 = i32(instant_ngp_hash(x0, y0, z1));
    let h010 = i32(instant_ngp_hash(x0, y1, z0));
    let h011 = i32(instant_ngp_hash(x0, y1, z1));
    let h100 = i32(instant_ngp_hash(x1, y0, z0));
    let h101 = i32(instant_ngp_hash(x1, y0, z1));
    let h110 = i32(instant_ngp_hash(x1, y1, z0));
    let h111 = i32(instant_ngp_hash(x1, y1, z1));
    let value =
        ix * iy * iz * hashTable[h000 * p.featuresPerLevel + f] +
        ix * iy * fz * hashTable[h001 * p.featuresPerLevel + f] +
        ix * fy * iz * hashTable[h010 * p.featuresPerLevel + f] +
        ix * fy * fz * hashTable[h011 * p.featuresPerLevel + f] +
        fx * iy * iz * hashTable[h100 * p.featuresPerLevel + f] +
        fx * iy * fz * hashTable[h101 * p.featuresPerLevel + f] +
        fx * fy * iz * hashTable[h110 * p.featuresPerLevel + f] +
        fx * fy * fz * hashTable[h111 * p.featuresPerLevel + f];
    output_[n * p.outputStride + p.levelOffset + f] = value;
}
";

    public static string Backward => @"
@group(0) @binding(0) var<storage, read> positions : array<f32>;
@group(0) @binding(1) var<storage, read> outputGradient : array<f32>;
@group(0) @binding(2) var<storage, read_write> tableGradient : array<f32>;
" + Common + @"
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.tableSize * p.featuresPerLevel;
    if (gid >= total) { return; }
    let entry = u32(gid / p.featuresPerLevel);
    let f = gid - i32(entry) * p.featuresPerLevel;
    var acc = 0.0;
    for (var n = 0; n < p.numPoints; n = n + 1) {
        let gx = instant_ngp_clamp_position(positions[n * 3]) * f32(p.resolution);
        let gy = instant_ngp_clamp_position(positions[n * 3 + 1]) * f32(p.resolution);
        let gz = instant_ngp_clamp_position(positions[n * 3 + 2]) * f32(p.resolution);
        let x0 = i32(floor(gx)); let y0 = i32(floor(gy)); let z0 = i32(floor(gz));
        let x1 = x0 + 1; let y1 = y0 + 1; let z1 = z0 + 1;
        let fx = gx - f32(x0); let fy = gy - f32(y0); let fz = gz - f32(z0);
        let ix = 1.0 - fx; let iy = 1.0 - fy; let iz = 1.0 - fz;
        let grad = outputGradient[n * p.outputStride + p.levelOffset + f];
        if (abs(grad) < 1.0e-10) { continue; }
        if (instant_ngp_hash(x0, y0, z0) == entry) { acc = acc + grad * ix * iy * iz; }
        if (instant_ngp_hash(x0, y0, z1) == entry) { acc = acc + grad * ix * iy * fz; }
        if (instant_ngp_hash(x0, y1, z0) == entry) { acc = acc + grad * ix * fy * iz; }
        if (instant_ngp_hash(x0, y1, z1) == entry) { acc = acc + grad * ix * fy * fz; }
        if (instant_ngp_hash(x1, y0, z0) == entry) { acc = acc + grad * fx * iy * iz; }
        if (instant_ngp_hash(x1, y0, z1) == entry) { acc = acc + grad * fx * iy * fz; }
        if (instant_ngp_hash(x1, y1, z0) == entry) { acc = acc + grad * fx * fy * iz; }
        if (instant_ngp_hash(x1, y1, z1) == entry) { acc = acc + grad * fx * fy * fz; }
    }
    tableGradient[gid] = acc;
}
";

    public static string UniqueConsecutive => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
@group(0) @binding(2) var<storage, read_write> outputCount : array<f32>;
struct UniqueP { length: i32 };
@group(0) @binding(3) var<uniform> p : UniqueP;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x != 0u) { return; }
    if (p.length <= 0) { outputCount[0] = 0.0; return; }
    var count = 1;
    output_[0] = input_[0];
    for (var i = 1; i < p.length; i = i + 1) {
        if (input_[i] != input_[i - 1]) {
            output_[count] = input_[i];
            count = count + 1;
        }
    }
    outputCount[0] = f32(count);
}
";

    public static string Nonzero => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read> strides : array<i32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
@group(0) @binding(3) var<storage, read_write> outputCount : array<f32>;
struct NonzeroP { length: i32, rank: i32 };
@group(0) @binding(4) var<uniform> p : NonzeroP;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x != 0u) { return; }
    var count = 0;
    for (var i = 0; i < p.length; i = i + 1) {
        if (input_[i] != 0.0) {
            var rem = i;
            for (var d = 0; d < p.rank; d = d + 1) {
                let stride = strides[d];
                output_[count * p.rank + d] = f32(rem / stride);
                rem = rem % stride;
            }
            count = count + 1;
        }
    }
    outputCount[0] = f32(count);
}
";

    public static string Mode => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct ModeP { length: i32 };
@group(0) @binding(2) var<uniform> p : ModeP;
fn mode_equal(left: f32, right: f32) -> bool {
    return left == right || (isNan(left) && isNan(right));
}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x != 0u) { return; }
    var bestValue = 0.0;
    var bestCount = -1;
    for (var i = 0; i < p.length; i = i + 1) {
        let candidate = input_[i];
        var first = true;
        for (var j = 0; j < i; j = j + 1) {
            if (mode_equal(candidate, input_[j])) { first = false; break; }
        }
        if (!first) { continue; }
        var count = 0;
        for (var j = 0; j < p.length; j = j + 1) {
            if (mode_equal(candidate, input_[j])) { count = count + 1; }
        }
        if (bestCount < 0 || count > bestCount ||
            (count == bestCount && candidate < bestValue)) {
            bestValue = candidate;
            bestCount = count;
        }
    }
    output_[0] = bestValue;
    output_[1] = f32(bestCount);
}
";

    public static string ConvertIndicesToInt32 => @"
@group(0) @binding(0) var<storage, read> input_ : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<i32>;
struct ConvertP { length: i32 };
@group(0) @binding(2) var<uniform> p : ConvertP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    if (gid < p.length) { output_[gid] = i32(input_[gid]); }
}
";

    public static string IndexAdd => @"
@group(0) @binding(0) var<storage, read> destination : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read> source : array<f32>;
@group(0) @binding(3) var<storage, read_write> output_ : array<f32>;
struct IndexAddP { outerSize: i32, sourceAxis: i32, destinationAxis: i32, innerSize: i32 };
@group(0) @binding(4) var<uniform> p : IndexAddP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.destinationAxis * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let dst = (gid / p.innerSize) % p.destinationAxis;
    let outer = (gid / p.innerSize) / p.destinationAxis;
    var value = destination[gid];
    for (var j = 0; j < p.sourceAxis; j = j + 1) {
        if (indices[j] == dst) {
            value = value + source[(outer * p.sourceAxis + j) * p.innerSize + inner];
        }
    }
    output_[gid] = value;
}
";

    public static string IndexSelect => @"
@group(0) @binding(0) var<storage, read> source : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
struct IndexSelectP { outerSize: i32, sourceAxis: i32, indexAxis: i32, innerSize: i32 };
@group(0) @binding(3) var<uniform> p : IndexSelectP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outerSize * p.indexAxis * p.innerSize;
    if (gid >= total) { return; }
    let inner = gid % p.innerSize;
    let j = (gid / p.innerSize) % p.indexAxis;
    let outer = (gid / p.innerSize) / p.indexAxis;
    let sourceIndex = indices[j];
    output_[gid] = source[(outer * p.sourceAxis + sourceIndex) * p.innerSize + inner];
}
";

    public static string ScatterMaxWithArgmaxRows => @"
@group(0) @binding(0) var<storage, read> source : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
@group(0) @binding(3) var<storage, read_write> argmax_ : array<f32>;
struct ScatterMaxP { sourceRows: i32, innerSize: i32, outputRows: i32, padding: i32 };
@group(0) @binding(4) var<uniform> p : ScatterMaxP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.outputRows * p.innerSize;
    if (gid >= total) { return; }
    let group = gid / p.innerSize;
    let inner = gid % p.innerSize;
    var best = bitcast<f32>(0xff800000u);
    var bestRow = -1;
    for (var row = 0; row < p.sourceRows; row = row + 1) {
        let value = source[row * p.innerSize + inner];
        if (indices[row] == group && value > best) {
            best = value;
            bestRow = row;
        }
    }
    output_[gid] = best;
    argmax_[gid] = f32(bestRow);
}
";

    public static string UniformMeshLaplacian => @"
@group(0) @binding(0) var<storage, read> faces : array<i32>;
@group(0) @binding(1) var<storage, read_write> output_ : array<f32>;
struct MeshP { numFaces: i32, numVertices: i32, padding0: i32, padding1: i32 };
@group(0) @binding(2) var<uniform> p : MeshP;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.numVertices * p.numVertices;
    if (gid >= total) { return; }
    let row = gid / p.numVertices;
    let column = gid % p.numVertices;
    var value = 0.0;
    for (var face = 0; face < p.numFaces; face = face + 1) {
        let v0 = faces[face * 3];
        let v1 = faces[face * 3 + 1];
        let v2 = faces[face * 3 + 2];
        if (row == v0 && column == v1) { value = value - 1.0; }
        if (row == v1 && column == v0) { value = value - 1.0; }
        if (row == v0 && column == v0) { value = value + 1.0; }
        if (row == v1 && column == v1) { value = value + 1.0; }
        if (row == v1 && column == v2) { value = value - 1.0; }
        if (row == v2 && column == v1) { value = value - 1.0; }
        if (row == v1 && column == v1) { value = value + 1.0; }
        if (row == v2 && column == v2) { value = value + 1.0; }
        if (row == v2 && column == v0) { value = value - 1.0; }
        if (row == v0 && column == v2) { value = value - 1.0; }
        if (row == v2 && column == v2) { value = value + 1.0; }
        if (row == v0 && column == v0) { value = value + 1.0; }
    }
    output_[gid] = value;
}
";

    public static string ScatterAddRows => @"
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read> indices:array<i32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
struct P{sourceRows:i32,innerSize:i32,outputRows:i32,pad:i32};@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.outputRows*p.innerSize){return;}let inner=gid%p.innerSize;let group=gid/p.innerSize;var sum=0.0;for(var row=0;row<p.sourceRows;row=row+1){if(indices[row]==group){sum=sum+source[row*p.innerSize+inner];}}output_[gid]=sum;}";

    public static string ScatterMeanRowsWithCounts => @"
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read> indices:array<i32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
@group(0) @binding(3) var<storage,read_write> counts:array<f32>;
struct P{sourceRows:i32,innerSize:i32,outputRows:i32,pad:i32};@group(0) @binding(4) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid<p.outputRows){var groupCount=0;for(var row=0;row<p.sourceRows;row=row+1){if(indices[row]==gid){groupCount=groupCount+1;}}counts[gid]=f32(groupCount);}if(gid>=p.outputRows*p.innerSize){return;}let inner=gid%p.innerSize;let group=gid/p.innerSize;var sum=0.0;var count=0;for(var row=0;row<p.sourceRows;row=row+1){if(indices[row]==group){sum=sum+source[row*p.innerSize+inner];count=count+1;}}if(count>0){output_[gid]=sum*(1.0/f32(count));}else{output_[gid]=0.0;}}";

    public static string ScatterSoftmaxRows => @"
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read> indices:array<i32>;
@group(0) @binding(2) var<storage,read_write> output_:array<f32>;
struct P{sourceRows:i32,innerSize:i32,numGroups:i32,pad:i32};@group(0) @binding(3) var<uniform> p:P;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.sourceRows*p.innerSize){return;}let inner=gid%p.innerSize;let row=gid/p.innerSize;let group=indices[row];if(group<0||group>=p.numGroups){output_[gid]=0.0;return;}var maximum=bitcast<f32>(0xff800000u);for(var other=0;other<p.sourceRows;other=other+1){if(indices[other]==group){maximum=max(maximum,source[other*p.innerSize+inner]);}}var sum=0.0;for(var other=0;other<p.sourceRows;other=other+1){if(indices[other]==group){sum=sum+exp(source[other*p.innerSize+inner]-maximum);}}let value=exp(source[gid]-maximum);if(sum!=0.0){output_[gid]=value/sum;}else{output_[gid]=value;}}";

    public static string ScatterAddBackwardRows => @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;@group(0) @binding(1) var<storage,read> indices:array<i32>;@group(0) @binding(2) var<storage,read_write> gradSource:array<f32>;struct P{sourceRows:i32,innerSize:i32,outputRows:i32,pad:i32};@group(0) @binding(3) var<uniform> p:P;@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.sourceRows*p.innerSize){return;}let inner=gid%p.innerSize;let group=indices[gid/p.innerSize];if(group>=0&&group<p.outputRows){gradSource[gid]=gradOutput[group*p.innerSize+inner];}else{gradSource[gid]=0.0;}}";

    public static string ScatterMeanBackwardRows => @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;@group(0) @binding(1) var<storage,read> indices:array<i32>;@group(0) @binding(2) var<storage,read> counts:array<i32>;@group(0) @binding(3) var<storage,read_write> gradSource:array<f32>;struct P{sourceRows:i32,innerSize:i32,outputRows:i32,pad:i32};@group(0) @binding(4) var<uniform> p:P;@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.sourceRows*p.innerSize){return;}let inner=gid%p.innerSize;let group=indices[gid/p.innerSize];if(group<0||group>=p.outputRows){gradSource[gid]=0.0;return;}let count=counts[group];var divisor=1.0;if(count>0){divisor=f32(count);}gradSource[gid]=gradOutput[group*p.innerSize+inner]/divisor;}";

    public static string ScatterMaxBackwardRows => @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;@group(0) @binding(1) var<storage,read> argmax_:array<i32>;@group(0) @binding(2) var<storage,read_write> gradSource:array<f32>;struct P{sourceRows:i32,innerSize:i32,outputRows:i32,pad:i32};@group(0) @binding(3) var<uniform> p:P;@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.sourceRows*p.innerSize){return;}let inner=gid%p.innerSize;let sourceRow=gid/p.innerSize;var value=0.0;for(var group=0;group<p.outputRows;group=group+1){if(argmax_[group*p.innerSize+inner]==sourceRow){value=gradOutput[group*p.innerSize+inner];}}gradSource[gid]=value;}";

    public static string ScatterSoftmaxBackwardRows => @"
@group(0) @binding(0) var<storage,read> gradOutput:array<f32>;@group(0) @binding(1) var<storage,read> output_:array<f32>;@group(0) @binding(2) var<storage,read> indices:array<i32>;@group(0) @binding(3) var<storage,read_write> gradSource:array<f32>;struct P{sourceRows:i32,innerSize:i32,numGroups:i32,pad:i32};@group(0) @binding(4) var<uniform> p:P;@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id:vec3<u32>){let gid=i32(id.x);if(gid>=p.sourceRows*p.innerSize){return;}let inner=gid%p.innerSize;let group=indices[gid/p.innerSize];if(group<0||group>=p.numGroups){gradSource[gid]=0.0;return;}var sum=0.0;for(var row=0;row<p.sourceRows;row=row+1){if(indices[row]==group){sum=sum+output_[row*p.innerSize+inner]*gradOutput[row*p.innerSize+inner];}}gradSource[gid]=output_[gid]*(gradOutput[gid]-sum);}";

    public static string ImportanceSampling => @"
@group(0) @binding(0) var<storage, read> tValues : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_ : array<f32>;
struct ImportanceP {
    numRays: i32,
    numCoarse: i32,
    numFine: i32,
    seed: u32,
};
@group(0) @binding(3) var<uniform> p : ImportanceP;
fn importance_hash(value: u32) -> u32 {
    var x = value;
    x = (x ^ (x >> 16u)) * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return x ^ (x >> 16u);
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let gid = i32(id.x);
    let total = p.numRays * p.numFine;
    if (gid >= total) { return; }
    let ray = gid / p.numFine;
    let sample = gid - ray * p.numFine;
    let base = ray * p.numCoarse;
    let bits = importance_hash(p.seed ^ (u32(gid) * 747796405u + 2891336453u));
    let random = f32(bits >> 8u) * (1.0 / 16777216.0);
    let u = (f32(sample) + random) / f32(p.numFine);
    var weightSum = 0.0;
    for (var s = 0; s < p.numCoarse; s = s + 1) {
        let weight = weights[base + s];
        if (weight > 0.0) { weightSum = weightSum + weight; }
    }
    if (weightSum <= 1.0e-10) {
        let tMin = tValues[base];
        let tMax = tValues[base + p.numCoarse - 1];
        output_[gid] = tMin + u * (tMax - tMin);
        return;
    }
    var previous = 0.0;
    var current = 0.0;
    var index = 0;
    for (var s = 0; s < p.numCoarse; s = s + 1) {
        let weight = weights[base + s];
        if (weight > 0.0) { current = current + weight / weightSum; }
        index = s;
        if (u <= current || s == p.numCoarse - 1) { break; }
        previous = current;
    }
    if (index == 0) { output_[gid] = tValues[base]; return; }
    let denominator = current - previous;
    let t0 = tValues[base + index - 1];
    let t1 = tValues[base + index];
    if (denominator > 1.0e-10) {
        output_[gid] = t0 + ((u - previous) / denominator) * (t1 - t0);
    } else {
        output_[gid] = t0;
    }
}
";

    public static string Nms => @"
@group(0) @binding(0) var<storage, read> boxes : array<f32>;
@group(0) @binding(1) var<storage, read> scores : array<f32>;
@group(0) @binding(2) var<storage, read> classIds : array<f32>;
@group(0) @binding(3) var<storage, read_write> suppressed : array<f32>;
@group(0) @binding(4) var<storage, read_write> output_ : array<f32>;
@group(0) @binding(5) var<storage, read_write> outputCount : array<f32>;
struct NmsP { length: i32, threshold: f32, batched: i32, pad0: i32 };
@group(0) @binding(6) var<uniform> p : NmsP;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x != 0u) { return; }
    var count = 0;
    for (var iteration = 0; iteration < p.length; iteration = iteration + 1) {
        var best = -1;
        var bestScore = -3.402823466e+38;
        for (var i = 0; i < p.length; i = i + 1) {
            if (suppressed[i] != 0.0) { continue; }
            let score = scores[i];
            if (best < 0 || (!isNan(score) &&
                (isNan(bestScore) || score > bestScore || (score == bestScore && i < best)))) {
                best = i; bestScore = score;
            }
        }
        if (best < 0) { break; }
        suppressed[best] = 1.0;
        output_[count] = f32(best);
        count = count + 1;
        let ix1 = boxes[best * 4]; let iy1 = boxes[best * 4 + 1];
        let ix2 = boxes[best * 4 + 2]; let iy2 = boxes[best * 4 + 3];
        let iw0 = ix2 - ix1; let ih0 = iy2 - iy1;
        var areaI = 0.0;
        if (iw0 > 0.0 && ih0 > 0.0) { areaI = iw0 * ih0; }
        for (var j = 0; j < p.length; j = j + 1) {
            if (suppressed[j] != 0.0) { continue; }
            if (p.batched != 0 && classIds[j] != classIds[best]) { continue; }
            let jx1 = boxes[j * 4]; let jy1 = boxes[j * 4 + 1];
            let jx2 = boxes[j * 4 + 2]; let jy2 = boxes[j * 4 + 3];
            let jw0 = jx2 - jx1; let jh0 = jy2 - jy1;
            var areaJ = 0.0;
            if (jw0 > 0.0 && jh0 > 0.0) { areaJ = jw0 * jh0; }
            let overlapW = max(0.0, min(ix2, jx2) - max(ix1, jx1));
            let overlapH = max(0.0, min(iy2, jy2) - max(iy1, jy1));
            let intersection = overlapW * overlapH;
            let unionArea = areaI + areaJ - intersection;
            if (unionArea > 0.0 && intersection / unionArea > p.threshold) {
                suppressed[j] = 1.0;
            }
        }
    }
    outputCount[0] = f32(count);
}
";

    public static string SpiralIndices => @"
@group(0) @binding(0) var<storage, read> vertices : array<f32>;
@group(0) @binding(1) var<storage, read> faces : array<f32>;
@group(0) @binding(2) var<storage, read_write> visited : array<f32>;
@group(0) @binding(3) var<storage, read_write> currentRing : array<f32>;
@group(0) @binding(4) var<storage, read_write> nextRing : array<f32>;
@group(0) @binding(5) var<storage, read_write> output_ : array<f32>;
struct SpiralP { numVertices: i32, numFaces: i32, spiralLength: i32, pad0: i32 };
@group(0) @binding(6) var<uniform> p : SpiralP;
fn append_next_unique(initialCount: i32, candidate: i32) -> i32 {
    var count = initialCount;
    for (var i = 0; i < count; i = i + 1) {
        if (i32(nextRing[i]) == candidate) { return count; }
    }
    if (count < p.numVertices) { nextRing[count] = f32(candidate); count = count + 1; }
    return count;
}
fn append_current_unique(initialCount: i32, candidate: i32) -> i32 {
    var count = initialCount;
    for (var i = 0; i < count; i = i + 1) {
        if (i32(currentRing[i]) == candidate) { return count; }
    }
    if (count < p.numVertices) { currentRing[count] = f32(candidate); count = count + 1; }
    return count;
}
fn build_neighbors(vertex: i32) -> i32 {
    var count = 0;
    for (var f = 0; f < p.numFaces; f = f + 1) {
        let v0 = i32(faces[f * 3]); let v1 = i32(faces[f * 3 + 1]); let v2 = i32(faces[f * 3 + 2]);
        if (vertex == v0) { count = append_current_unique(count, v1); count = append_current_unique(count, v2); }
        else if (vertex == v1) { count = append_current_unique(count, v0); count = append_current_unique(count, v2); }
        else if (vertex == v2) { count = append_current_unique(count, v0); count = append_current_unique(count, v1); }
    }
    return count;
}
fn spiral_angle(center: i32, reference: i32, vertex: i32) -> f32 {
    let cx = vertices[center * 3]; let cy = vertices[center * 3 + 1]; let cz = vertices[center * 3 + 2];
    let rx = vertices[reference * 3] - cx; let ry = vertices[reference * 3 + 1] - cy;
    let rz = vertices[reference * 3 + 2] - cz;
    let ax = vertices[vertex * 3] - cx; let ay = vertices[vertex * 3 + 1] - cy;
    let az = vertices[vertex * 3 + 2] - cz;
    return atan2(ax * ry - ay * rx, ax * rx + ay * ry + az * rz);
}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x != 0u) { return; }
    for (var center = 0; center < p.numVertices; center = center + 1) {
        for (var i = 0; i < p.numVertices; i = i + 1) { visited[i] = 0.0; }
        var currentCount = build_neighbors(center);
        if (currentCount > 1) {
            let reference = i32(currentRing[0]);
            for (var i = 1; i < currentCount; i = i + 1) {
                let key = currentRing[i];
                let keyAngle = spiral_angle(center, reference, i32(key));
                var j = i - 1;
                loop {
                    if (j < 0 || spiral_angle(center, reference, i32(currentRing[j])) <= keyAngle) { break; }
                    currentRing[j + 1] = currentRing[j];
                    j = j - 1;
                }
                currentRing[j + 1] = key;
            }
        }
        visited[center] = 1.0;
        var outputIndex = 0;
        loop {
            if (outputIndex >= p.spiralLength || currentCount <= 0) { break; }
            var nextCount = 0;
            for (var r = 0; r < currentCount; r = r + 1) {
                if (outputIndex >= p.spiralLength) { break; }
                let neighbor = i32(currentRing[r]);
                if (neighbor < 0 || neighbor >= p.numVertices || visited[neighbor] != 0.0) { continue; }
                output_[center * p.spiralLength + outputIndex] = f32(neighbor);
                outputIndex = outputIndex + 1;
                visited[neighbor] = 1.0;
                for (var f = 0; f < p.numFaces; f = f + 1) {
                    let v0 = i32(faces[f * 3]); let v1 = i32(faces[f * 3 + 1]); let v2 = i32(faces[f * 3 + 2]);
                    var a = -1; var b = -1;
                    if (neighbor == v0) { a = v1; b = v2; }
                    else if (neighbor == v1) { a = v0; b = v2; }
                    else if (neighbor == v2) { a = v0; b = v1; }
                    if (a >= 0 && a < p.numVertices && visited[a] == 0.0) { nextCount = append_next_unique(nextCount, a); }
                    if (b >= 0 && b < p.numVertices && visited[b] == 0.0) { nextCount = append_next_unique(nextCount, b); }
                }
            }
            currentCount = nextCount;
            for (var i = 0; i < nextCount; i = i + 1) { currentRing[i] = nextRing[i]; }
        }
        while (outputIndex < p.spiralLength) {
            output_[center * p.spiralLength + outputIndex] = -1.0;
            outputIndex = outputIndex + 1;
        }
    }
}
";

    public static string CtcLoss => @"
@group(0) @binding(0) var<storage, read> logProbs : array<f32>;
@group(0) @binding(1) var<storage, read> targets : array<f32>;
@group(0) @binding(2) var<storage, read> inputLengths : array<f32>;
@group(0) @binding(3) var<storage, read> targetLengths : array<f32>;
@group(0) @binding(4) var<storage, read_write> workspace : array<f32>;
@group(0) @binding(5) var<storage, read_write> losses : array<f32>;
struct CtcP {
    maxTime: i32,
    batchSize: i32,
    numClasses: i32,
    maxTargetLength: i32,
    blank: i32,
    pad0: i32,
    pad1: i32,
    pad2: i32,
};
@group(0) @binding(6) var<uniform> p : CtcP;
fn ctc_log_add(a: f32, b: f32) -> f32 {
    let negativeSentinel = -3.402823466e+38;
    if (a <= negativeSentinel) { return b; }
    if (b <= negativeSentinel) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let n = i32(id.x);
    if (n >= p.batchSize) { return; }
    let timeLength = i32(inputLengths[n]);
    let targetLength = i32(targetLengths[n]);
    var targetOffset = 0;
    for (var i = 0; i < n; i = i + 1) {
        targetOffset = targetOffset + i32(targetLengths[i]);
    }
    let states = 2 * targetLength + 1;
    let maxStates = 2 * p.maxTargetLength + 1;
    var previous = n * 2 * maxStates;
    var current = previous + maxStates;
    let negativeSentinel = -3.402823466e+38;
    for (var s = 0; s < states; s = s + 1) {
        workspace[previous + s] = negativeSentinel;
    }
    workspace[previous] = logProbs[n * p.numClasses + p.blank];
    if (states > 1) {
        let label = i32(targets[targetOffset]);
        workspace[previous + 1] = logProbs[n * p.numClasses + label];
    }
    for (var t = 1; t < timeLength; t = t + 1) {
        for (var s = 0; s < states; s = s + 1) {
            var label = p.blank;
            if ((s & 1) != 0) { label = i32(targets[targetOffset + s / 2]); }
            var sum = workspace[previous + s];
            if (s >= 1) { sum = ctc_log_add(sum, workspace[previous + s - 1]); }
            if (s >= 2) {
                var priorLabel = p.blank;
                if ((s & 1) != 0) {
                    priorLabel = i32(targets[targetOffset + (s - 2) / 2]);
                }
                if (label != p.blank && label != priorLabel) {
                    sum = ctc_log_add(sum, workspace[previous + s - 2]);
                }
            }
            workspace[current + s] = sum +
                logProbs[(t * p.batchSize + n) * p.numClasses + label];
        }
        let swap = previous; previous = current; current = swap;
    }
    var logProbability = workspace[previous + states - 1];
    if (states >= 2) {
        logProbability = ctc_log_add(logProbability, workspace[previous + states - 2]);
    }
    losses[n] = -logProbability;
}
";
}
#endif
