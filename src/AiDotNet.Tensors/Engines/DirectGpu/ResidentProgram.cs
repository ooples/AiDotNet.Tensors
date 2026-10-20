using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>A stage of a resident program (#890 phase 3).</summary>
internal enum ResidentOp
{
    /// <summary><c>out[r,c] = max(0, a[r,c] * b[c] + c[c])</c>: a per-column scale and bias, then ReLU.</summary>
    AffineRelu = 0,

    /// <summary><c>out[r,c] = a[r,c] * b[c] + c[c]</c>.</summary>
    Affine = 1,

    /// <summary><c>out[r,:] = softmax(a[r,:])</c>, one row per block.</summary>
    RowSoftmax = 2,

    /// <summary><c>out = a + b</c>, elementwise.</summary>
    Add = 3,
}

/// <summary>One declared stage: the op and its operands over a [rows, cols] tensor (unused operands may be null).</summary>
internal readonly struct ResidentStage
{
    public ResidentStage(ResidentOp op, int rows, int cols, IGpuBuffer a, IGpuBuffer? b, IGpuBuffer? c, IGpuBuffer output)
    {
        if (rows <= 0 || cols <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "A stage needs a non-empty [rows, cols] shape.");
        if ((long)rows * cols > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(rows), "A stage is limited to int.MaxValue elements.");
        bool needsB = op != ResidentOp.RowSoftmax;
        bool needsC = op == ResidentOp.Affine || op == ResidentOp.AffineRelu;
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (needsB && b is null) throw new ArgumentNullException(nameof(b), $"{op} reads a second operand.");
        if (needsC && c is null) throw new ArgumentNullException(nameof(c), $"{op} reads a bias operand.");
        // A short buffer would make the kernel read or write past it: on CUDA error 700, which poisons the context --
        // and the backend runs on the device's SHARED primary context, so one bad stage would break every engine in
        // the process. Refused here as an ordinary exception instead, for every backend.
        long elements = (long)rows * cols;
        RequireSize(a, elements, op, "input", nameof(a));
        RequireSize(output, elements, op, "output", nameof(output));
        if (needsB && b is not null) RequireSize(b, op == ResidentOp.Add ? elements : cols, op, "second operand", nameof(b));
        if (needsC && c is not null) RequireSize(c, cols, op, "bias", nameof(c));
        Op = op; Rows = rows; Cols = cols; A = a; B = b; C = c; Output = output;
    }

    private static void RequireSize(IGpuBuffer buffer, long needed, ResidentOp op, string role, string parameter)
    {
        if (buffer.Size < needed)
            throw new ArgumentException($"{op}: the {role} holds {buffer.Size} elements; the stage needs {needed}.", parameter);
    }

    public ResidentOp Op { get; }
    public int Rows { get; }
    public int Cols { get; }
    public IGpuBuffer A { get; }
    public IGpuBuffer? B { get; }
    public IGpuBuffer? C { get; }
    public IGpuBuffer Output { get; }
}

/// <summary>Source dialects of <see cref="ResidentProgramKernelTemplate"/>.</summary>
internal enum ResidentKernelDialect
{
    OpenCl,
    Hip,
    Metal,
    Glsl,
    Wgsl,
}

/// <summary>
/// The resident program's stage kernel for HIP, OpenCL, Vulkan, Metal and WebGPU (#1448-#1452), written once: the
/// arithmetic of CUDA's <c>res_run_stage</c> (<c>CUDA/Kernels/CudaResidentProgramKernels.cs</c>), with the row
/// softmax reduced by a fixed-order shared-memory tree rather than warp shuffles, since a wavefront is 32 or 64 lanes
/// depending on the AMD device and the shader languages have no portable shuffle.
/// <para>
/// Only CUDA and HIP have a cooperative launch, the co-residency guarantee a spin-wait grid barrier needs; HIP adds
/// <see cref="HipProgramKernel"/>, a port of CUDA's persistent kernel. OpenCL, Vulkan, Metal and WebGPU guarantee no
/// forward progress across work-groups, so a spin barrier could deadlock there. Their resident program is the chain
/// in ONE submission (one flush, command buffer or queue submit) with the API's own ordering between the stage
/// dispatches; run separately, each stage is its own submission.
/// </para>
/// <para>
/// Indexing is 32-bit unsigned: a stage holds at most int.MaxValue elements and a launch at most
/// <see cref="MaxGroups"/> groups of <see cref="GroupThreads"/>, so an index plus one grid stride stays below 2^32.
/// </para>
/// </summary>
internal static class ResidentProgramKernelTemplate
{
    internal const string StageKernel = "resident_stage";

    /// <summary>HIP only: the persistent kernel that walks the stage list with a grid barrier between stages.</summary>
    internal const string HipProgramKernel = "resident_program";

    /// <summary>Threads per group; a softmax row is reduced by one group of this size.</summary>
    internal const int GroupThreads = 256;

    /// <summary>The most stages one program holds (HIP's stage array travels in the kernel's parameter block).</summary>
    internal const int MaxStages = 8;

    /// <summary>The most groups one stage launches (the smallest per-dimension limit: Vulkan's and WebGPU's
    /// guaranteed 65535); the kernel strides over the rest.</summary>
    internal const int MaxGroups = 65535;

    /// <summary>Groups for one stage: a row each for the softmax, one per <see cref="GroupThreads"/> elements
    /// otherwise, capped at <see cref="MaxGroups"/>.</summary>
    internal static int GroupsFor(in ResidentStage s)
    {
        long wanted = s.Op == ResidentOp.RowSoftmax
            ? s.Rows
            : ((long)s.Rows * s.Cols + GroupThreads - 1) / GroupThreads;
        return (int)Math.Max(1, Math.Min(MaxGroups, wanted));
    }

    /// <summary>The kernels <see cref="GetModuleSource"/> defines for <paramref name="dialect"/>.</summary>
    internal static string[] GetKernelNames(ResidentKernelDialect dialect)
        => dialect == ResidentKernelDialect.Hip ? new[] { StageKernel, HipProgramKernel } : new[] { StageKernel };

    /// <summary>Validates a chain's length; every backend refuses an empty or over-long chain the same way.</summary>
    internal static void RequireChain(ReadOnlySpan<ResidentStage> stages)
    {
        if (stages.Length == 0) throw new ArgumentException("A resident program needs at least one stage.", nameof(stages));
        if (stages.Length > MaxStages)
            throw new ArgumentException($"A resident program holds at most {MaxStages} stages.", nameof(stages));
    }

    /// <summary>
    /// The stage kernel (and on HIP the program kernel) as one OpenCL C, HIP C++ or Metal Shading Language module.
    /// Parameters: buffers <c>a, b, c, o</c> then ints <c>op, rows, cols</c>; Metal binds them at indices 0-3 and
    /// 4-6.
    /// </summary>
    internal static string GetModuleSource(ResidentKernelDialect dialect)
    {
        if (dialect == ResidentKernelDialect.Glsl) return GlslSource;
        if (dialect == ResidentKernelDialect.Wgsl) return WgslSource;
        string gin, gout, lq, fn, bar, fmaxf, fmaf, expf, negInf;
        switch (dialect)
        {
            case ResidentKernelDialect.OpenCl:
                gin = "__global const float*"; gout = "__global float*"; lq = "__local float*"; fn = "";
                bar = "barrier(CLK_LOCAL_MEM_FENCE)"; fmaxf = "fmax"; fmaf = "fma"; expf = "exp"; negInf = "(-INFINITY)";
                break;
            case ResidentKernelDialect.Metal:
                gin = "device const float*"; gout = "device float*"; lq = "threadgroup float*"; fn = "static ";
                bar = "threadgroup_barrier(mem_flags::mem_threadgroup)"; fmaxf = "fmax"; fmaf = "fma";
                expf = "precise::exp"; negInf = "(-INFINITY)";
                break;
            default:
                gin = "const float*"; gout = "float*"; lq = "float*"; fn = "__device__ ";
                bar = "__syncthreads()"; fmaxf = "fmaxf"; fmaf = "fmaf"; expf = "expf"; negInf = "__int_as_float(0xff800000)";
                break;
        }

        var sb = new StringBuilder();
        if (dialect == ResidentKernelDialect.Metal) sb.Append("#include <metal_stdlib>\nusing namespace metal;\n");
        if (dialect == ResidentKernelDialect.Hip) sb.Append("typedef unsigned int uint;\n");
        sb.Append("#define RES_THREADS ").Append(GroupThreads).Append("u\n");
        sb.Append(fn).Append("float res_reduce(float v, int isMax, ").Append(lq).Append(@" scratch, uint lid)
{
    scratch[lid] = v;
    ").Append(bar).Append(@";
    for (uint s = RES_THREADS / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            float o2 = scratch[lid + s];
            scratch[lid] = isMax != 0 ? ").Append(fmaxf).Append(@"(scratch[lid], o2) : scratch[lid] + o2;
        }
        ").Append(bar).Append(@";
    }
    float r = scratch[0];
    ").Append(bar).Append(@";
    return r;
}

").Append(fn).Append("void res_run_stage(int op, int rows, int cols, ").Append(gin).Append(" a, ").Append(gin).Append(" b, ")
            .Append(gin).Append(" c, ").Append(gout).Append(" o, ").Append(lq).Append(@" scratch, uint lid, uint grp, uint ngrp)
{
    uint ucols = (uint)cols;
    if (op == 2) {
        for (uint r = grp; r < (uint)rows; r += ngrp) {
            uint base0 = r * ucols;
            float m = ").Append(negInf).Append(@";
            for (uint j = lid; j < ucols; j += RES_THREADS) m = ").Append(fmaxf).Append(@"(m, a[base0 + j]);
            m = res_reduce(m, 1, scratch, lid);
            float sum = 0.0f;
            for (uint j = lid; j < ucols; j += RES_THREADS) sum += ").Append(expf).Append(@"(a[base0 + j] - m);
            sum = res_reduce(sum, 0, scratch, lid);
            float inv = 1.0f / sum;
            for (uint j = lid; j < ucols; j += RES_THREADS) o[base0 + j] = ").Append(expf).Append(@"(a[base0 + j] - m) * inv;
        }
        return;
    }
    uint n = (uint)rows * ucols;
    uint stride = ngrp * RES_THREADS;
    for (uint i = grp * RES_THREADS + lid; i < n; i += stride) {
        float v;
        if (op == 3) {
            v = a[i] + b[i];
        } else {
            uint col = i % ucols;
            v = ").Append(fmaf).Append(@"(a[i], b[col], c[col]);
            if (op == 0) v = ").Append(fmaxf).Append(@"(v, 0.0f);
        }
        o[i] = v;
    }
}
");
        switch (dialect)
        {
            case ResidentKernelDialect.OpenCl:
                sb.Append(@"
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void resident_stage(__global const float* a, __global const float* b, __global const float* c, __global float* o,
    const int op, const int rows, const int cols)
{
    __local float scratch[RES_THREADS];
    res_run_stage(op, rows, cols, a, b, c, o, scratch, (uint)get_local_id(0), (uint)get_group_id(0), (uint)get_num_groups(0));
}
");
                break;
            case ResidentKernelDialect.Metal:
                sb.Append(@"
kernel void resident_stage(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]], device float* o [[buffer(3)]],
    constant int& op [[buffer(4)]], constant int& rows [[buffer(5)]], constant int& cols [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]],
    uint ngrp [[threadgroups_per_grid]])
{
    threadgroup float scratch[RES_THREADS];
    res_run_stage(op, rows, cols, a, b, c, o, scratch, lid, grp, ngrp);
}
");
                break;
            default:
                sb.Append(HipKernels);
                break;
        }
        return sb.ToString();
    }

    // HIP: the stage kernel and the persistent program kernel. The barrier is CUDA's: arrivals counted in bar[0], a
    // generation published in bar[1], the last arrival resetting bar[0] so the two words serve launch after launch.
    // The spin wait is safe only under a cooperative launch, which guarantees every block is co-resident.
    private static readonly string HipKernels = @"
struct ResStage {
    int op; int rows; int cols; int pad;
    const float* a; const float* b; const float* c; float* out;
};

struct ResProgram {
    ResStage stages[" + MaxStages + @"];
    int count; int pad0; unsigned int* bar;
};

__device__ __forceinline__ void res_grid_barrier(unsigned int* bar)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int* gen = bar + 1;
        unsigned int g = *gen;
        __threadfence();
        unsigned int arrived = atomicAdd(bar, 1u);
        if (arrived == gridDim.x - 1) {
            bar[0] = 0u;
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) { }
        }
        __threadfence();
    }
    __syncthreads();
}

extern ""C"" __global__ void __launch_bounds__(RES_THREADS) resident_stage(const float* a, const float* b,
    const float* c, float* o, int op, int rows, int cols)
{
    __shared__ float scratch[RES_THREADS];
    res_run_stage(op, rows, cols, a, b, c, o, scratch, threadIdx.x, blockIdx.x, gridDim.x);
}

extern ""C"" __global__ void __launch_bounds__(RES_THREADS) resident_program(ResProgram p)
{
    __shared__ float scratch[RES_THREADS];
    for (int k = 0; k < p.count; k++) {
        const ResStage& s = p.stages[k];
        res_run_stage(s.op, s.rows, s.cols, s.a, s.b, s.c, s.out, scratch, threadIdx.x, blockIdx.x, gridDim.x);
        if (k + 1 < p.count) res_grid_barrier(p.bar);
    }
}
";

    // Vulkan: buffers at bindings 0-3 (a, b, c, o), push constants op, rows, cols.
    private static readonly string GlslSource = @"#version 450
layout(local_size_x = " + GroupThreads + @") in;
layout(set=0, binding=0) readonly buffer BA { float a[]; };
layout(set=0, binding=1) readonly buffer BB { float b[]; };
layout(set=0, binding=2) readonly buffer BC { float c[]; };
layout(set=0, binding=3) buffer BO { float o[]; };
layout(push_constant) uniform PC { int op; int rows; int cols; };
const uint RES_THREADS = " + GroupThreads + @"u;
shared float scratch[RES_THREADS];

float res_reduce(float v, bool isMax, uint lid)
{
    scratch[lid] = v;
    memoryBarrierShared(); barrier();
    for (uint s = RES_THREADS / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            float o2 = scratch[lid + s];
            scratch[lid] = isMax ? max(scratch[lid], o2) : scratch[lid] + o2;
        }
        memoryBarrierShared(); barrier();
    }
    float r = scratch[0];
    memoryBarrierShared(); barrier();
    return r;
}

void main()
{
    uint lid = gl_LocalInvocationID.x, grp = gl_WorkGroupID.x, ngrp = gl_NumWorkGroups.x;
    uint ucols = uint(cols);
    if (op == 2) {
        for (uint r = grp; r < uint(rows); r += ngrp) {
            uint base0 = r * ucols;
            float m = uintBitsToFloat(0xff800000u);
            for (uint j = lid; j < ucols; j += RES_THREADS) m = max(m, a[base0 + j]);
            m = res_reduce(m, true, lid);
            float sum = 0.0;
            for (uint j = lid; j < ucols; j += RES_THREADS) sum += exp(a[base0 + j] - m);
            sum = res_reduce(sum, false, lid);
            float inv = 1.0 / sum;
            for (uint j = lid; j < ucols; j += RES_THREADS) o[base0 + j] = exp(a[base0 + j] - m) * inv;
        }
        return;
    }
    uint n = uint(rows) * ucols;
    uint stride = ngrp * RES_THREADS;
    for (uint i = grp * RES_THREADS + lid; i < n; i += stride) {
        float v;
        if (op == 3) {
            v = a[i] + b[i];
        } else {
            uint col = i % ucols;
            v = fma(a[i], b[col], c[col]);
            if (op == 0) v = max(v, 0.0);
        }
        o[i] = v;
    }
}
";

    // WebGPU: storage buffers at bindings 0-3 (a, b, c, o) in group 0, the ints in a uniform at binding 4. A
    // writable binding may not alias a read one in one dispatch, so WebGpuBackend refuses an output that is an input.
    private static readonly string WgslSource = @"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read> c : array<f32>;
@group(0) @binding(3) var<storage, read_write> o : array<f32>;
struct Params { op : i32, rows : i32, cols : i32, pad : i32 };
@group(0) @binding(4) var<uniform> p : Params;
const RES_THREADS : u32 = " + GroupThreads + @"u;
var<workgroup> scratch : array<f32, " + GroupThreads + @">;

fn res_reduce(v : f32, isMax : bool, lid : u32) -> f32 {
    scratch[lid] = v;
    workgroupBarrier();
    var s : u32 = RES_THREADS / 2u;
    loop {
        if (s == 0u) { break; }
        if (lid < s) {
            let o2 = scratch[lid + s];
            scratch[lid] = select(scratch[lid] + o2, max(scratch[lid], o2), isMax);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let r = scratch[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(" + GroupThreads + @")
fn main(@builtin(local_invocation_id) l : vec3<u32>, @builtin(workgroup_id) w : vec3<u32>,
        @builtin(num_workgroups) nw : vec3<u32>) {
    let lid = l.x;
    let ngrp = nw.x;
    let ucols = u32(p.cols);
    if (p.op == 2) {
        // Every invocation of a workgroup runs the same number of rows, so the barriers in res_reduce stay uniform.
        for (var r : u32 = w.x; r < u32(p.rows); r = r + ngrp) {
            let base0 = r * ucols;
            var m : f32 = bitcast<f32>(0xff800000u);
            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { m = max(m, a[base0 + j]); }
            m = res_reduce(m, true, lid);
            var sum : f32 = 0.0;
            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { sum = sum + exp(a[base0 + j] - m); }
            sum = res_reduce(sum, false, lid);
            let inv = 1.0 / sum;
            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { o[base0 + j] = exp(a[base0 + j] - m) * inv; }
        }
        return;
    }
    let n = u32(p.rows) * ucols;
    let stride = ngrp * RES_THREADS;
    for (var i : u32 = w.x * RES_THREADS + lid; i < n; i = i + stride) {
        var v : f32;
        if (p.op == 3) {
            v = a[i] + b[i];
        } else {
            let col = i % ucols;
            v = fma(a[i], b[col], c[col]);
            if (p.op == 0) { v = max(v, 0.0); }
        }
        o[i] = v;
    }
}
";
}
