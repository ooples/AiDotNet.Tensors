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
        => dialect == ResidentKernelDialect.Hip
            ? new[] { StageKernel, HipProgramKernel, RowProgramKernel }
            : new[] { StageKernel, RowProgramKernel };

    /// <summary>
    /// The whole chain as ONE dispatch when every stage is row-local (<see cref="TryPlanRowProgram"/>): a group takes
    /// a row through every stage, a group barrier with a device-memory fence between stages, then its next row. No
    /// group ever waits on another, so it needs no forward-progress guarantee and runs on every backend.
    /// </summary>
    internal const string RowProgramKernel = "resident_rows";

    /// <summary>Distinct buffers a row program binds (WebGPU's default limit of storage buffers per stage).</summary>
    internal const int MaxRowProgramBuffers = 8;

    /// <summary>
    /// A row-local chain lowered to buffer slots: <see cref="Buffers"/> are the chain's distinct buffers, and each
    /// stage word packs the op (bits 0-1) and the slots of a, b, c and the output (3 bits each from bit 2). An unread
    /// operand's slot is the input's.
    /// </summary>
    internal sealed class RowProgramPlan
    {
        internal RowProgramPlan(int rows, int cols, IGpuBuffer[] buffers, int[] stageWords)
        {
            Rows = rows; Cols = cols; Buffers = buffers; StageWords = stageWords;
        }

        internal int Rows { get; }
        internal int Cols { get; }
        internal IGpuBuffer[] Buffers { get; }
        internal int[] StageWords { get; }

        /// <summary>Groups to launch: a row each, capped at <see cref="MaxGroups"/> (groups then stride over rows).</summary>
        internal int Groups => Math.Max(1, Math.Min(MaxGroups, Rows));

        /// <summary>rows, cols, stage count, 0, then the eight stage words (unused ones 0): the kernels' parameter
        /// block, in this order everywhere.</summary>
        internal int[] Parameters()
        {
            var p = new int[4 + MaxStages];
            p[0] = Rows; p[1] = Cols; p[2] = StageWords.Length;
            Array.Copy(StageWords, 0, p, 4, StageWords.Length);
            return p;
        }
    }

    /// <summary>
    /// Lowers <paramref name="stages"/> to a row program when it is one: every stage has the same [rows, cols], the
    /// chain touches at most <see cref="MaxRowProgramBuffers"/> distinct buffers, and no stage writes a buffer any
    /// stage reads as a per-column operand (an affine's scale or bias), since those are read by every row and a row
    /// would see another row's write. Each stage then reads only rows that the same group wrote earlier. Buffers are
    /// told apart by reference: two views of one allocation count as two buffers.
    /// </summary>
    internal static bool TryPlanRowProgram(ReadOnlySpan<ResidentStage> stages, out RowProgramPlan? plan)
    {
        plan = null;
        if (stages.Length == 0 || stages.Length > MaxStages) return false;
        int rows = stages[0].Rows, cols = stages[0].Cols;
        var buffers = new System.Collections.Generic.List<IGpuBuffer>(MaxRowProgramBuffers);
        int Slot(IGpuBuffer b)
        {
            for (int i = 0; i < buffers.Count; i++)
                if (ReferenceEquals(buffers[i], b)) return i;
            buffers.Add(b);
            return buffers.Count - 1;
        }
        var words = new int[stages.Length];
        for (int k = 0; k < stages.Length; k++)
        {
            var s = stages[k];
            if (s.Rows != rows || s.Cols != cols) return false;
            int a = Slot(s.A), b = s.B is null ? a : Slot(s.B), c = s.C is null ? a : Slot(s.C), o = Slot(s.Output);
            if (buffers.Count > MaxRowProgramBuffers) return false;
            words[k] = (int)s.Op | (a << 2) | (b << 5) | (c << 8) | (o << 11);
        }
        foreach (var s in stages)
        {
            if (s.Op != ResidentOp.Affine && s.Op != ResidentOp.AffineRelu) continue;
            foreach (var w in stages)
                if (ReferenceEquals(w.Output, s.B) || ReferenceEquals(w.Output, s.C)) return false;
        }
        plan = new RowProgramPlan(rows, cols, buffers.ToArray(), words);
        return true;
    }

    // The row program's body for the C dialects: `bufs` (the eight slots) and `w` (the stage words) are locals of the
    // kernel. Per element it is res_run_stage's arithmetic, threads striding over a row's columns the same way, so the
    // results are bit-identical to the stages run one by one.
    private static string RowBody(string gf, string fmaxf, string fmaf, string expf, string negInf, string gbar) => @"
    uint ucols = (uint)cols;
    for (uint r = grp; r < (uint)rows; r += ngrp) {
        uint base0 = r * ucols;
        for (int k = 0; k < count; k++) {
            int wd = w[k];
            int op = wd & 3;
            " + gf + "* a = bufs[(wd >> 2) & 7]; " + gf + "* b = bufs[(wd >> 5) & 7]; " + gf + "* c = bufs[(wd >> 8) & 7]; "
            + gf + @"* o = bufs[(wd >> 11) & 7];
            if (op == 2) {
                float m = " + negInf + @";
                for (uint j = lid; j < ucols; j += RES_THREADS) m = " + fmaxf + @"(m, a[base0 + j]);
                m = res_reduce(m, 1, scratch, lid);
                float sum = 0.0f;
                for (uint j = lid; j < ucols; j += RES_THREADS) sum += " + expf + @"(a[base0 + j] - m);
                sum = res_reduce(sum, 0, scratch, lid);
                float inv = 1.0f / sum;
                for (uint j = lid; j < ucols; j += RES_THREADS) o[base0 + j] = " + expf + @"(a[base0 + j] - m) * inv;
            } else {
                for (uint j = lid; j < ucols; j += RES_THREADS) {
                    uint i = base0 + j;
                    float v;
                    if (op == 3) {
                        v = a[i] + b[i];
                    } else {
                        v = " + fmaf + @"(a[i], b[j], c[j]);
                        if (op == 0) v = " + fmaxf + @"(v, 0.0f);
                    }
                    o[i] = v;
                }
            }
            // With these four ops a thread reads back only elements it wrote itself (every stage strides over a row's
            // columns the same way, and the softmax reduces through scratch, behind its own barriers), so no result
            // depends on this barrier; removing it passed every test. It stays for an op that would read another
            // thread's element of the row.
            " + gbar + @";
        }
    }
";

    private static string RowKernel(ResidentKernelDialect dialect)
    {
        switch (dialect)
        {
            case ResidentKernelDialect.OpenCl:
                return @"
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void resident_rows(__global float* b0, __global float* b1, __global float* b2, __global float* b3,
    __global float* b4, __global float* b5, __global float* b6, __global float* b7,
    const int rows, const int cols, const int count, const int unused,
    const int w0, const int w1, const int w2, const int w3, const int w4, const int w5, const int w6, const int w7)
{
    __local float scratch[RES_THREADS];
    __global float* bufs[8] = { b0, b1, b2, b3, b4, b5, b6, b7 };
    int w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    uint lid = (uint)get_local_id(0), grp = (uint)get_group_id(0), ngrp = (uint)get_num_groups(0);
" + RowBody("__global float", "fmax", "fma", "exp", "(-INFINITY)", "barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)") + "}\n";
            case ResidentKernelDialect.Metal:
                return @"
struct ResRowParams { int rows; int cols; int count; int unused; int w[8]; };

kernel void resident_rows(device float* b0 [[buffer(0)]], device float* b1 [[buffer(1)]],
    device float* b2 [[buffer(2)]], device float* b3 [[buffer(3)]], device float* b4 [[buffer(4)]],
    device float* b5 [[buffer(5)]], device float* b6 [[buffer(6)]], device float* b7 [[buffer(7)]],
    constant ResRowParams& p [[buffer(8)]],
    uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]],
    uint ngrp [[threadgroups_per_grid]])
{
    threadgroup float scratch[RES_THREADS];
    device float* bufs[8] = { b0, b1, b2, b3, b4, b5, b6, b7 };
    int w[8] = { p.w[0], p.w[1], p.w[2], p.w[3], p.w[4], p.w[5], p.w[6], p.w[7] };
    int rows = p.rows, cols = p.cols, count = p.count;
" + RowBody("device float", "fmax", "fma", "precise::exp", "(-INFINITY)",
                    "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)") + "}\n";
            default:
                return @"
extern ""C"" __global__ void __launch_bounds__(RES_THREADS) resident_rows(float* b0, float* b1, float* b2, float* b3,
    float* b4, float* b5, float* b6, float* b7, int rows, int cols, int count, int unused,
    int w0, int w1, int w2, int w3, int w4, int w5, int w6, int w7)
{
    __shared__ float scratch[RES_THREADS];
    float* bufs[8] = { b0, b1, b2, b3, b4, b5, b6, b7 };
    int w[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
    uint lid = threadIdx.x, grp = blockIdx.x, ngrp = gridDim.x;
" + RowBody("float", "fmaxf", "fmaf", "expf", "__int_as_float(0xff800000)", "__syncthreads()") + "}\n";
        }
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> s_rowShaders = new();

    /// <summary>
    /// The row program for Vulkan (GLSL) or WebGPU (WGSL), written for <paramref name="plan"/>'s chain: each stage's
    /// op and buffers are spelled out, so the shader neither branches on the op nor selects a buffer at run time (a
    /// generic shader that did both through a switch per element measured 0.80-0.90x the backend's ordinary dispatch
    /// on Vulkan). Bindings 0..n-1 are the plan's buffers, each bound once, read-write (WebGPU refuses a writable
    /// buffer bound twice); rows and cols come from <see cref="RowProgramPlan.Parameters"/>, as push constants on
    /// Vulkan and a uniform at binding n on WebGPU. Cached per dialect, buffer count and stage words.
    /// </summary>
    internal static string GetRowProgramShader(ResidentKernelDialect dialect, RowProgramPlan plan)
    {
        if (dialect != ResidentKernelDialect.Glsl && dialect != ResidentKernelDialect.Wgsl)
            throw new ArgumentException("Only GLSL and WGSL have a per-chain row shader.", nameof(dialect));
        string key = dialect + ":" + plan.Buffers.Length + ":" + string.Join(",", plan.StageWords);
        return s_rowShaders.GetOrAdd(key, _ => dialect == ResidentKernelDialect.Glsl ? BuildGlslRows(plan) : BuildWgslRows(plan));
    }

    private static string BuildGlslRows(RowProgramPlan plan)
    {
        int n = plan.Buffers.Length;
        var sb = new StringBuilder("#version 450\nlayout(local_size_x = " + GroupThreads + ") in;\n");
        // coherent, with a buffer barrier between stages: a guard for an op that would read another invocation's
        // element of the row (the four ops here read back only their own; see RowBody).
        for (int i = 0; i < n; i++)
            sb.Append("layout(set=0, binding=").Append(i).Append(") coherent buffer B").Append(i).Append(" { float d[]; } b").Append(i).Append(";\n");
        sb.Append("layout(push_constant) uniform PC { int rows; int cols; int count; int unused; int w[8]; };\n");
        sb.Append("const uint RES_THREADS = ").Append(GroupThreads).Append(@"u;
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
    for (uint r = grp; r < uint(rows); r += ngrp) {
        uint base0 = r * ucols;
");
        foreach (int word in plan.StageWords)
        {
            DecodeWord(word, out var op, out int a, out int b, out int c, out int o);
            string A = "b" + a + ".d", B = "b" + b + ".d", C = "b" + c + ".d", O = "b" + o + ".d";
            if (op == ResidentOp.RowSoftmax)
            {
                sb.Append("        {\n            float m = uintBitsToFloat(0xff800000u);\n")
                  .Append("            for (uint j = lid; j < ucols; j += RES_THREADS) m = max(m, ").Append(A).Append("[base0 + j]);\n")
                  .Append("            m = res_reduce(m, true, lid);\n            float sum = 0.0;\n")
                  .Append("            for (uint j = lid; j < ucols; j += RES_THREADS) sum += exp(").Append(A).Append("[base0 + j] - m);\n")
                  .Append("            sum = res_reduce(sum, false, lid);\n            float inv = 1.0 / sum;\n")
                  .Append("            for (uint j = lid; j < ucols; j += RES_THREADS) ").Append(O).Append("[base0 + j] = exp(")
                  .Append(A).Append("[base0 + j] - m) * inv;\n        }\n");
            }
            else
            {
                string value = op == ResidentOp.Add
                    ? A + "[i] + " + B + "[i]"
                    : op == ResidentOp.AffineRelu
                        ? "max(fma(" + A + "[i], " + B + "[j], " + C + "[j]), 0.0)"
                        : "fma(" + A + "[i], " + B + "[j], " + C + "[j])";
                sb.Append("        for (uint j = lid; j < ucols; j += RES_THREADS) { uint i = base0 + j; ")
                  .Append(O).Append("[i] = ").Append(value).Append("; }\n");
            }
            sb.Append("        memoryBarrierBuffer(); barrier();\n");
        }
        sb.Append("    }\n}\n");
        return sb.ToString();
    }

    private static string BuildWgslRows(RowProgramPlan plan)
    {
        int n = plan.Buffers.Length;
        var sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.Append("@group(0) @binding(").Append(i).Append(") var<storage, read_write> b").Append(i).Append(" : array<f32>;\n");
        sb.Append("struct Params { rows : i32, cols : i32, count : i32, unused : i32, w0 : vec4<i32>, w1 : vec4<i32> };\n");
        sb.Append("@group(0) @binding(").Append(n).Append(") var<uniform> p : Params;\n");
        sb.Append("const RES_THREADS : u32 = ").Append(GroupThreads).Append("u;\nvar<workgroup> scratch : array<f32, ").Append(GroupThreads).Append(@">;

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
fn main(@builtin(local_invocation_id) l : vec3<u32>, @builtin(workgroup_id) wg : vec3<u32>,
        @builtin(num_workgroups) nw : vec3<u32>) {
    let lid = l.x;
    let ucols = u32(p.cols);
    for (var r : u32 = wg.x; r < u32(p.rows); r = r + nw.x) {
        let base0 = r * ucols;
");
        foreach (int word in plan.StageWords)
        {
            DecodeWord(word, out var op, out int a, out int b, out int c, out int o);
            string A = "b" + a, B = "b" + b, C = "b" + c, O = "b" + o;
            if (op == ResidentOp.RowSoftmax)
            {
                // WGSL may not hold -inf even as a constant: the lowest finite f32 is the same start.
                sb.Append("        {\n            var m : f32 = -0x1.fffffep+127f;\n")
                  .Append("            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { m = max(m, ").Append(A).Append("[base0 + j]); }\n")
                  .Append("            m = res_reduce(m, true, lid);\n            var sum : f32 = 0.0;\n")
                  .Append("            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { sum = sum + exp(").Append(A).Append("[base0 + j] - m); }\n")
                  .Append("            sum = res_reduce(sum, false, lid);\n            let inv = 1.0 / sum;\n")
                  .Append("            for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { ").Append(O).Append("[base0 + j] = exp(")
                  .Append(A).Append("[base0 + j] - m) * inv; }\n        }\n");
            }
            else
            {
                string value = op == ResidentOp.Add
                    ? A + "[i] + " + B + "[i]"
                    : op == ResidentOp.AffineRelu
                        ? "max(fma(" + A + "[i], " + B + "[j], " + C + "[j]), 0.0)"
                        : "fma(" + A + "[i], " + B + "[j], " + C + "[j])";
                sb.Append("        for (var j : u32 = lid; j < ucols; j = j + RES_THREADS) { let i = base0 + j; ")
                  .Append(O).Append("[i] = ").Append(value).Append("; }\n");
            }
            // A guard for an op that would read another invocation's element of the row (see RowBody).
            sb.Append("        storageBarrier();\n        workgroupBarrier();\n");
        }
        sb.Append("    }\n}\n");
        return sb.ToString();
    }

    private static void DecodeWord(int word, out ResidentOp op, out int a, out int b, out int c, out int o)
    {
        op = (ResidentOp)(word & 3);
        a = (word >> 2) & 7; b = (word >> 5) & 7; c = (word >> 8) & 7; o = (word >> 11) & 7;
    }

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
    /// <para>With <paramref name="rowKernelOnly"/> the module defines only <see cref="RowProgramKernel"/> (its HIP form is
    /// also valid CUDA C++): CUDA compiles it beside its own stage and program kernels, which share their names with
    /// HIP's.</para>
    internal static string GetModuleSource(ResidentKernelDialect dialect, bool rowKernelOnly = false)
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
        if (!rowKernelOnly) switch (dialect)
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
        sb.Append(RowKernel(dialect));
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
            // WGSL may not hold -inf even as a constant expression: the lowest finite f32 is the same start for any
            // row with a finite element.
            var m : f32 = -0x1.fffffep+127f;
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
