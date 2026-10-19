namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// The resident program (#890 phase 3): a chain of declared stages run by ONE persistent kernel, every block walking
/// the stage list in order with a grid-wide barrier between stages, so the chain pays one launch instead of one per
/// stage. A small chain is launch-bound: each separate launch costs more than the stage's work.
/// <para>
/// <c>resident_stage</c> runs a single stage with the same device code, so a chain launched stage by stage computes
/// bit-identical results and the comparison isolates what residency removes. A row softmax keeps each row inside one
/// block (a fixed block size and reduction order), so neither kernel's result depends on the grid.
/// </para>
/// <para>
/// The barrier counts arrivals in <c>bar[0]</c> and publishes a new generation in <c>bar[1]</c>; the last arrival
/// resets <c>bar[0]</c>, so the two words are reusable launch after launch once zeroed at allocation. The spin wait is
/// safe because a cooperative launch guarantees every block is co-resident.
/// </para>
/// <para>
/// CUDA only for now; the other backends are tracked per backend, since each needs a different design for the
/// barrier: HIP #1448 (cooperative launch, a direct port), OpenCL #1449, Vulkan #1450, Metal #1451 and WebGPU #1452
/// (no cross-work-group forward-progress guarantee, so one submission of ordered dispatches instead of a spin barrier).
/// </para>
/// </summary>
internal static class CudaResidentProgramKernels
{
    internal const string ProgramKernel = "resident_program";
    internal const string StageKernel = "resident_stage";

    /// <summary>Threads per block of both kernels; a softmax row is reduced by one block of this size.</summary>
    internal const int BlockThreads = 256;

    /// <summary>The most stages one program holds (the stage array travels in the kernel's parameter block).</summary>
    internal const int MaxStages = 8;

    internal static string[] GetKernelNames() => new[] { ProgramKernel, StageKernel };

    internal static string GetSource() => @"
struct ResStage {
    int op; int rows; int cols; int pad;
    const float* a; const float* b; const float* c; float* out;
};

struct ResProgram {
    ResStage stages[" + MaxStages + @"];
    int count; int pad0; unsigned int* bar;
};

#define RES_OP_AFFINE_RELU 0
#define RES_OP_AFFINE      1
#define RES_OP_ROW_SOFTMAX 2
#define RES_OP_ADD         3
#define RES_THREADS " + BlockThreads + @"

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

// Block-wide reduction of one value per thread; every thread receives the result. Fixed order for a fixed block size.
__device__ __forceinline__ float res_block_reduce(float v, bool isMax, float* scratch)
{
    for (int o = 16; o > 0; o >>= 1) {
        float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = isMax ? fmaxf(v, other) : v + other;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < (RES_THREADS / 32) ? scratch[lane] : (isMax ? __int_as_float(0xff800000) : 0.0f);
        for (int o = 16; o > 0; o >>= 1) {
            float other = __shfl_xor_sync(0xffffffffu, v, o);
            v = isMax ? fmaxf(v, other) : v + other;
        }
        if (lane == 0) scratch[0] = v;
    }
    __syncthreads();
    float r = scratch[0];
    __syncthreads();
    return r;
}

// One stage over the whole tensor, the work split over the launch's blocks: elementwise ops by a grid-stride loop,
// the row softmax one row per block (a block strides over rows).
__device__ void res_run_stage(const ResStage& s, float* scratch)
{
    // 64-bit: a stage may hold up to int.MaxValue elements, and a 32-bit start index or stride step wraps negative
    // within one grid of the end, passing the i < n test and addressing out of bounds.
    long long n = (long long)s.rows * s.cols;
    if (s.op == RES_OP_ROW_SOFTMAX) {
        for (int r = blockIdx.x; r < s.rows; r += gridDim.x) {
            const float* in = s.a + (long long)r * s.cols;
            float* out = s.out + (long long)r * s.cols;
            float m = __int_as_float(0xff800000);
            for (int j = threadIdx.x; j < s.cols; j += RES_THREADS) m = fmaxf(m, in[j]);
            m = res_block_reduce(m, true, scratch);
            float sum = 0.0f;
            for (int j = threadIdx.x; j < s.cols; j += RES_THREADS) sum += expf(in[j] - m);
            sum = res_block_reduce(sum, false, scratch);
            float inv = 1.0f / sum;
            for (int j = threadIdx.x; j < s.cols; j += RES_THREADS) out[j] = expf(in[j] - m) * inv;
        }
        return;
    }
    long long stride = (long long)gridDim.x * RES_THREADS;
    for (long long i = (long long)blockIdx.x * RES_THREADS + threadIdx.x; i < n; i += stride) {
        float v;
        if (s.op == RES_OP_ADD) {
            v = s.a[i] + s.b[i];
        } else {
            int col = (int)(i % s.cols);
            v = fmaf(s.a[i], s.b[col], s.c[col]);
            if (s.op == RES_OP_AFFINE_RELU) v = fmaxf(v, 0.0f);
        }
        s.out[i] = v;
    }
}

extern ""C"" __global__ void __launch_bounds__(RES_THREADS) resident_program(ResProgram p)
{
    __shared__ float scratch[RES_THREADS / 32];
    for (int k = 0; k < p.count; k++) {
        res_run_stage(p.stages[k], scratch);
        if (k + 1 < p.count) res_grid_barrier(p.bar);
    }
}

extern ""C"" __global__ void __launch_bounds__(RES_THREADS) resident_stage(ResStage s)
{
    __shared__ float scratch[RES_THREADS / 32];
    res_run_stage(s, scratch);
}
";
}
