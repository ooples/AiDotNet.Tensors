// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// The first and last nodes of a captured training step whose host side is plain memory traffic
/// (<see cref="CudaBackend"/>'s graph step I/O). Outside a graph, every launch, copy and event record a step issues costs
/// the host a driver call; on the parity LSTM those calls (input refresh, optimizer, loss readback) were ~140 us of a
/// ~160 us step while the graph itself ran in ~48 us.
/// </summary>
/// <remarks>
/// The host owns two page-locked staging slots per step input and fills slot <c>r % 2</c> for replay <c>r</c>; the device
/// keeps the replay count itself (<c>state[0]</c>), so the two sides agree on the slot without any per-step call. The
/// prologue copies the slot into the graph's input buffers (the device reads the page-locked memory directly) and, when
/// the optimizer is captured too, picks this step's learning rate from the slot's table by the device's own count of
/// applied steps (<c>state[1]</c>): the host does not wait for the previous steps' finiteness verdicts to choose it. The
/// epilogue publishes the step's loss and gradient sum of squares to page-locked memory, advances the applied count only
/// for a finite step, then the replay count, and writes the replay count to the progress word last (after a system
/// fence), so a host that sees progress r can read every result of replays below r.
/// </remarks>
internal static class CudaStepIoKernels
{
    // Must match StepIoMaxInputs / the StepIoArgs layout in CudaBackend.StepIo.cs.
    public const int MaxInputs = 16;

    public static string GetSource() => @"
#define SIO_MAX 16

struct StepIoArgs
{
    unsigned long long src0[SIO_MAX];   // page-locked slot 0 per input
    unsigned long long src1[SIO_MAX];   // page-locked slot 1 per input
    unsigned long long dst[SIO_MAX];    // the graph's device input buffers
    int start[SIO_MAX + 1];             // prefix sums of the inputs' copy units (4 floats when vector, else 1)
    int count;
    unsigned long long lrTable0;        // page-locked [base step (int), n (int), lr[n] (float)] for slot 0, or 0
    unsigned long long lrTable1;        // the same for slot 1
    unsigned long long dynScalars;      // device float[2]: this step's learning rate, step number (as float)
    unsigned long long state;           // device unsigned int[4]: replays done, finite steps applied, fault, unused
    int len[SIO_MAX];                   // the inputs' float counts
    int vector;                         // 1: every address is 16-byte aligned, copy float4 units
    unsigned long long guard;           // device double: the captured update's finiteness guard, or 0
};

// grid-stride over every input float; block 0 thread 0 also resolves the learning rate.
extern ""C"" __global__ __launch_bounds__(256) void step_io_prologue(StepIoArgs a)
{
    unsigned int* state = (unsigned int*)a.state;
    unsigned int slot = state[0] & 1u;
    int total = a.start[a.count];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        int t = 0;
        while (t + 1 < a.count && a.start[t + 1] <= i) t++;
        const float* src = (const float*)(slot ? a.src1[t] : a.src0[t]);
        float* dst = (float*)a.dst[t];
        int k = i - a.start[t];
        if (a.vector) {
            // Reads of page-locked host memory cross the bus: four floats per request.
            int e = k * 4, len = a.len[t];
            if (e + 4 <= len) *reinterpret_cast<float4*>(dst + e) = *reinterpret_cast<const float4*>(src + e);
            else for (; e < len; e++) dst[e] = src[e];
        } else {
            dst[k] = src[k];
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && a.dynScalars != 0) {
        const int* table = (const int*)(slot ? a.lrTable1 : a.lrTable0);
        int step = (int)state[1] + 1;
        int idx = step - table[0];
        int n = table[1];
        // The host's table always covers the device's step number; one that does not is a bookkeeping fault, reported
        // through the published results. The step is not run with a neighbouring step's rate: the guard the gradients'
        // sum of squares accumulates onto is poisoned, so the update kernels skip it and the epilogue does not count it.
        if (idx < 0 || idx >= n) {
            state[2] = 1u;
            idx = idx < 0 ? 0 : n - 1;
            if (a.guard != 0) *(double*)a.guard = __longlong_as_double(0x7ff8000000000000LL);
        }
        float* dyn = (float*)a.dynScalars;
        dyn[0] = ((const float*)(table + 2))[idx];
        dyn[1] = (float)step;
    }
}

struct StepIoResults
{
    unsigned int progress;      // replays complete; written last
    unsigned int applied;       // finite steps applied after that replay
    unsigned int fault;         // nonzero: a learning-rate table did not cover the step number
    unsigned int unused;
    double sumSquares[2];       // the gradients' sum of squares, per slot
    float loss[2][64];          // the loss values, per slot
};

// One warp. guard: the gradients' double sum of squares on the device (0: no captured optimizer); reset to zero after
// it is read, for the next replay's reduction to accumulate onto.
extern ""C"" __global__ __launch_bounds__(32) void step_io_epilogue(
    unsigned long long stateAddress, unsigned long long resultsAddress,
    double* __restrict__ guard, const float* __restrict__ loss, int lossCount)
{
    unsigned int* state = (unsigned int*)stateAddress;
    volatile StepIoResults* results = (volatile StepIoResults*)resultsAddress;
    unsigned int slot = state[0] & 1u;
    for (int i = threadIdx.x; i < lossCount; i += 32) results->loss[slot][i] = loss[i];
    // Every lane's loss writes reach the host before lane 0 publishes the progress that says they are there.
    __threadfence_system();
    __syncwarp();
    if (threadIdx.x == 0) {
        double sum = guard != 0 ? *guard : 0.0;
        if (guard != 0) *guard = 0.0;
        results->sumSquares[slot] = sum;
        if (guard != 0 && isfinite(sum)) state[1] += 1u;
        unsigned int done = state[0] + 1u;
        state[0] = done;
        results->applied = state[1];
        results->fault = state[2];
        __threadfence_system();
        results->progress = done;
    }
}
";

    public static string[] GetKernelNames() => new[] { "step_io_prologue", "step_io_epilogue" };
}
