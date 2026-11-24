// A convolution's epilogue from its channels-last result: dst[n, c, s] = act(src[n, s, c] + bias[c]) for an NHWC src
// (what cuDNN's tensor-core kernels write natively) and an NCHW dst. The layout change cuDNN otherwise runs as its own
// pass after an NCHW convolution is this kernel's load and store, so the bias add and the activation cost no pass of
// their own. Each block moves a 32 x 32 (spatial x channel) tile through shared memory: both the load (over channels)
// and the store (over spatial) are coalesced. The values are conv2d_bias_add's sum and the relu / relu6 kernels' own
// operations, so the result equals the convolution, then conv2d_bias_add, then the activation pass.
// act: 0 none, 1 ReLU, 2 ReLU6. bias may be null. Grid (ceil(spatial / 32), ceil(channels / 32), batch), block (32, 8).
extern "C" __global__ __launch_bounds__(256) void conv_epilogue_nhwc_nchw(const float* __restrict__ src,
    const float* __restrict__ bias, float* __restrict__ dst, int channels, int spatial, int act)
{
    __shared__ float tile[32][33];
    const long long image = (long long)blockIdx.z * channels * spatial;
    const int s0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
    #pragma unroll
    for (int r = threadIdx.y; r < 32; r += 8)
    {
        const int s = s0 + r, c = c0 + threadIdx.x;
        if (s < spatial && c < channels) tile[r][threadIdx.x] = src[image + (long long)s * channels + c];
    }
    __syncthreads();
    #pragma unroll
    for (int r = threadIdx.y; r < 32; r += 8)
    {
        const int c = c0 + r, s = s0 + threadIdx.x;
        if (c < channels && s < spatial)
        {
            float v = tile[threadIdx.x][r];
            if (bias != nullptr) v += __ldg(&bias[c]);
            if (act == 1) v = relu_nan_safe(v);
            else if (act == 2) v = xf_clampf(v, 0.0f, 6.0f);
            dst[image + (long long)c * spatial + s] = v;
        }
    }
}

// A convolution's per-channel epilogue in one pass (IChannelEpilogueBackend): y = act(((z + bias[c]) * scale[c]) +
// shift[c]) over NCHW, each of bias / scale / shift optional (null skips it). Each step is rounded on its own (__fadd_rn
// and __fmul_rn: no fused multiply-add), and the activations are the relu (relu_nan_safe) and leaky_relu kernels'
// expressions, so the values equal the separate conv2d_bias_add, broadcast multiply, broadcast add and activation passes
// bit for bit. act: 0 none, 1 ReLU, 2 LeakyReLU(slope). Grid (ceil(spatial / 256), min(rows, 65535)), rows = batch *
// channels; the y blocks stride over rows past the grid.
extern "C" __global__ __launch_bounds__(256) void channel_epilogue_fwd(const float* __restrict__ z,
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift,
    float* __restrict__ y, int rows, int channels, int spatial, int act, float slope)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        const int c = row % channels;
        const long long i = (long long)row * spatial + s;
        float v = z[i];
        if (bias != nullptr) v = __fadd_rn(v, __ldg(&bias[c]));
        if (scale != nullptr) v = __fmul_rn(v, __ldg(&scale[c]));
        if (shift != nullptr) v = __fadd_rn(v, __ldg(&shift[c]));
        if (act == 1) v = relu_nan_safe(v);
        else if (act == 2) v = v > 0.0f ? v : __fmul_rn(slope, v);
        y[i] = v;
    }
}

// The epilogue's backward to z: dz = act'(y) * dy * scale[c], the mask from the forward's output (its input's sign for
// ReLU and a positive LeakyReLU slope), as relu_backward / leaky_relu_backward and the broadcast multiply's backward
// compute it. Grid as channel_epilogue_fwd.
extern "C" __global__ __launch_bounds__(256) void channel_epilogue_bwd(const float* __restrict__ dy,
    const float* __restrict__ y, const float* __restrict__ scale, float* __restrict__ dz, int rows, int channels,
    int spatial, int act, float slope)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y)
    {
        const int c = row % channels;
        const long long i = (long long)row * spatial + s;
        float g = dy[i];
        if (act == 1) g = y[i] > 0.0f ? g : 0.0f;
        else if (act == 2) g = y[i] > 0.0f ? g : __fmul_rn(slope, g);
        if (scale != nullptr) g = __fmul_rn(g, __ldg(&scale[c]));
        dz[i] = g;
    }
}

// The epilogue over a flat index, for a small spatial extent (a channels-last activation is [rows, C] with spatial 1):
// element i is in channel (i / spatial) % channels, the same arithmetic as channel_epilogue_fwd / _bwd.
extern "C" __global__ __launch_bounds__(256) void channel_epilogue_flat_fwd(const float* __restrict__ z,
    const float* __restrict__ bias, const float* __restrict__ scale, const float* __restrict__ shift,
    float* __restrict__ y, long long count, int channels, int spatial, int act, float slope)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (long long)gridDim.x * blockDim.x)
    {
        const int c = (int)((i / spatial) % channels);
        float v = z[i];
        if (bias != nullptr) v = __fadd_rn(v, __ldg(&bias[c]));
        if (scale != nullptr) v = __fmul_rn(v, __ldg(&scale[c]));
        if (shift != nullptr) v = __fadd_rn(v, __ldg(&shift[c]));
        if (act == 1) v = relu_nan_safe(v);
        else if (act == 2) v = v > 0.0f ? v : __fmul_rn(slope, v);
        y[i] = v;
    }
}

extern "C" __global__ __launch_bounds__(256) void channel_epilogue_flat_bwd(const float* __restrict__ dy,
    const float* __restrict__ y, const float* __restrict__ scale, float* __restrict__ dz, long long count, int channels,
    int spatial, int act, float slope)
{
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (long long)gridDim.x * blockDim.x)
    {
        const int c = (int)((i / spatial) % channels);
        float g = dy[i];
        if (act == 1) g = y[i] > 0.0f ? g : 0.0f;
        else if (act == 2) g = y[i] > 0.0f ? g : __fmul_rn(slope, g);
        if (scale != nullptr) g = __fmul_rn(g, __ldg(&scale[c]));
        dz[i] = g;
    }
}

// The layout change of each sample's [rows, cols] matrix to [cols, rows] through a 32 x 33 shared tile (grid: col
// tiles, row tiles, samples; 32 x 8 threads): NCHW -> NHWC with rows = C, cols = H*W, and back with them swapped.
extern "C" __global__ __launch_bounds__(256) void layout_transpose_tiles(const float* __restrict__ input,
    float* __restrict__ output, int rows, int cols)
{
    __shared__ float tile[32][33];
    const long long sample = (long long)blockIdx.z * rows * cols;
    const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += 8)
    {
        const int r = r0 + j, c = c0 + threadIdx.x;
        if (r < rows && c < cols) tile[j][threadIdx.x] = input[sample + (long long)r * cols + c];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8)
    {
        const int c = c0 + j, r = r0 + threadIdx.x;
        if (r < rows && c < cols) output[sample + (long long)c * rows + r] = tile[threadIdx.x][j];
    }
}
