// Typed kernels of the CUDA generic FFT module (CudaFFTGenericKernels), kept out of the user-string heap (CS8103).
// @SUFFIX@ is the element type suffix; STORE_T, aidn_ld and aidn_st are the module's.

// Narrowing and widening fused with a layout change: [batch][rows][cols] -> [batch][cols][rows] (rows = C and cols = H*W
// is NCHW -> NHWC; the reverse call is NHWC -> NCHW). A mixed-precision convolution runs channels-last, the layout
// cuDNN's 16-bit tensor-core kernels read natively, so one pass both converts and transposes instead of a conversion
// here and cuDNN's own layout pass. 32x32 tiles through shared memory (padded against bank conflicts), block 32x8,
// grid (ceil(cols/32), ceil(rows/32), batch).
extern "C" __global__ __launch_bounds__(AIDN_FFTG_THREADS_PER_BLOCK) void fftg_transpose_from_f32@SUFFIX@(
    const float* src, STORE_T* dst, int rows, int cols)
{
    __shared__ float tile[32][33];
    size_t plane = (size_t)rows * cols;
    const float* s = src + blockIdx.z * plane;
    STORE_T* d = dst + blockIdx.z * plane;
    int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += 8)
    {
        int r = r0 + i, c = c0 + threadIdx.x;
        if (r < rows && c < cols) tile[i][threadIdx.x] = s[(size_t)r * cols + c];
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += 8)
    {
        int c = c0 + i, r = r0 + threadIdx.x;
        if (c < cols && r < rows) d[(size_t)c * rows + r] = aidn_st(tile[threadIdx.x][i]);
    }
}

extern "C" __global__ __launch_bounds__(AIDN_FFTG_THREADS_PER_BLOCK) void fftg_transpose_to_f32@SUFFIX@(
    const STORE_T* src, float* dst, int rows, int cols)
{
    __shared__ float tile[32][33];
    size_t plane = (size_t)rows * cols;
    const STORE_T* s = src + blockIdx.z * plane;
    float* d = dst + blockIdx.z * plane;
    int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += 8)
    {
        int r = r0 + i, c = c0 + threadIdx.x;
        if (r < rows && c < cols) tile[i][threadIdx.x] = aidn_ld(s[(size_t)r * cols + c]);
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += 8)
    {
        int c = c0 + i, r = r0 + threadIdx.x;
        if (c < cols && r < rows) d[(size_t)c * rows + r] = tile[threadIdx.x][i];
    }
}

// Channels-last rows re-strided between channel counts: dst[r][c] = src[r][c] where r < srcRows and c < srcC, else 0.
// Padding a channel count to the tensor cores' multiple of 8 (zeros contribute nothing to a convolution's sums), and
// with srcC > dstC dropping the padding again. Rows past srcRows (padded filters) are zero.
extern "C" __global__ __launch_bounds__(AIDN_FFTG_THREADS_PER_BLOCK) void fftg_channel_pad@SUFFIX@(
    const STORE_T* src, STORE_T* dst, int dstRows, int srcRows, int srcC, int dstC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dstRows * dstC) return;
    int r = idx / dstC, c = idx - r * dstC;
    dst[idx] = (r < srcRows && c < srcC) ? src[(size_t)r * srcC + c] : (STORE_T)0;
}
