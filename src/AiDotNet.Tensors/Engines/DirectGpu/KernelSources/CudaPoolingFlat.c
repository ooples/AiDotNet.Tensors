
// Max pool over a flat output index: one thread per output cell, batch * channels * outHeight * outWidth cells.
// maxpool2d launches a 16x16 block per (batch, channel) plane, so a plane of a few outputs (VGG's 2x2 and 1x1 pools)
// left all but a few of each block's 256 threads idle across thousands of blocks. The window rule, winner index and
// output layout are maxpool2d_at's (appended to the pooling module, so pool_take is the same function).
extern "C" __global__ __launch_bounds__(256) void maxpool2d_flat(
    const float* __restrict__ input, float* __restrict__ output, int* __restrict__ indices,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW, int saveIndices)
{
    long long total = (long long)batch * channels * outHeight * outWidth;
    for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += (long long)gridDim.x * blockDim.x)
    {
        int ow = (int)(o % outWidth);
        long long r = o / outWidth;
        int oh = (int)(r % outHeight);
        long long plane = r / outHeight;
        const float* in = input + plane * inHeight * inWidth;

        unsigned int maxBits = 0xFF800000u;
        bool have = false;
        int firstH = min(max(oh * strideH - padH, 0), inHeight - 1);
        int firstW = min(max(ow * strideW - padW, 0), inWidth - 1);
        int maxIdx = firstH * inWidth + firstW;
        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH - padH + kh;
            if (ih < 0 || ih >= inHeight) continue;
            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW - padW + kw;
                if (iw < 0 || iw >= inWidth) continue;
                unsigned int val = __float_as_uint(in[ih * inWidth + iw]);
                if (pool_take(have, val, maxBits)) {
                    maxBits = val;
                    maxIdx = ih * inWidth + iw;
                    have = true;
                }
            }
        }
        output[o] = __uint_as_float(maxBits);
        if (saveIndices) indices[o] = maxIdx;
    }
}

// avgpool2d_at over a flat output index, for the same small planes (DenseNet's transition pools). Same summation
// order and divisor, so the results are bitwise avgpool2d's.
extern "C" __global__ __launch_bounds__(256) void avgpool2d_flat(
    const float* __restrict__ input, float* __restrict__ output,
    int batch, int channels, int inHeight, int inWidth,
    int outHeight, int outWidth, int kernelH, int kernelW,
    int strideH, int strideW, int padH, int padW, int countIncludePad)
{
    long long total = (long long)batch * channels * outHeight * outWidth;
    for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += (long long)gridDim.x * blockDim.x)
    {
        int ow = (int)(o % outWidth);
        long long r = o / outWidth;
        int oh = (int)(r % outHeight);
        long long plane = r / outHeight;
        const float* in = input + plane * inHeight * inWidth;
        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ih = oh * strideH - padH + kh;
                int iw = ow * strideW - padW + kw;
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    sum += in[ih * inWidth + iw];
                    count++;
                } else if (countIncludePad) {
                    count++;
                }
            }
        }
        int divisor = countIncludePad ? (kernelH * kernelW) : count;
        output[o] = sum / (float)max(divisor, 1);
    }
}
