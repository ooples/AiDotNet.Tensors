// copy_many8: up to eight independent device-to-device copies in one launch (CudaBackend.CopyMany).
// blockIdx.y picks the copy; blocks stride over its elements. A captured graph records one kernel node
// instead of one memcpy node per copy - a stateful plan's trailing state writes, for example.
// The copies must not overlap one another.
extern "C" __global__ __launch_bounds__(256) void copy_many8(
    const float* s0, const float* s1, const float* s2, const float* s3,
    const float* s4, const float* s5, const float* s6, const float* s7,
    float* d0, float* d1, float* d2, float* d3, float* d4, float* d5, float* d6, float* d7,
    int n0, int n1, int n2, int n3, int n4, int n5, int n6, int n7)
{
    int k = blockIdx.y;
    const float* s = k == 0 ? s0 : k == 1 ? s1 : k == 2 ? s2 : k == 3 ? s3 : k == 4 ? s4 : k == 5 ? s5 : k == 6 ? s6 : s7;
    float* d = k == 0 ? d0 : k == 1 ? d1 : k == 2 ? d2 : k == 3 ? d3 : k == 4 ? d4 : k == 5 ? d5 : k == 6 ? d6 : d7;
    int n = k == 0 ? n0 : k == 1 ? n1 : k == 2 ? n2 : k == 3 ? n3 : k == 4 ? n4 : k == 5 ? n5 : k == 6 ? n6 : n7;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = s[i];
}
