namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    public unsafe void CosineSimilarity(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
    {
        if (batchSize <= 0) return;
        var kernel = ResolveParity210Kernel("cosine_similarity");
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outputPtr;
        args[3] = &batchSize; args[4] = &dim;
        LaunchKernel(kernel, (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void PairwiseDistance(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => LaunchScientificPairwise("parity210_cdist_l2", a, b, output, m, n, dim);

    public unsafe void PairwiseDistanceSquared(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
        => LaunchScientificPairwise("pairwise_distance_squared", a, b, output, m, n, dim);

    private unsafe void LaunchScientificPairwise(
        string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
        int m, int n, int dim)
    {
        long totalLong = (long)m * n;
        if (totalLong <= 0) return;
        if (totalLong > int.MaxValue)
            throw new OverflowException(
                $"Pairwise-distance work-item count {totalLong} exceeds Int32.MaxValue.");
        int total = (int)totalLong;
        var kernel = ResolveParity210Kernel(kernelName);
        IntPtr aPtr = a.Handle, bPtr = b.Handle, outputPtr = output.Handle;
        void** args = stackalloc void*[6];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outputPtr;
        args[3] = &m; args[4] = &n; args[5] = &dim;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }
}
