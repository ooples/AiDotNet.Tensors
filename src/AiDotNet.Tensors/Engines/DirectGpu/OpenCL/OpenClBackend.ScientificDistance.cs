namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        public void CosineSimilarity(
            IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int batchSize, int dim)
        {
            if (_context == null || batchSize <= 0) return;
            var kernel = _kernelCache["cosine_similarity"];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(2, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3, batchSize);
            kernel.SetArg(4, dim);
            kernel.Execute1D(batchSize, CalculateOptimalWorkGroupSize1D(batchSize));
        }

        public void PairwiseDistance(
            IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
            => ExecuteScientificPairwise("pairwise_distance", a, b, output, m, n, dim);

        public void PairwiseDistanceSquared(
            IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int m, int n, int dim)
            => ExecuteScientificPairwise("pairwise_distance_squared", a, b, output, m, n, dim);

        private void ExecuteScientificPairwise(
            string kernelName, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
            int m, int n, int dim)
        {
            int total = m * n;
            if (_context == null || total <= 0) return;
            var kernel = _kernelCache[kernelName];
            kernel.SetArg(0, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
            kernel.SetArg(1, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
            kernel.SetArg(2, ((DirectOpenClGpuBuffer)output).Buffer.Handle);
            kernel.SetArg(3, m);
            kernel.SetArg(4, n);
            kernel.SetArg(5, dim);
            kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
        }
    }
}
