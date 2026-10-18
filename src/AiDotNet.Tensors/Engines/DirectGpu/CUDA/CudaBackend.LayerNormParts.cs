using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// LayerNorm's backward as placeable pieces (see ILayerNormBackwardPartsBackend), and the same deterministic
// column reduction for bias gradients (IColumnSumBackend).
public sealed partial class CudaBackend : ILayerNormBackwardPartsBackend, IColumnSumBackend, IBroadcastReducedBackend,
    ISumAroundKeptBackend
{
    // column_sum_small serves matrices up to this many rows (32 row lanes, so at most 32 loads per thread); taller ones
    // keep the chunked partial/finish pair, which spreads the rows over ~128 blocks.
    private const int ColumnSumSmallMaxRows = 1024;

    int IColumnSumBackend.ColumnSumScratchFloats(int rows, int cols) =>
        rows <= 0 || cols <= 0 ? 0 : checked(LayerNormAffineGradientChunks(rows, cols) * cols);

    unsafe bool IColumnSumBackend.TryColumnSums(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output,
        int rows, int cols, bool add)
    {
        if (!IsAvailable || rows <= 0 || cols <= 0 || input.Size < (long)rows * cols || output.Size < cols
            || output.Handle == input.Handle)
            return false;
        if (rows <= ColumnSumSmallMaxRows && _kernelCache.TryGetValue("column_sum_small", out var smallKernel))
        {
            using var __ = PushContext();
            IntPtr sx = input.Handle, so = output.Handle;
            int sr = rows, sc = cols, sad = add ? 1 : 0;
            void** sargs = stackalloc void*[5];
            sargs[0] = &sx; sargs[1] = &so; sargs[2] = &sr; sargs[3] = &sc; sargs[4] = &sad;
            LaunchKernel2D(smallKernel, (uint)((cols + 31) / 32), 1, 32, 32, sargs);
            return true;
        }
        if (!_kernelCache.TryGetValue("column_sum_partial", out var partialKernel)
            || !_kernelCache.TryGetValue("column_sum_finish", out var finishKernel))
            return false;
        int chunks = LayerNormAffineGradientChunks(rows, cols);
        if (scratch.Size < (long)chunks * cols) return false;
        int rowsPerChunk = (rows + chunks - 1) / chunks;

        using var _ = PushContext();
        IntPtr x = input.Handle, part = scratch.Handle, o = output.Handle;
        int r = rows, c = cols, rpc = rowsPerChunk, ch = chunks, ad = add ? 1 : 0;
        void** pargs = stackalloc void*[5];
        pargs[0] = &x; pargs[1] = &part; pargs[2] = &r; pargs[3] = &c; pargs[4] = &rpc;
        LaunchKernel2D(partialKernel, (uint)((cols + 31) / 32), (uint)chunks, 32, 8, pargs);
        void** fargs = stackalloc void*[5];
        fargs[0] = &part; fargs[1] = &o; fargs[2] = &c; fargs[3] = &ch; fargs[4] = &ad;
        LaunchKernel(finishKernel, (uint)((cols + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, fargs);
        return true;
    }

    unsafe bool IBroadcastReducedBackend.TryBroadcastMiddle(IGpuBuffer g, IGpuBuffer output, int outer, int reduced,
        int inner, float scale, bool add)
    {
        long total = (long)outer * reduced * inner;
        if (!IsAvailable || outer <= 0 || reduced <= 0 || inner <= 0 || g.Size < (long)outer * inner
            || output.Size < total || output.Handle == g.Handle
            || ((g.Handle.ToInt64() | output.Handle.ToInt64()) & 15) != 0
            || !_kernelCache.TryGetValue("broadcast_reduced_middle", out var broadcastKernel))
            return false;
        using var _ = PushContext();
        IntPtr gp = g.Handle, op = output.Handle;
        int o = outer, r = reduced, i = inner, ad = add ? 1 : 0;
        float sc = scale;
        void** args = stackalloc void*[7];
        args[0] = &gp; args[1] = &op; args[2] = &o; args[3] = &r; args[4] = &i; args[5] = &sc; args[6] = &ad;
        long work = (inner & 3) == 0 ? total / 4 : total;
        uint blocks = (uint)Math.Min(65535L, Math.Max(1L, (work + DefaultBlockSize - 1) / DefaultBlockSize));
        LaunchKernel(broadcastKernel, blocks, DefaultBlockSize, args);
        return true;
    }
    // Chunks of the leading block per kept channel for sum_around_kept: ~256 blocks in all, at least one row each.
    private static int SumAroundKeptChunks(int leading, int kept) =>
        Math.Max(1, Math.Min(leading, (256 + kept - 1) / Math.Max(1, kept)));

    int ISumAroundKeptBackend.SumAroundKeptScratchFloats(int leading, int kept, int trailing) =>
        leading <= 0 || kept <= 0 || trailing <= 0 ? 0 : checked(SumAroundKeptChunks(leading, kept) * kept);

    unsafe bool ISumAroundKeptBackend.TrySumAroundKept(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output,
        int leading, int kept, int trailing, float scale)
    {
        long n = (long)leading * kept * trailing;
        if (!IsAvailable || leading <= 0 || kept <= 0 || trailing <= 0 || kept > 65535 || input.Size < n
            || output.Size < kept || output.Handle == input.Handle
            || !_kernelCache.TryGetValue("sum_around_kept_partial", out var aroundPartialKernel)
            || !_kernelCache.TryGetValue("sum_around_kept_finish", out var aroundFinishKernel))
            return false;
        int chunks = SumAroundKeptChunks(leading, kept);
        if (scratch.Size < (long)chunks * kept) return false;
        int rowsPerChunk = (leading + chunks - 1) / chunks;

        using var _ = PushContext();
        IntPtr x = input.Handle, part = scratch.Handle, o = output.Handle;
        int l = leading, c = kept, t = trailing, rpc = rowsPerChunk, ch = chunks;
        float sc = scale;
        void** pargs = stackalloc void*[6];
        pargs[0] = &x; pargs[1] = &part; pargs[2] = &l; pargs[3] = &c; pargs[4] = &t; pargs[5] = &rpc;
        LaunchKernel2D(aroundPartialKernel, (uint)kept, (uint)chunks, 1, DefaultBlockSize, 1, pargs);
        void** fargs = stackalloc void*[5];
        fargs[0] = &part; fargs[1] = &o; fargs[2] = &c; fargs[3] = &ch; fargs[4] = &sc;
        LaunchKernel(aroundFinishKernel, (uint)((kept + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, fargs);
        return true;
    }
    // Chunks of rows for the affine-gradient reduction: enough blocks (~128 with the column tiles) to occupy the
    // device, and at least 32 rows per chunk (4 per row lane) so a chunk's partial is worth its write.
    private static int LayerNormAffineGradientChunks(int rows, int cols)
    {
        int columnTiles = (cols + 31) / 32;
        int wanted = Math.Max(1, 128 / columnTiles);
        int most = Math.Max(1, (rows + 31) / 32);
        return Math.Min(wanted, most);
    }

    // layernorm_backward_rows holds each row in registers: at most 32 lanes x 16 columns.
    private const int LayerNormRowBackwardMaxColumns = 512;

    // Chunks for layernorm_backward_rows: one row per warp (8 per block) until there are ~128 blocks, then more rows
    // per chunk.
    private static int LayerNormRowBackwardChunks(int rows) => Math.Max(1, Math.Min(128, (rows + 7) / 8));

    int ILayerNormBackwardPartsBackend.LayerNormRowBackwardScratchFloats(int rows, int cols) =>
        rows <= 0 || cols <= 0 || cols > LayerNormRowBackwardMaxColumns ? 0
            : checked(2 * LayerNormRowBackwardChunks(rows) * cols);

    unsafe bool ILayerNormBackwardPartsBackend.TryLayerNormRowBackward(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer variance, IGpuBuffer scratch, IGpuBuffer gradInput,
        IGpuBuffer gradGamma, IGpuBuffer gradBeta, int rows, int cols, float epsilon, bool addInput, bool addGamma,
        bool addBeta)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || cols > LayerNormRowBackwardMaxColumns
            || gradOutput.Size < n || input.Size < n || gradInput.Size < n || gamma.Size < cols
            || mean.Size < rows || variance.Size < rows || gradGamma.Size < cols || gradBeta.Size < cols
            || gradGamma.Handle == gradBeta.Handle
            || !_kernelCache.TryGetValue("layernorm_grad_params_finish_warp", out var rowsFinishKernel))
            return false;
        IntPtr rowsKernel;
        bool found = cols <= 64 ? _kernelCache.TryGetValue("layernorm_backward_rows2", out rowsKernel)
            : cols <= 128 ? _kernelCache.TryGetValue("layernorm_backward_rows4", out rowsKernel)
            : cols <= 256 ? _kernelCache.TryGetValue("layernorm_backward_rows8", out rowsKernel)
            : _kernelCache.TryGetValue("layernorm_backward_rows16", out rowsKernel);
        int chunks = LayerNormRowBackwardChunks(rows);
        if (!found || scratch.Size < 2L * chunks * cols) return false;
        int rowsPerChunk = (rows + chunks - 1) / chunks;

        using var _ = PushContext();
        IntPtr g = gradOutput.Handle, x = input.Handle, ga = gamma.Handle, mu = mean.Handle, va = variance.Handle,
            dx = gradInput.Handle, part = scratch.Handle;
        int r = rows, c = cols, rpc = rowsPerChunk, ai = addInput ? 1 : 0;
        float eps = epsilon;
        void** rargs = stackalloc void*[12];
        rargs[0] = &g; rargs[1] = &x; rargs[2] = &ga; rargs[3] = &mu; rargs[4] = &va; rargs[5] = &eps;
        rargs[6] = &dx; rargs[7] = &part; rargs[8] = &r; rargs[9] = &c; rargs[10] = &rpc; rargs[11] = &ai;
        LaunchKernel(rowsKernel, (uint)chunks, DefaultBlockSize, rargs);

        IntPtr gg = gradGamma.Handle, gb = gradBeta.Handle;
        int ch = chunks, ag = addGamma ? 1 : 0, ab = addBeta ? 1 : 0;
        void** fargs = stackalloc void*[7];
        fargs[0] = &part; fargs[1] = &gg; fargs[2] = &gb; fargs[3] = &c; fargs[4] = &ch; fargs[5] = &ag; fargs[6] = &ab;
        // One warp per column: 8 columns per 256-thread block.
        LaunchKernel(rowsFinishKernel, (uint)((cols + 7) / 8), DefaultBlockSize, fargs);
        return true;
    }

    int ILayerNormBackwardPartsBackend.LayerNormAffineGradientScratchFloats(int rows, int cols) =>
        rows <= 0 || cols <= 0 ? 0 : checked(2 * LayerNormAffineGradientChunks(rows, cols) * cols);

    unsafe bool ILayerNormBackwardPartsBackend.TryLayerNormInverseStd(IGpuBuffer variance, IGpuBuffer inverseStd,
        int rows, float epsilon)
    {
        if (!IsAvailable || rows <= 0 || variance.Size < rows || inverseStd.Size < rows
            || !_kernelCache.TryGetValue("layernorm_inverse_std", out var inverseStdKernel))
            return false;
        using var _ = PushContext();
        IntPtr v = variance.Handle, s = inverseStd.Handle;
        int n = rows;
        float eps = epsilon;
        void** args = stackalloc void*[4];
        args[0] = &v; args[1] = &s; args[2] = &n; args[3] = &eps;
        LaunchKernel(inverseStdKernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        return true;
    }

    unsafe bool ILayerNormBackwardPartsBackend.TryLayerNormVarianceFromInverseStd(IGpuBuffer values, int rows, float epsilon)
    {
        if (!IsAvailable || rows <= 0 || values.Size < rows
            || !_kernelCache.TryGetValue("layernorm_variance_from_inverse_std", out var varianceKernel))
            return false;
        using var _ = PushContext();
        IntPtr v = values.Handle;
        int n = rows;
        float eps = epsilon;
        void** args = stackalloc void*[3];
        args[0] = &v; args[1] = &n; args[2] = &eps;
        LaunchKernel(varianceKernel, (uint)((rows + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
        return true;
    }

    bool ILayerNormBackwardPartsBackend.TryLayerNormInputGradient(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer gradInput, int rows, int cols)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || gradOutput.Size < n || input.Size < n || gradInput.Size < n
            || gamma.Size < cols || mean.Size < rows || inverseStd.Size < rows)
            return false;
        using var _ = PushContext();
        // As LayerNormBackward's input-gradient half: the tuned registry's choice, else the reference kernel.
        var args = new CudaNormBackwardArgs(gradOutput, input, gamma, mean, inverseStd, gradInput, null, rows, cols);
        if (!LayerNormBackwardSlot.TryExecute(RowShape(rows, cols), args))
            LaunchLayerNormBackwardReference(args);
        return true;
    }

    bool ILayerNormBackwardPartsBackend.SupportsLayerNormInputGradientAdd =>
        IsAvailable && _kernelCache.ContainsKey("layernorm_input_grad_add");

    unsafe bool ILayerNormBackwardPartsBackend.TryLayerNormInputGradientAdd(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer gradInput, int rows, int cols)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || gradOutput.Size < n || input.Size < n || gradInput.Size < n
            || gamma.Size < cols || mean.Size < rows || inverseStd.Size < rows
            || !_kernelCache.TryGetValue("layernorm_input_grad_add", out var addKernel))
            return false;
        using var _ = PushContext();
        IntPtr g = gradOutput.Handle, x = input.Handle, ga = gamma.Handle, mu = mean.Handle, inv = inverseStd.Handle,
            dx = gradInput.Handle;
        int r = rows, cl = cols;
        void** args = stackalloc void*[8];
        args[0] = &g; args[1] = &x; args[2] = &ga; args[3] = &mu; args[4] = &inv; args[5] = &dx; args[6] = &r; args[7] = &cl;
        // One warp per row: 8 rows per 256-thread block.
        LaunchKernel(addKernel, (uint)((rows + 7) / 8), DefaultBlockSize, args);
        return true;
    }
    unsafe bool ILayerNormBackwardPartsBackend.TryLayerNormAffineGradients(IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer mean, IGpuBuffer inverseStd, IGpuBuffer scratch, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int rows, int cols, bool addGamma, bool addBeta)
    {
        long n = (long)rows * cols;
        if (!IsAvailable || rows <= 0 || cols <= 0 || gradOutput.Size < n || input.Size < n
            || mean.Size < rows || inverseStd.Size < rows || gradGamma.Size < cols || gradBeta.Size < cols
            || gradGamma.Handle == gradBeta.Handle
            || !_kernelCache.TryGetValue("layernorm_grad_params_partial", out var partialKernel)
            || !_kernelCache.TryGetValue("layernorm_grad_params_finish", out var finishKernel))
            return false;
        int chunks = LayerNormAffineGradientChunks(rows, cols);
        if (scratch.Size < 2L * chunks * cols) return false;
        int rowsPerChunk = (rows + chunks - 1) / chunks;

        using var _ = PushContext();
        IntPtr g = gradOutput.Handle, x = input.Handle, mu = mean.Handle, inv = inverseStd.Handle, part = scratch.Handle;
        int r = rows, c = cols, rpc = rowsPerChunk;
        void** pargs = stackalloc void*[8];
        pargs[0] = &g; pargs[1] = &x; pargs[2] = &mu; pargs[3] = &inv; pargs[4] = &part;
        pargs[5] = &r; pargs[6] = &c; pargs[7] = &rpc;
        LaunchKernel2D(partialKernel, (uint)((cols + 31) / 32), (uint)chunks, 32, 8, pargs);

        IntPtr gg = gradGamma.Handle, gb = gradBeta.Handle;
        int ch = chunks, ag = addGamma ? 1 : 0, ab = addBeta ? 1 : 0;
        void** fargs = stackalloc void*[7];
        fargs[0] = &part; fargs[1] = &gg; fargs[2] = &gb; fargs[3] = &c; fargs[4] = &ch; fargs[5] = &ag; fargs[6] = &ab;
        LaunchKernel(finishKernel, (uint)((cols + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, fargs);
        return true;
    }
}
