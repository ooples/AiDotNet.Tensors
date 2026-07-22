using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Small deterministic targets for Nsight Compute release evidence.</summary>
internal static class DirectPtxProfileTarget
{
    internal static void RunAttention()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-attention-start");
        using var runtime = new DirectPtxRuntime();
        foreach (int sequence in new[] { 16, 32, 64, 128 })
        foreach (bool causal in new[] { false, true })
        foreach (bool fused in new[] { false, true })
        {
            using var kernel = new PtxOnlineFusedAttention128x64Kernel(
                runtime, 128, causal, fused,
                sequenceLength: sequence, emitSoftmaxStats: false);
            using var q = runtime.AllocateBytes(kernel.QBytes);
            using var k = runtime.AllocateBytes(kernel.KBytes);
            using var v = runtime.AllocateBytes(kernel.VBytes);
            using var gamma = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
            using var beta = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);
            var input = new ushort[128 * sequence * 64];
            q.Upload<ushort>(input);
            k.Upload<ushort>(input);
            v.Upload<ushort>(input);
            gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
            beta.Upload<float>(new float[64]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]),
                default);
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        runtime.Synchronize();
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-attention-end");
    }

    internal static void RunResidualRmsNorm()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-residual-rmsnorm-start");
        using var runtime = new DirectPtxRuntime();
        foreach (int rows in new[] { 32, 256, 2048, 8192 })
        {
            using var kernel = new PtxFusedResidualRmsNormD64Kernel(runtime, rows);
            using var input = runtime.AllocateBytes(kernel.InputBytes);
            using var residual = runtime.AllocateBytes(kernel.InputBytes);
            using var gamma = runtime.AllocateBytes(PtxFusedResidualRmsNormD64Kernel.GammaBytes);
            using var output = runtime.AllocateBytes(kernel.OutputBytes);
            using var rms = runtime.AllocateBytes(kernel.RmsBytes);
            input.Upload<float>(new float[rows * 64]);
            residual.Upload<float>(new float[rows * 64]);
            gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(rms, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-residual-rmsnorm-end");
    }

    internal static void RunDecode()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-decode-start");
        using var runtime = new DirectPtxRuntime();
        const int heads = 8, kvHeads = 1, sequence = 128, blockSize = 16, poolBlocks = 10;
        using var dense = new PtxFusedDecodeAttentionD64Kernel(
            runtime, false, heads, kvHeads, sequence, 0, 0, 0.125f);
        using var paged = new PtxFusedDecodeAttentionD64Kernel(
            runtime, true, heads, kvHeads, sequence, blockSize, poolBlocks, 0.125f);
        using var q = runtime.AllocateBytes(dense.QueryBytes);
        using var denseK = runtime.AllocateBytes(dense.KeyValueBytes);
        using var denseV = runtime.AllocateBytes(dense.KeyValueBytes);
        using var denseOutput = runtime.AllocateBytes(dense.OutputBytes);
        using var pagedK = runtime.AllocateBytes(paged.KeyValueBytes);
        using var pagedV = runtime.AllocateBytes(paged.KeyValueBytes);
        using var table = runtime.AllocateBytes(paged.BlockTableBytes);
        using var pagedOutput = runtime.AllocateBytes(paged.OutputBytes);
        q.Upload<float>(new float[heads * 64]);
        denseK.Upload<float>(new float[sequence * kvHeads * 64]);
        denseV.Upload<float>(new float[sequence * kvHeads * 64]);
        pagedK.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        pagedV.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        table.Upload<int>(Enumerable.Range(0, sequence / blockSize).Reverse().ToArray());
        Action launchDense = () => dense.LaunchDense(
            DirectPtxTensorView.CreateOwned(q, dense.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(denseK, dense.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(denseV, dense.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(denseOutput, dense.Blueprint.Tensors[3]));
        Action launchPaged = () => paged.LaunchPaged(
            DirectPtxTensorView.CreateOwned(q, paged.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(pagedK, paged.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(pagedV, paged.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(table, paged.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(pagedOutput, paged.Blueprint.Tensors[4]));
        launchDense();
        launchPaged();
        runtime.Synchronize();
        Console.WriteLine(dense.Audit.ToJson());
        Console.WriteLine(paged.Audit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-decode-end");
    }

    internal static void RunPagedPrefill()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-paged-prefill-start");
        using var runtime = new DirectPtxRuntime();
        const int heads = 8, kvHeads = 2, queries = 16, start = 112;
        const int blockSize = 16, poolBlocks = 10;
        using var kernel = new PtxFusedPagedPrefillAttentionD64Kernel(
            runtime, heads, kvHeads, queries, start,
            blockSize, poolBlocks, 0.125f);
        using var q = runtime.AllocateBytes(kernel.QueryBytes);
        using var k = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var v = runtime.AllocateBytes(kernel.KeyValueBytes);
        using var table = runtime.AllocateBytes(kernel.BlockTableBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        q.Upload<float>(new float[queries * heads * 64]);
        k.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        v.Upload<float>(new float[poolBlocks * blockSize * kvHeads * 64]);
        table.Upload<int>(Enumerable.Range(2, 8).Reverse().ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(table, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-paged-prefill-end");
    }

    internal static void RunAttentionBackward()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-attention-backward-start");
        using var runtime = new DirectPtxRuntime();
        const int batch = 1, heads = 8, kvHeads = 2, querySequence = 64, keySequence = 64;
        using var kernel = new PtxFusedAttentionBackwardD64Kernel(
            runtime, batch, heads, kvHeads, querySequence, keySequence, 0.125f);
        using var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var key = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var value = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var probabilities = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var gradQuery = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        using var gradKey = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
        using var gradValue = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
        gradOutput.Upload<float>(new float[batch * heads * querySequence * 64]);
        query.Upload<float>(new float[batch * heads * querySequence * 64]);
        key.Upload<float>(new float[batch * kvHeads * keySequence * 64]);
        value.Upload<float>(new float[batch * kvHeads * keySequence * 64]);
        probabilities.Upload<float>(Enumerable.Repeat(
            1f / keySequence, batch * heads * querySequence * keySequence).ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(probabilities, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[7]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.RowDeltaAudit.ToJson());
        Console.WriteLine(kernel.GradKeyValueAudit.ToJson());
        Console.WriteLine(kernel.GradQueryAudit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-attention-backward-end");
    }

    internal static void RunFlashAttentionBackward()
    {
        GpuBenchmarkEnvironment.RequireIdleGpu("ncu-flash-attention-backward-start");
        using var runtime = new DirectPtxRuntime();
        const int batch = 1, heads = 8, querySequence = 64, keySequence = 64;
        using var kernel = new PtxFlashAttentionBackwardD64Kernel(
            runtime, batch, heads, querySequence, keySequence,
            0.125f, isCausal: true);
        using var gradOutput = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        using var key = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
        using var value = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
        using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
        using var stats = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
        using var gradQuery = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
        using var gradKey = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
        using var gradValue = runtime.AllocateBytes(kernel.Blueprint.Tensors[8].RequiredBytes);
        gradOutput.Upload<float>(new float[batch * heads * querySequence * 64]);
        query.Upload<float>(new float[batch * heads * querySequence * 64]);
        key.Upload<float>(new float[batch * heads * keySequence * 64]);
        value.Upload<float>(new float[batch * heads * keySequence * 64]);
        output.Upload<float>(new float[batch * heads * querySequence * 64]);
        stats.Upload<float>(Enumerable.Repeat(
            MathF.Log(keySequence), batch * heads * querySequence).ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(key, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(value, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(stats, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(gradQuery, kernel.Blueprint.Tensors[6]),
            DirectPtxTensorView.CreateOwned(gradKey, kernel.Blueprint.Tensors[7]),
            DirectPtxTensorView.CreateOwned(gradValue, kernel.Blueprint.Tensors[8]));
        launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.GradQueryAudit.ToJson());
        Console.WriteLine(kernel.GradKeyValueAudit.ToJson());
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-flash-attention-backward-end");
    }

    internal static void RunQkvRopeCache()
    {
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-qkv-rope-cache-start");
        using var runtime = new DirectPtxRuntime();
        (int Heads, int Capacity, int Position)[] shapes =
        [
            (4, 16, 0),
            (8, 64, 17),
            (16, 128, 127)
        ];
        foreach ((int heads, int capacity, int position) in shapes)
        {
            using var kernel = new PtxFusedQkvRopeCacheD64Kernel(
                runtime, heads, capacity, position);
            int model = heads * PtxFusedQkvRopeCacheD64Kernel.HeadDimension;
            using var input = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var weights = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var bias = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            using var cosine = runtime.AllocateBytes(kernel.Blueprint.Tensors[3].RequiredBytes);
            using var sine = runtime.AllocateBytes(kernel.Blueprint.Tensors[4].RequiredBytes);
            using var query = runtime.AllocateBytes(kernel.Blueprint.Tensors[5].RequiredBytes);
            using var keyCache = runtime.AllocateBytes(kernel.Blueprint.Tensors[6].RequiredBytes);
            using var valueCache = runtime.AllocateBytes(kernel.Blueprint.Tensors[7].RequiredBytes);
            input.Upload<float>(new float[model]);
            weights.Upload<float>(new float[3 * model * model]);
            bias.Upload<float>(new float[3 * model]);
            cosine.Upload<float>(Enumerable.Repeat(1f, capacity * 32).ToArray());
            sine.Upload<float>(new float[capacity * 32]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(cosine, kernel.Blueprint.Tensors[3]),
                DirectPtxTensorView.CreateOwned(sine, kernel.Blueprint.Tensors[4]),
                DirectPtxTensorView.CreateOwned(query, kernel.Blueprint.Tensors[5]),
                DirectPtxTensorView.CreateOwned(keyCache, kernel.Blueprint.Tensors[6]),
                DirectPtxTensorView.CreateOwned(valueCache, kernel.Blueprint.Tensors[7]));
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-qkv-rope-cache-end");
    }

    internal static void RunVisionBoxIou()
    {
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-vision-box-iou-start");
        using var runtime = new DirectPtxRuntime();
        (int N, int M)[] shapes = [(256, 256), (1024, 256), (1024, 1024), (4096, 256)];
        foreach ((int n, int m) in shapes)
        {
            using var kernel = new PtxFusedPairwiseBoxIouF32Kernel(runtime, n, m);
            using var boxesA = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
            using var boxesB = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
            using var output = runtime.AllocateBytes(kernel.Blueprint.Tensors[2].RequiredBytes);
            boxesA.Upload<float>(new float[n * 4]);
            boxesB.Upload<float>(new float[m * 4]);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(boxesA, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(boxesB, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            Console.WriteLine(kernel.Audit.ToJson());
        }
        foreach (DirectPtxVisionSpec spec in VisionProfileSpecs())
        {
            using var kernel = new PtxVisionKernel(runtime, spec);
            var buffers = new DirectPtxBuffer[kernel.Blueprint.Tensors.Count];
            try
            {
                var views = new DirectPtxTensorView[buffers.Length];
                for (int i = 0; i < buffers.Length; i++)
                {
                    DirectPtxTensorContract contract = kernel.Blueprint.Tensors[i];
                    buffers[i] = runtime.AllocateBytes(contract.RequiredBytes);
                    if (contract.PhysicalType == DirectPtxPhysicalType.Int32)
                        buffers[i].Upload<int>(new int[checked((int)(contract.RequiredBytes / 4))]);
                    else
                        buffers[i].Upload<float>(new float[checked((int)(contract.RequiredBytes / 4))]);
                    views[i] = DirectPtxTensorView.CreateOwned(buffers[i], contract);
                }
                kernel.Launch(
                    views[0],
                    views.Length > 1 ? views[1] : default,
                    views.Length > 2 ? views[2] : default,
                    views.Length > 3 ? views[3] : default,
                    views.Length > 4 ? views[4] : default,
                    views.Length > 5 ? views[5] : default);
                runtime.Synchronize();
                Console.WriteLine(kernel.Audit.ToJson());
            }
            finally
            {
                foreach (DirectPtxBuffer? buffer in buffers) buffer?.Dispose();
            }
        }
        GpuBenchmarkEnvironment.RequireNoForeignCompute("ncu-vision-box-iou-end");
    }

    private static IEnumerable<DirectPtxVisionSpec> VisionProfileSpecs()
    {
        yield return new(DirectPtxVisionOperation.GeneralizedBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.DistanceBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.CompleteBoxIou, 256, 256);
        yield return new(DirectPtxVisionOperation.BoxArea, 256);
        yield return new(DirectPtxVisionOperation.BoxConvert, 256, 0, 2);
        yield return new(DirectPtxVisionOperation.IoULoss, 256);
        yield return new(DirectPtxVisionOperation.GIoULoss, 256);
        yield return new(DirectPtxVisionOperation.DIoULoss, 256);
        yield return new(DirectPtxVisionOperation.CIoULoss, 256);
        yield return new(DirectPtxVisionOperation.IoULossBackward, 256);
        yield return new(DirectPtxVisionOperation.GIoULossBackward, 256);
        yield return new(DirectPtxVisionOperation.DIoULossBackward, 256);
        yield return new(DirectPtxVisionOperation.CIoULossBackward, 256);
        yield return new(DirectPtxVisionOperation.IouFamilyBackwardA, 256, 256, 0);
        yield return new(DirectPtxVisionOperation.IouFamilyBackwardB, 256, 256, 0);
        yield return new(DirectPtxVisionOperation.Nms, 256,
            Flags: 0, ScalarBits: BitConverter.SingleToInt32Bits(0.5f));
        yield return new(DirectPtxVisionOperation.Nms, 256,
            Flags: 1, ScalarBits: BitConverter.SingleToInt32Bits(0.5f));
        yield return new(DirectPtxVisionOperation.MasksToBoxes, 256, 28, 28);
        yield return new(DirectPtxVisionOperation.RoiAlign,
            1, 256, 56, 56, 256, 7, 7, 256, 2 | 0x100,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.RoiPool,
            1, 256, 56, 56, 256, 7, 7, 256, 0,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.PsRoiAlign,
            1, 196, 56, 56, 256, 7, 7, 4, 2,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.PsRoiPool,
            1, 196, 56, 56, 256, 7, 7, 4, 0,
            BitConverter.SingleToInt32Bits(0.25f));
        yield return new(DirectPtxVisionOperation.Cross3, 256, 1);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 0);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 1);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 2);
        yield return new(DirectPtxVisionOperation.Meshgrid2D, 256, 256, Flags: 3);
    }

    internal static void VerifyNcuCsv(string path)
    {
        DirectPtxProfilerEvidence evidence = DirectPtxProfilerEvidence.FromNcuCsv(path);
        Console.WriteLine($"source: {evidence.Source}");
        Console.WriteLine($"register-spill instructions: {evidence.RegisterSpillInstructions}");
        Console.WriteLine($"local-load instructions: {evidence.LocalLoadInstructions}");
        Console.WriteLine($"local-store instructions: {evidence.LocalStoreInstructions}");
        Console.WriteLine($"required metric groups observed: {evidence.ObservedMetricGroups}/3");
        Console.WriteLine(evidence.ProvesZeroExecutedSpills ? "PASS: zero executed spills" : "FAIL: spill/local traffic detected");
        if (!evidence.ProvesZeroExecutedSpills) Environment.ExitCode = 2;
    }
}
