using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>Small deterministic targets for Nsight Compute release evidence.</summary>
internal static class DirectPtxProfileTarget
{
    internal static void RunAttention()
    {
        using var runtime = new DirectPtxRuntime();
        using var kernel = new PtxOnlineFusedAttention128x64Kernel(
            runtime, 128, isCausal: false, fuseLayerNormGelu: false,
            sequenceLength: 128, emitSoftmaxStats: false);
        using var q = runtime.AllocateBytes(kernel.QBytes);
        using var k = runtime.AllocateBytes(kernel.KBytes);
        using var v = runtime.AllocateBytes(kernel.VBytes);
        using var gamma = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.GammaBytes);
        using var beta = runtime.AllocateBytes(PtxOnlineFusedAttention128x64Kernel.BetaBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        using var stats = runtime.AllocateBytes(kernel.StatsBytes);
        q.Upload<ushort>(new ushort[128 * 128 * 64]);
        k.Upload<ushort>(new ushort[128 * 128 * 64]);
        v.Upload<ushort>(new ushort[128 * 128 * 64]);
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(q, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(k, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(v, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(beta, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[5]),
            DirectPtxTensorView.CreateOwned(stats, kernel.Blueprint.Tensors[6]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunResidualRmsNorm()
    {
        using var runtime = new DirectPtxRuntime();
        using var kernel = new PtxFusedResidualRmsNormD64Kernel(runtime, 8192);
        using var input = runtime.AllocateBytes(kernel.InputBytes);
        using var residual = runtime.AllocateBytes(kernel.InputBytes);
        using var gamma = runtime.AllocateBytes(PtxFusedResidualRmsNormD64Kernel.GammaBytes);
        using var output = runtime.AllocateBytes(kernel.OutputBytes);
        using var rms = runtime.AllocateBytes(kernel.RmsBytes);
        input.Upload<float>(new float[8192 * 64]);
        residual.Upload<float>(new float[8192 * 64]);
        gamma.Upload<float>(Enumerable.Repeat(1f, 64).ToArray());
        Action launch = () => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(residual, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gamma, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(rms, kernel.Blueprint.Tensors[4]));
        for (int i = 0; i < 10; i++) launch();
        runtime.Synchronize();
        Console.WriteLine(kernel.Audit.ToJson());
    }

    internal static void RunDecode()
    {
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
        for (int i = 0; i < 10; i++)
        {
            launchDense();
            launchPaged();
        }
        runtime.Synchronize();
        Console.WriteLine(dense.Audit.ToJson());
        Console.WriteLine(paged.Audit.ToJson());
    }

    internal static void VerifyNcuCsv(string path)
    {
        DirectPtxProfilerEvidence evidence = DirectPtxProfilerEvidence.FromNcuCsv(path);
        Console.WriteLine($"source: {evidence.Source}");
        Console.WriteLine($"register-spill instructions: {evidence.RegisterSpillInstructions}");
        Console.WriteLine($"local-load instructions: {evidence.LocalLoadInstructions}");
        Console.WriteLine($"local-store instructions: {evidence.LocalStoreInstructions}");
        Console.WriteLine($"local-spill requests: {evidence.LocalSpillRequests}");
        Console.WriteLine($"required metric groups observed: {evidence.ObservedMetricGroups}/4");
        Console.WriteLine(evidence.ProvesZeroExecutedSpills ? "PASS: zero executed spills" : "FAIL: spill/local traffic detected");
        if (!evidence.ProvesZeroExecutedSpills) Environment.ExitCode = 2;
    }
}
