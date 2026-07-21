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
