#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A 1-D dispatch past WebGPU's 65,535 workgroups per dimension runs folded on a 2-D grid (WebGpuDispatchFold), on
/// Dawn through the shipped webgpu-tensors.js (<see cref="DawnHostedWebGpu"/>; skipped unless AIDOTNET_WEBGPU_NODE_DIR
/// is set up). Before the fold these dispatches were refused, and the engine left the GPU for them.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class WebGpuDispatchFoldTests
{
    // 777 elements past 65,535 workgroups of 256: two grid rows, and a partial last workgroup.
    private const int Elements = 65535 * 256 + 777;

    [SkippableFact]
    public void ElementwiseKernels_PastTheWorkgroupLimit_CoverEveryElement()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        float[] a = new float[Elements], b = new float[Elements];
        for (int i = 0; i < Elements; i++) { a[i] = (i % 1013) * 0.25f - 100f; b[i] = (i % 769) * 0.5f; }
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dc = backend.AllocateBuffer(Elements);

        backend.Fill(dc, float.NaN, Elements);
        backend.Add(da, db, dc, Elements);
        float[] sum = backend.DownloadBuffer(dc);
        for (int i = 0; i < Elements; i++)
            if (sum[i] != a[i] + b[i]) Assert.Fail($"sum[{i}]: expected {a[i] + b[i]}, got {sum[i]}");

        backend.Fill(dc, 3.5f, Elements);
        float[] filled = backend.DownloadBuffer(dc);
        for (int i = 0; i < Elements; i++)
            if (filled[i] != 3.5f) Assert.Fail($"fill[{i}]: expected 3.5, got {filled[i]}");
    }

    // More rows than one dimension holds take the workgroup-per-row kernel (it used to fall back to one thread per
    // row). Small integers keep every sum exact, whatever the reduction order.
    [SkippableFact]
    public void SumAxis_PastTheWorkgroupLimit_SumsEveryRow()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        const int rows = 70001, cols = WebGpuBackend.SumAxisRowsMinReduce;
        var a = new float[rows * cols];
        for (int i = 0; i < a.Length; i++) a[i] = i % 7 - 3;
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(rows);
        backend.Fill(db, float.NaN, rows);
        backend.SumAxis(da, db, rows, cols);
        float[] sums = backend.DownloadBuffer(db);
        for (int r = 0; r < rows; r++)
        {
            float expected = 0;
            for (int c = 0; c < cols; c++) expected += a[r * cols + c];
            if (sums[r] != expected) Assert.Fail($"row {r}: expected {expected}, got {sums[r]}");
        }
    }

    // One workgroup per output, as the row kernels index them: workgroup_id, num_workgroups, a workgroup barrier and
    // global_invocation_id must all read as they would in one (groups, 1, 1) dispatch, and the grid's surplus workgroup
    // must not run (it would write the two sentinel words past the last group's).
    private const string PerWorkgroup = @"
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
var<workgroup> s: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    s[lid.x] = gid.x;
    workgroupBarrier();
    if (lid.x == 0u) {
        var acc = 0u;
        for (var i = 0u; i < 64u; i = i + 1u) { acc = acc + s[i]; }
        o[2u * wid.x] = acc;
        o[2u * wid.x + 1u] = n.x + gid.y + gid.z + wid.y + wid.z;
    }
}";

    [SkippableFact]
    public void PerWorkgroupKernel_PastTheWorkgroupLimit_SeesOneDimensionalBuiltins()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        const int groups = 70001;   // folds to 35001 x 2: one surplus workgroup
        // A quiet NaN: the host round trip quiets a signalling one (0x7F80_1234 came back 0x7FC0_1234 on Dawn), which
        // would read as overwritten although nothing touched it.
        var sentinel = BitConverter.Int32BitsToSingle(0x7FC0_1234);
        var init = new float[2 * groups + 2];
        Array.Fill(init, sentinel);
        using var output = backend.AllocateBuffer(init);

        int module = WebGpuNativeBindings.CreateShaderModule(PerWorkgroup);
        Assert.True(module >= 0, "the shader did not compile");
        int pipeline = WebGpuNativeBindings.CreateComputePipelineAsync(module, "main").GetAwaiter().GetResult();
        Assert.True(pipeline >= 0, "the pipeline was not created");
        int bindGroup = WebGpuNativeBindings.CreateBindGroup(pipeline, new[] { ((WebGpuBuffer)output).BufferId });
        try
        {
            WebGpuNativeBindings.DispatchComputeAsync(pipeline, bindGroup, groups, 1, 1).GetAwaiter().GetResult();
            WebGpuNativeBindings.SubmitAndWaitAsync().GetAwaiter().GetResult();
            float[] back = backend.DownloadBuffer(output);
            for (int w = 0; w < groups; w++)
            {
                uint acc = (uint)BitConverter.SingleToInt32Bits(back[2 * w]);
                uint count = (uint)BitConverter.SingleToInt32Bits(back[2 * w + 1]);
                if (acc != 64u * 64u * (uint)w + 2016u) Assert.Fail($"workgroup {w}: global ids summed to {acc}");
                if (count != groups) Assert.Fail($"workgroup {w}: num_workgroups.x + the y and z builtins read {count}, not {groups}");
            }
            Assert.Equal(0x7FC0_1234, BitConverter.SingleToInt32Bits(back[2 * groups]));
            Assert.Equal(0x7FC0_1234, BitConverter.SingleToInt32Bits(back[2 * groups + 1]));
        }
        finally
        {
            WebGpuNativeBindings.DestroyBindGroup(bindGroup);
            WebGpuNativeBindings.DestroyPipeline(pipeline);
            WebGpuNativeBindings.DestroyShaderModule(module);
        }
    }
}
#endif
