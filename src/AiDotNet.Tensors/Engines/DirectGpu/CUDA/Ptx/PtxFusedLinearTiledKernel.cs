using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The fused activation applied by <see cref="PtxFusedLinearTiledKernel"/> in the
/// epilogue, after the bias add and before the final global store. One tiled GEMM
/// serves every activation; only the emitted epilogue differs, which is what lets
/// this single kernel own the GemmBias* and FusedLinear* families for M&gt;1.
/// </summary>
internal enum DirectPtxLinearActivation
{
    /// <summary>Raw GEMM + bias (GemmBias).</summary>
    None,
    Relu,
    GeluTanh,
    Sigmoid,
    Tanh,
    Swish,
    LeakyRelu
}

/// <summary>
/// General-M FP32 fused linear + bias + activation, computed as a real
/// register-blocked, shared-memory-staged tiled GEMM rather than the warp-reduction
/// dot product of the M=1 decode kernel. Computes
/// <c>C[M,N] = activation(A[M,K] @ transpose(W[N,K]) + bias[N])</c> where A is
/// row-major input tokens and W is canonical output-major weights (row n contiguous
/// over K, identical to the M=1 contract), so a block loads one BM×BK tile of A and
/// one BN×BK tile of W into shared memory each K-step and every thread accumulates a
/// TM×TN micro-tile entirely in registers.
///
/// Tile: BM=BN=64, BK=8, TM=TN=4 → 256 threads, 16 FP32 accumulators/thread, 4 KiB
/// shared (2 KiB A + 2 KiB W). Grid = (N/BN, M/BM). This is the correct fused
/// foundation; cp.async double-buffering and FP16 Tensor-Core tiles are the tracked
/// performance follow-ups.
/// </summary>
internal sealed class PtxFusedLinearTiledKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_fused_linear_tiled";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearTiledKernel(
        DirectPtxRuntime runtime,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in tiled fused-linear specialization is measured only on GA10x/SM86.");
        ValidateShape(m, k, n);

        M = m;
        K = k;
        N = n;
        Activation = activation;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n, activation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n, activation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, weights) || Overlaps(output, bias))
            throw new ArgumentException("Fused-linear output may not alias input, weights, or bias.");

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &outputPointer;
        _module.Launch(
            _function,
            (uint)(N / BlockN), (uint)(M / BlockM), 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        ValidateShape(m, k, n);
        int kBytes = checked(k * sizeof(float));
        int nBytes = checked(n * sizeof(float));

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// tiled fused-linear M={m} K={k} N={n} act={activation} " +
            $"tile={BlockM}x{BlockN}x{BlockK} thread={ThreadM}x{ThreadN}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<48>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        // A tile [BM][BK] then W tile [BN][BK], both row-major over BK.
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 w_tile[{BlockN * BlockK * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, w_tile;");

        // Thread / block coordinates.
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");                         // tid 0..255
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                        // block col
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");                        // block row
        ptx.AppendLine($"    shr.u32 %r3, %r0, 4;");                         // threadRow = tid / (BN/TN=16)
        ptx.AppendLine($"    and.b32 %r4, %r0, 15;");                        // threadCol = tid % 16
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockM};");               // blockRow*BM (base A row)
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockN};");               // blockCol*BN (base W row / N col)
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");             // threadRow*TM (A sub-row)
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");             // threadCol*TN (W sub-row)

        // Zero the 16 accumulators %f0..%f15.
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");

        // k0 loop counter in %r9 (element units), and running global-K offsets.
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("K_TILE_LOOP:");

        // ---- Cooperative global -> shared load ----
        // 512 A elements + 512 W elements over 256 threads => 2 each. Element index
        // e in {tid, tid+256}; row = e>>3 (e / BK), kk = e & 7 (e % BK).
        EmitCooperativeLoad(ptx, kBytes);
        ptx.AppendLine("    bar.sync 0;");

        // ---- Register-blocked inner product over BK ----
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");

        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");

        // ---- Epilogue: + bias[n], activation, store C[m,n] ----
        EmitEpilogue(ptx, activation, nBytes);

        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Each thread loads 2 A elements and 2 W elements into shared memory.
    private static void EmitCooperativeLoad(StringBuilder ptx, int kBytes)
    {
        // %r9 holds k0 (element units into K).
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");   // e = tid + slot*256
            // A: row = e>>3, kk = e&7. globalRow = blockRow*BM + row = %r5 + row.
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");                       // row
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");                       // kk
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");                     // global A row
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                     // global K col = k0 + kk
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");          // row * K * 4
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");                  // col * 4
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");                 // shared byte offset = e*4
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // W: same layout, row over N; globalRow = blockCol*BN + row = %r6 + row.
            ptx.AppendLine("    add.u32 %r14, %r6, %r12;");                     // global W row (N index)
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    // For each k in 0..BK-1: load TM A-fragment + TN W-fragment from shared, 16 FMAs.
    private static void EmitInnerAccumulate(StringBuilder ptx)
    {
        // A_sh row for micro-row i = (threadRow*TM + i); linear index = row*BK + k.
        // W_sh row for micro-col j = (threadCol*TN + j); linear index = row*BK + k.
        // Base shared byte pointers for this thread's fragment start (row*BK*4).
        ptx.AppendLine($"    mul.lo.u32 %r16, %r7, {BlockK};");                 // (threadRow*TM)*BK
        ptx.AppendLine($"    mul.lo.u32 %r17, %r8, {BlockK};");                 // (threadCol*TN)*BK
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");                      // &A_sh[(threadRow*TM)*BK]
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");                      // &W_sh[(threadCol*TN)*BK]
        for (int k = 0; k < BlockK; k++)
        {
            int kByte = k * sizeof(float);
            // a[i] at &A_sh + (i*BK + k)*4 ; w[j] at &W_sh + (j*BK + k)*4.
            for (int i = 0; i < ThreadM; i++)
            {
                int off = (i * BlockK + k) * sizeof(float);
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd13+{off}];");
            }
            for (int j = 0; j < ThreadN; j++)
            {
                int off = (j * BlockK + k) * sizeof(float);
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{off}];");
            }
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
            _ = kByte;
        }
    }

    private static void EmitEpilogue(StringBuilder ptx, DirectPtxLinearActivation activation, int nBytes)
    {
        // Global output row base m0 = blockRow*BM + threadRow*TM = %r5 + %r7.
        // Global output col base n0 = blockCol*BN + threadCol*TN = %r6 + %r8.
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");                          // m0
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");                          // n0
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");                    // m = m0 + i
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {nBytes};");         // m * N * 4
            ptx.AppendLine("    add.u64 %rd16, %rd3, %rd15;");                  // &C[m,0]
            for (int j = 0; j < ThreadN; j++)
            {
                int acc = i * ThreadN + j;
                ptx.AppendLine($"    add.u32 %r21, %r19, {j};");               // n = n0 + j
                ptx.AppendLine("    mul.wide.u32 %rd17, %r21, 4;");            // n * 4
                ptx.AppendLine("    add.u64 %rd18, %rd2, %rd17;");             // &bias[n]
                ptx.AppendLine("    ld.global.nc.f32 %f24, [%rd18];");
                ptx.AppendLine($"    add.rn.f32 %f{acc}, %f{acc}, %f24;");      // + bias
                EmitActivation(ptx, activation, $"%f{acc}");
                ptx.AppendLine("    add.u64 %rd19, %rd16, %rd17;");            // &C[m,n]
                ptx.AppendLine($"    st.global.f32 [%rd19], %f{acc};");
            }
        }
    }

    // Emit the activation in place on <paramref name="v"/>. Scratch: %f25..%f27.
    private static void EmitActivation(StringBuilder ptx, DirectPtxLinearActivation activation, string v)
    {
        switch (activation)
        {
            case DirectPtxLinearActivation.None:
                break;
            case DirectPtxLinearActivation.Relu:
                ptx.AppendLine($"    max.f32 {v}, {v}, 0f00000000;");
                break;
            case DirectPtxLinearActivation.LeakyRelu:
                // 0.01 * x for x < 0, else x.  slope 0f3C23D70A = 0.01f
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3C23D70A;");
                ptx.AppendLine($"    max.f32 {v}, {v}, %f25;");
                break;
            case DirectPtxLinearActivation.Tanh:
                ptx.AppendLine($"    tanh.approx.f32 {v}, {v};");
                break;
            case DirectPtxLinearActivation.Sigmoid:
                // 1 / (1 + exp(-x)) = 0.5 * tanh(0.5x) + 0.5
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3F000000;");       // 0.5x
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F000000;");       // 0.5*tanh
                ptx.AppendLine($"    add.rn.f32 {v}, %f25, 0f3F000000;");       // +0.5
                break;
            case DirectPtxLinearActivation.Swish:
                // x * sigmoid(x)
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3F000000;");
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F000000;");
                ptx.AppendLine("    add.rn.f32 %f25, %f25, 0f3F000000;");       // sigmoid(x)
                ptx.AppendLine($"    mul.rn.f32 {v}, {v}, %f25;");
                break;
            case DirectPtxLinearActivation.GeluTanh:
                // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))), matching the M=1 kernel.
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, {v};");             // x^2
                ptx.AppendLine($"    mul.rn.f32 %f25, %f25, {v};");            // x^3
                ptx.AppendLine($"    fma.rn.f32 %f25, %f25, 0f3D372713, {v};");// x + 0.044715 x^3
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F4C422A;");       // * sqrt(2/pi)
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    add.rn.f32 %f25, %f25, 0f3F800000;");       // + 1
                ptx.AppendLine($"    mul.rn.f32 %f25, %f25, {v};");            // * x
                ptx.AppendLine($"    mul.rn.f32 {v}, %f25, 0f3F000000;");       // * 0.5
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        var input = new DirectPtxExtent(m, k);
        var weights = new DirectPtxExtent(n, k);
        var bias = new DirectPtxExtent(n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-tiled",
            Version: 1,
            Architecture: architecture,
            Variant: $"gemm-fp32-m{m}-k{k}-n{n}-{activation}".ToLowerInvariant(),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: (BlockM + BlockN) * BlockK * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "activation(A[M,K] @ transpose(W[N,K]) + bias[N])",
                ["activation"] = activation.ToString(),
                ["weights"] = "output-major-row-major-fp32",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["accumulator"] = "thread-private-fp32-registers",
                ["staging"] = "shared-memory-double-tile-a-and-w",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int k, int n) =>
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    // Fail closed until three clean promotion runs clear the release gate.
    internal static bool IsPromotedShape(int m, int k, int n) => false;

    private static void ValidateShape(int m, int k, int n)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Tiled fused-linear supports M in {64,128,256,512,1024,2048}, " +
                "K/N in {256,512,1024,2048,4096}; M%64==0, N%64==0, K%8==0.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
