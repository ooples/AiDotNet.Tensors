using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// General-M FP32 fused <c>C[M,N] = activation(A[M,K] @ B[K,N] + bias[N])</c> for
/// the GemmBias* family, using a register-blocked, shared-memory-staged tile. This
/// is the standard-B counterpart of <see cref="PtxFusedLinearTiledKernel"/>: B is
/// canonical row-major [K,N] (K the row, N contiguous) rather than the output-major
/// [N,K] weight layout, so a block stages one BM×BK A tile and one BK×BN B tile each
/// K-step and every thread accumulates a TM×TN micro-tile in registers.
///
/// Tile: BM=BN=64, BK=8, TM=TN=4 → 256 threads, 16 FP32 accumulators/thread, 4 KiB
/// shared. Grid = (N/BN, M/BM). The fused bias+activation epilogue is parameterized
/// over the same activation set as the linear tile, so one kernel serves GemmBias,
/// GemmBiasRelu/Gelu/Sigmoid/Tanh/Swish/LeakyRelu.
/// </summary>
internal sealed class PtxFusedGemmBiasKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_fused_gemm_bias";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedGemmBiasKernel(
        DirectPtxRuntime runtime, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in GemmBias specialization is measured only on GA10x/SM86.");
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
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView a,
        DirectPtxTensorView b,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (Overlaps(output, a) || Overlaps(output, b) || Overlaps(output, bias))
            throw new ArgumentException("GemmBias output may not alias A, B, or bias.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &outputPointer;
        _module.Launch(
            _function, (uint)(N / BlockN), (uint)(M / BlockM), 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        ValidateShape(m, k, n);
        int kBytes = checked(k * sizeof(float));
        int nBytes = checked(n * sizeof(float));

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// tiled gemm-bias M={m} K={k} N={n} act={activation} " +
            $"tile={BlockM}x{BlockN}x{BlockK} thread={ThreadM}x{ThreadN}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<48>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockK * BlockN * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, b_tile;");

        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                          // block col
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");                          // block row
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");                            // threadRow = tid / 16
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");                           // threadCol = tid % 16
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockM};");                 // blockRow*BM
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockN};");                 // blockCol*BN
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");               // threadRow*TM
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");               // threadCol*TN

        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");

        ptx.AppendLine("    mov.u32 %r9, 0;");                                 // k0 (element units)
        ptx.AppendLine("K_TILE_LOOP:");
        EmitCooperativeLoad(ptx, kBytes, nBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");

        EmitEpilogue(ptx, activation, nBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitCooperativeLoad(StringBuilder ptx, int kBytes, int nBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");   // e = tid + slot*256
            // A tile [BM][BK]: row = e>>3, kk = e&7. globalRow = blockRow*BM + row.
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");                     // global A row
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                     // global K col = k0 + kk
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");                 // A_sh[e]
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // B tile [BK][BN]: bk = e>>6, bn = e&63. globalRow = k0 + bk, globalCol = blockCol*BN + bn.
            ptx.AppendLine($"    shr.u32 %r16, {e}, 6;");                       // bk
            ptx.AppendLine($"    and.b32 %r17, {e}, 63;");                      // bn
            ptx.AppendLine("    add.u32 %r18, %r9, %r16;");                     // global B row = k0 + bk
            ptx.AppendLine("    add.u32 %r19, %r6, %r17;");                     // global B col
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r18, {nBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r19, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");                  // B_sh[e] (same linear e)
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    private static void EmitInnerAccumulate(StringBuilder ptx)
    {
        // base_A = &A_sh[(threadRow*TM)*BK]; a[i] at +(i*BK + k)*4.
        // base_B = &B_sh[threadCol*TN];      b[j] at +(k*BN + j)*4.
        ptx.AppendLine($"    mul.lo.u32 %r20, %r7, {BlockK};");                 // (threadRow*TM)*BK
        ptx.AppendLine("    mul.wide.u32 %rd13, %r20, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");                      // base_A
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");                      // threadCol*TN * 4
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");                      // base_B
        for (int k = 0; k < BlockK; k++)
        {
            for (int i = 0; i < ThreadM; i++)
            {
                int off = (i * BlockK + k) * sizeof(float);
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd13+{off}];");
            }
            for (int j = 0; j < ThreadN; j++)
            {
                int off = (k * BlockN + j) * sizeof(float);
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{off}];");
            }
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    private static void EmitEpilogue(StringBuilder ptx, DirectPtxLinearActivation activation, int nBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");                          // m0
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");                          // n0
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");                    // m = m0 + i
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {nBytes};");
            ptx.AppendLine("    add.u64 %rd16, %rd3, %rd15;");                  // &C[m,0]
            for (int j = 0; j < ThreadN; j++)
            {
                int acc = i * ThreadN + j;
                ptx.AppendLine($"    add.u32 %r21, %r19, {j};");               // n = n0 + j
                ptx.AppendLine("    mul.wide.u32 %rd17, %r21, 4;");
                ptx.AppendLine("    add.u64 %rd18, %rd2, %rd17;");             // &bias[n]
                ptx.AppendLine("    ld.global.nc.f32 %f24, [%rd18];");
                ptx.AppendLine($"    add.rn.f32 %f{acc}, %f{acc}, %f24;");
                PtxLinearActivationEmit.Emit(ptx, activation, $"%f{acc}");
                ptx.AppendLine("    add.u64 %rd19, %rd16, %rd17;");            // &C[m,n]
                ptx.AppendLine($"    st.global.f32 [%rd19], %f{acc};");
            }
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int k, int n, DirectPtxLinearActivation activation)
    {
        var a = new DirectPtxExtent(m, k);
        var b = new DirectPtxExtent(k, n);
        var bias = new DirectPtxExtent(n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-gemm-bias",
            Version: 1,
            Architecture: architecture,
            Variant: $"gemm-fp32-m{m}-k{k}-n{n}-{activation}".ToLowerInvariant(),
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
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
                ["formula"] = "activation(A[M,K] @ B[K,N] + bias[N])",
                ["activation"] = activation.ToString(),
                ["b"] = "standard-row-major-fp32",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["accumulator"] = "thread-private-fp32-registers",
                ["staging"] = "shared-memory-double-tile-a-and-b",
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

    internal static bool IsPromotedShape(int m, int k, int n) => false;

    private static void ValidateShape(int m, int k, int n)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "GemmBias supports M in {64,128,256,512,1024,2048}, K/N in " +
                "{256,512,1024,2048,4096}; M%64==0, N%64==0, K%8==0.");
    }

    private static void Require(
        DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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
