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
/// Physical weight contract baked into a fused-linear specialization. The
/// kernel never receives a stride or layout flag at launch time.
/// </summary>
internal enum DirectPtxLinearWeightLayout
{
    /// <summary>Canonical AiDotNet linear/GEMM weights, row-major [K,N].</summary>
    InputMajor,
    /// <summary>Prepacked decode weights, row-major [N,K].</summary>
    OutputMajor
}

/// <summary>
/// General-M FP32 fused linear + bias + activation, computed as a real
/// register-blocked, shared-memory-staged tiled GEMM rather than the warp-reduction
/// dot product of the M=1 decode kernel. Computes
/// <c>C[M,N] = activation(A[M,K] @ B[K,N] + bias[N])</c> for canonical
/// input-major weights, or the equivalent product with prepacked output-major
/// <c>W[N,K]</c>. A block loads one BM-by-BK tile of A and one BN-by-BK tile of
/// weights into shared memory on each K-step; every thread accumulates a
/// TM-by-TN micro-tile entirely in registers. The selected weight layout is
/// baked into PTX, so admitted launches contain no runtime layout branch.
///
/// Tile: BM=BN=16, BK=32, TM=TN=2: 64 threads, four FP32 accumulators/thread,
/// and 4 KiB shared. The smaller output tile supplies enough independent blocks
/// for transformer matrices whose row count is only 64 while retaining coalesced
/// cooperative loads and a register-only epilogue.
/// </summary>
internal sealed class PtxFusedLinearTiledKernel : IDisposable
{
    internal const int BlockM = 16;
    internal const int BlockN = 16;
    internal const int BlockK = 32;
    internal const int ThreadM = 2;
    internal const int ThreadN = 2;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 64
    internal const string EntryPoint = "aidotnet_fused_linear_tiled";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal int BatchCount { get; }
    internal bool HasBias { get; }
    internal DirectPtxLinearActivation Activation { get; }
    internal DirectPtxLinearWeightLayout WeightLayout { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearTiledKernel(
        DirectPtxRuntime runtime,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout = DirectPtxLinearWeightLayout.OutputMajor,
        bool hasBias = true,
        int batchCount = 1)
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
        if (batchCount <= 0 || batchCount > 64)
            throw new ArgumentOutOfRangeException(nameof(batchCount));
        if (!hasBias && activation != DirectPtxLinearActivation.None)
            throw new ArgumentException("A no-bias GEMM specialization cannot apply a fused activation.", nameof(activation));
        BatchCount = batchCount;
        HasBias = hasBias;
        Activation = activation;
        WeightLayout = weightLayout;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, m, k, n, activation, weightLayout, hasBias, batchCount);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            m, k, n, activation, weightLayout, hasBias, batchCount);
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
        if (!HasBias) throw new InvalidOperationException("This specialization has no bias parameter.");
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
            (uint)(N / BlockN), (uint)(M / BlockM), (uint)BatchCount,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchGemm(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView output)
    {
        if (HasBias) throw new InvalidOperationException("This specialization requires a bias parameter.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, weights))
            throw new ArgumentException("GEMM output may not alias either input.");

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            (uint)(N / BlockN), (uint)(M / BlockM), (uint)BatchCount,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout = DirectPtxLinearWeightLayout.OutputMajor,
        bool hasBias = true,
        int batchCount = 1)
    {
        ValidateShape(m, k, n);
        if (batchCount <= 0 || batchCount > 64)
            throw new ArgumentOutOfRangeException(nameof(batchCount));
        if (!hasBias && activation != DirectPtxLinearActivation.None)
            throw new ArgumentException("A no-bias GEMM specialization cannot apply a fused activation.", nameof(activation));
        int kBytes = checked(k * sizeof(float));
        int nBytes = checked(n * sizeof(float));
        int inputBatchBytes = checked(m * k * sizeof(float));
        int weightBatchBytes = checked(k * n * sizeof(float));
        int outputBatchBytes = checked(m * n * sizeof(float));

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// tiled {(hasBias ? "fused-linear" : "gemm")} M={m} K={k} N={n} " +
            $"batch={batchCount} act={activation} weights={weightLayout} " +
            $"tile={BlockM}x{BlockN}x{BlockK} thread={ThreadM}x{ThreadN}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        if (hasBias) ptx.AppendLine("    .param .u64 bias_ptr,");
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
        if (hasBias) ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, w_tile;");

        // Thread / block coordinates.
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");                         // tid 0..63
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                        // block col
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");                        // block row
        ptx.AppendLine("    shr.u32 %r3, %r0, 3;");                          // threadRow = tid / (BN/TN=8)
        ptx.AppendLine("    and.b32 %r4, %r0, 7;");                          // threadCol = tid % 8
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockM};");               // blockRow*BM (base A row)
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockN};");               // blockCol*BN (base W row / N col)
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");             // threadRow*TM (A sub-row)
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");             // threadCol*TN (W sub-row)

        if (batchCount > 1)
        {
            ptx.AppendLine("    mov.u32 %r22, %ctaid.z;");
            ptx.AppendLine($"    mul.wide.u32 %rd20, %r22, {inputBatchBytes};");
            ptx.AppendLine("    add.u64 %rd0, %rd0, %rd20;");
            ptx.AppendLine($"    mul.wide.u32 %rd20, %r22, {weightBatchBytes};");
            ptx.AppendLine("    add.u64 %rd1, %rd1, %rd20;");
            ptx.AppendLine($"    mul.wide.u32 %rd20, %r22, {outputBatchBytes};");
            ptx.AppendLine("    add.u64 %rd3, %rd3, %rd20;");
        }

        // Zero the TM*TN accumulator registers (%f0..%f3 for the 2x2 tile).
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");

        // k0 loop counter in %r9 (element units), and running global-K offsets.
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("K_TILE_LOOP:");

        // ---- Cooperative global -> shared load ----
        // Each 2 KiB tile contributes 512 FP32 elements, so every thread owns
        // eight A and eight W scalar copies into the selected shared layout.
        EmitCooperativeLoad(ptx, kBytes, nBytes, weightLayout);
        ptx.AppendLine("    bar.sync 0;");

        // ---- Register-blocked inner product over BK ----
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");

        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");

        // ---- Epilogue: + bias[n], activation, store C[m,n] ----
        EmitEpilogue(ptx, activation, nBytes, hasBias);

        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Each thread loads eight A elements and eight W elements into shared memory.
    private static void EmitCooperativeLoad(
        StringBuilder ptx,
        int kBytes,
        int nBytes,
        DirectPtxLinearWeightLayout weightLayout)
    {
        // %r9 holds k0 (element units into K).
        for (int slot = 0; slot < 8; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 5;");                       // row=e/32
            ptx.AppendLine($"    and.b32 %r13, {e}, 31;");                      // kk=e%32
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");                     // global A row
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                     // global K col
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");          // row * K * 4
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");                  // col * 4
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            ptx.AppendLine("    add.u32 %r14, %r6, %r12;");                     // global N / W row
            if (weightLayout == DirectPtxLinearWeightLayout.OutputMajor)
            {
                ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
                ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
                ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            }
            else
            {
                ptx.AppendLine($"    mul.wide.u32 %rd8, %r15, {nBytes};");
                ptx.AppendLine("    mul.wide.u32 %rd9, %r14, 4;");
                ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
                ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            }
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    // For each k in 0..BK-1, load the register fragments and issue TM*TN FMAs.
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
        }
    }

    private static void EmitEpilogue(
        StringBuilder ptx,
        DirectPtxLinearActivation activation,
        int nBytes,
        bool hasBias)
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
                if (hasBias)
                {
                    ptx.AppendLine("    add.u64 %rd18, %rd2, %rd17;");         // &bias[n]
                    ptx.AppendLine("    ld.global.nc.f32 %f24, [%rd18];");
                    ptx.AppendLine($"    add.rn.f32 %f{acc}, %f{acc}, %f24;");  // + bias
                }
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
        DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout,
        bool hasBias,
        int batchCount)
    {
        var input = batchCount == 1
            ? new DirectPtxExtent(m, k)
            : new DirectPtxExtent(batchCount, m, k);
        var weights = weightLayout == DirectPtxLinearWeightLayout.OutputMajor
            ? (batchCount == 1
                ? new DirectPtxExtent(n, k)
                : new DirectPtxExtent(batchCount, n, k))
            : (batchCount == 1
                ? new DirectPtxExtent(k, n)
                : new DirectPtxExtent(batchCount, k, n));
        var bias = new DirectPtxExtent(n);
        var output = batchCount == 1
            ? new DirectPtxExtent(m, n)
            : new DirectPtxExtent(batchCount, m, n);
        var tensors = new List<DirectPtxTensorContract>
        {
            new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
            new("weights", DirectPtxPhysicalType.Float32,
                weightLayout == DirectPtxLinearWeightLayout.OutputMajor
                    ? DirectPtxPhysicalLayout.LinearWeightOutputMajor
                    : DirectPtxPhysicalLayout.LinearWeightInputMajor,
                weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
        };
        if (hasBias)
            tensors.Add(new DirectPtxTensorContract(
                "bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        tensors.Add(new DirectPtxTensorContract(
            "output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
            output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact));

        return new DirectPtxKernelBlueprint(
            Operation: hasBias ? "fused-linear-tiled" : "gemm-tiled",
            Version: 1,
            Architecture: architecture,
            Variant: $"gemm-fp32-b{batchCount}-m{m}-k{k}-n{n}-{activation}-{weightLayout}".ToLowerInvariant(),
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: (BlockM + BlockN) * BlockK * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = Formula(weightLayout, hasBias, batchCount),
                ["activation"] = activation.ToString(),
                ["weights"] = weightLayout == DirectPtxLinearWeightLayout.OutputMajor
                    ? "output-major-row-major-fp32"
                    : "input-major-row-major-fp32",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["accumulator"] = "thread-private-fp32-registers",
                ["staging"] = "shared-memory-double-tile-a-and-w",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["batch-count"] = batchCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["bias"] = hasBias ? "fused-register-epilogue" : "none",
                ["stride-parameters"] = "none"
            });
    }

    private static string Formula(
        DirectPtxLinearWeightLayout weightLayout,
        bool hasBias,
        int batchCount)
    {
        string batch = batchCount == 1 ? string.Empty : "batched ";
        string product = weightLayout == DirectPtxLinearWeightLayout.OutputMajor
            ? "A[M,K] @ transpose(W[N,K])"
            : "A[M,K] @ W[K,N]";
        return hasBias
            ? $"{batch}activation({product} + bias[N])"
            : $"{batch}{product}";
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
