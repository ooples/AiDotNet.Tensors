using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// Describes a binary SIMD operation by its x86 opcode.
/// Adding a new operation (Divide, Min, Max) is a one-liner — no switch statements
/// or existing code modifications needed (Open/Closed Principle).
/// </summary>
internal sealed class JitBinaryOp
{
    /// <summary>The x86 opcode for this packed single-precision operation.</summary>
    public byte Opcode { get; }

    /// <summary>Unique ID for cache key generation.</summary>
    public int Id { get; }

    private JitBinaryOp(byte opcode, int id)
    {
        Opcode = opcode;
        Id = id;
    }

    // To add a new op: define a new static field here. That's it — no switch changes needed.
    public static readonly JitBinaryOp Add = new(0x58, 1);       // VADDPS
    public static readonly JitBinaryOp Multiply = new(0x59, 2);  // VMULPS
    public static readonly JitBinaryOp Subtract = new(0x5C, 3);  // VSUBPS
    public static readonly JitBinaryOp Divide = new(0x5E, 4);    // VDIVPS
    public static readonly JitBinaryOp Min = new(0x5D, 5);       // VMINPS
    public static readonly JitBinaryOp Max = new(0x5F, 6);       // VMAXPS
}

/// <summary>
/// Generates JIT-compiled x86-64 machine code for hot tensor operations.
/// Kernels are specialized for exact problem dimensions with:
/// - Constants baked as immediates (no register pressure for loop bounds)
/// - Optimal unroll factors chosen per size
/// - Non-temporal stores on aligned output (AlignedBuffer guarantees 64-byte alignment)
///
/// This is the same approach oneDNN/Xbyak and libtorch use — raw machine code
/// generation specialized per problem size — but written entirely in C#.
///
/// Windows x64 ABI: RCX=arg0(src0*), RDX=arg1(src1*), R8=arg2(dst*), R9=arg3(length)
/// </summary>
internal static class CpuJitKernels
{
    // Delegate types for JIT-compiled kernels
    // These match the pointer signatures: (float* src0, float* src1, float* dst, int length)
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    internal unsafe delegate void BinaryKernel(float* src0, float* src1, float* dst, int length);

    // Unary: (float* src, float* dst, int length)
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    internal unsafe delegate void UnaryKernel(float* src, float* dst, int length);

    // Ternary: (float* a, float* b, float* c, float* dst, int length)
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    internal unsafe delegate void TernaryKernel(float* a, float* b, float* c, float* dst, int length);

    // GEMM micro-kernel: (float* packedA, float* packedB, float* c, int kc)
    // ldc is baked into the kernel at JIT time
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    internal unsafe delegate void GemmMicroKernel(float* packedA, float* packedB, float* c, int kc);

    // Cache compiled kernels by (opId, aligned, length)
    // Uses Lazy<> to ensure only one kernel is generated per key (avoids ExecutableBuffer leak under contention)
    private static readonly ConcurrentDictionary<long, Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>> _cache = new();

    // Pack (opId, aligned, length) into a single long key
    private static long MakeKey(int opId, bool aligned, int length)
        => ((long)opId << 33) | ((aligned ? 1L : 0L) << 32) | (uint)length;

    // Separate key space for unary/fused ops (opId > 100 to avoid collisions)
    private const int OP_RELU = 101;
    private const int OP_FUSED_ADD_RELU = 102;
    private const int OP_SIGMOID = 103;
    private const int OP_GEMM_MICRO = 200; // key space: (OP_GEMM_MICRO + ldc) to differentiate by ldc

    /// <summary>
    /// Gets or compiles a JIT binary kernel for any operation (Add, Multiply, Subtract, etc.).
    /// The operation is defined by <see cref="JitBinaryOp"/> — adding new ops requires
    /// no changes to this method (Open/Closed Principle).
    /// </summary>
    /// <param name="op">The binary operation to compile.</param>
    /// <param name="length">Number of floats to process.</param>
    /// <param name="aligned">True for aligned NT stores (AlignedBuffer), false for VMOVUPS (GC arrays).</param>
    public static unsafe BinaryKernel GetBinaryKernel(JitBinaryOp op, int length, bool aligned = false)
    {
        long key = MakeKey(op.Id, aligned, length);
        var entry = _cache.GetOrAdd(key, _ => new Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>(() =>
        {
            var buf = aligned
                ? GenerateBinaryKernelAligned(length, op)
                : GenerateBinaryKernelUnaligned(length, op);
            return (buf, (Delegate)buf.CreateDelegate<BinaryKernel>());
        }));
        return (BinaryKernel)entry.Value.Kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT kernel for ReLU: dst[i] = max(src[i], 0).
    /// </summary>
    /// <param name="length">Number of floats to process.</param>
    /// <param name="aligned">True for aligned NT stores, false for VMOVUPS.</param>
    public static unsafe UnaryKernel GetReLUKernel(int length, bool aligned = false)
    {
        long key = MakeKey(OP_RELU, aligned, length);
        var entry = _cache.GetOrAdd(key, _ => new Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>(() =>
        {
            var buf = aligned
                ? GenerateReLUKernelAligned(length)
                : GenerateReLUKernelUnaligned(length);
            return (buf, (Delegate)buf.CreateDelegate<UnaryKernel>());
        }));
        return (UnaryKernel)entry.Value.Kernel;
    }

    /// <summary>
    /// Gets or compiles a fused Add+ReLU kernel: dst[i] = max(a[i] + b[i], 0).
    /// Eliminates one full array pass compared to separate Add then ReLU.
    /// </summary>
    public static unsafe BinaryKernel GetFusedAddReLUKernel(int length)
    {
        long key = MakeKey(OP_FUSED_ADD_RELU, false, length);
        var entry = _cache.GetOrAdd(key, _ => new Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>(() =>
        {
            var buf = GenerateFusedAddReLUKernel(length);
            return (buf, (Delegate)buf.CreateDelegate<BinaryKernel>());
        }));
        return (BinaryKernel)entry.Value.Kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT Sigmoid kernel: dst[i] = sigmoid(src[i]).
    /// Uses 5th-order polynomial approximation with constants baked into the data section.
    /// </summary>
    public static unsafe UnaryKernel GetSigmoidKernel(int length)
    {
        long key = MakeKey(OP_SIGMOID, false, length);
        var entry = _cache.GetOrAdd(key, _ => new Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>(() =>
        {
            var buf = GenerateSigmoidKernel(length);
            return (buf, (Delegate)buf.CreateDelegate<UnaryKernel>());
        }));
        return (UnaryKernel)entry.Value.Kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT 6x16 GEMM micro-kernel with ldc baked as immediate.
    /// Computes C[6,16] += packedA[6,kc] * packedB[kc,16] using 12 FMA accumulators.
    /// ldc is the leading dimension of C (in floats, not bytes).
    /// </summary>
    public static unsafe GemmMicroKernel GetGemmMicroKernel(int kc, int ldc)
    {
        // Pack ldc into the key: OP_GEMM_MICRO in opId, ldc in aligned flag area, kc in length
        long key = ((long)(OP_GEMM_MICRO + ldc) << 33) | (uint)kc;
        var entry = _cache.GetOrAdd(key, _ => new Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>(() =>
        {
            var buf = GenerateGemmMicroKernel(kc, ldc);
            return (buf, (Delegate)buf.CreateDelegate<GemmMicroKernel>());
        }));
        return (GemmMicroKernel)entry.Value.Kernel;
    }

    /// <summary>
    /// Check if the current CPU and OS support our JIT kernels.
    /// Requires AVX2+FMA and Windows x64 ABI (our emitter generates Windows x64 calling convention).
    /// </summary>
    public static bool IsSupported
    {
        get
        {
#if NET5_0_OR_GREATER
            return RuntimeInformation.IsOSPlatform(OSPlatform.Windows) &&
                   System.Runtime.Intrinsics.X86.Avx2.IsSupported &&
                   System.Runtime.Intrinsics.X86.Fma.IsSupported;
#else
            return false;
#endif
        }
    }

    // ==================== Kernel generators ====================
    // All binary ops use the same structure — only the x86 opcode differs.
    // The opcode is carried by JitBinaryOp, so no switch statements needed.

    /// <summary>
    /// Generates a binary kernel with non-temporal (aligned) stores.
    /// Requires 32-byte aligned output (AlignedBuffer).
    /// </summary>
    private static ExecutableBuffer GenerateBinaryKernelAligned(int length, JitBinaryOp op)
    {
        var e = new X86Emitter();
        e.Prologue();

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            // Load 4 vectors from src0 (RCX)
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            // Apply op with src1 (RDX) — no switch, just use the opcode directly
            EmitBinaryOp4Wide(e, op, X86Emitter.RDX);

            // Non-temporal stores to dst (R8) — requires alignment
            e.VmovntpsStore(X86Emitter.YMM0, X86Emitter.R8, 0);
            e.VmovntpsStore(X86Emitter.YMM1, X86Emitter.R8, 32);
            e.VmovntpsStore(X86Emitter.YMM2, X86Emitter.R8, 64);
            e.VmovntpsStore(X86Emitter.YMM3, X86Emitter.R8, 96);

            // Advance pointers by 128 bytes (32 floats)
            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);
            e.AddImm32(X86Emitter.R8, 128);

            e.AddImm32(X86Emitter.RBX, 128);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);

            e.Sfence();
        }

        // Remainder in groups of 8
        EmitBinaryRemainder(e, op, length - simdCount);

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Generates a ReLU kernel with non-temporal (aligned) stores.
    /// </summary>
    private static ExecutableBuffer GenerateReLUKernelAligned(int length)
    {
        var e = new X86Emitter();
        e.Prologue();

        e.Vxorps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM15);

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM15);

            e.VmovntpsStore(X86Emitter.YMM0, X86Emitter.RDX, 0);
            e.VmovntpsStore(X86Emitter.YMM1, X86Emitter.RDX, 32);
            e.VmovntpsStore(X86Emitter.YMM2, X86Emitter.RDX, 64);
            e.VmovntpsStore(X86Emitter.YMM3, X86Emitter.RDX, 96);

            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);

            e.AddImm32(X86Emitter.RBX, 128);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);

            e.Sfence();
        }

        EmitReLURemainder(e, length - simdCount);

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Generates a binary kernel using VMOVUPS stores (safe for GC arrays).
    /// Same 4x unrolled structure as aligned kernels but no alignment requirement.
    /// </summary>
    private static ExecutableBuffer GenerateBinaryKernelUnaligned(int length, JitBinaryOp op)
    {
        var e = new X86Emitter();
        e.Prologue();

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            // Load 4 vectors from src0 (RCX)
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            // Apply op with src1 (RDX) — uses JitBinaryOp.Opcode, no switch needed
            EmitBinaryOp4Wide(e, op, X86Emitter.RDX);

            // Unaligned stores to dst (R8)
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.R8, 0);
            e.VmovupsStore(X86Emitter.YMM1, X86Emitter.R8, 32);
            e.VmovupsStore(X86Emitter.YMM2, X86Emitter.R8, 64);
            e.VmovupsStore(X86Emitter.YMM3, X86Emitter.R8, 96);

            // Advance pointers by 128 bytes (32 floats)
            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);
            e.AddImm32(X86Emitter.R8, 128);

            e.AddImm32(X86Emitter.RBX, 128);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);
        }

        // Remainder in groups of 8
        EmitBinaryRemainder(e, op, length - simdCount);

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Generates a ReLU kernel using VMOVUPS stores (safe for GC arrays).
    /// </summary>
    private static ExecutableBuffer GenerateReLUKernelUnaligned(int length)
    {
        var e = new X86Emitter();
        e.Prologue();

        e.Vxorps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM15);

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM15);

            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.RDX, 0);
            e.VmovupsStore(X86Emitter.YMM1, X86Emitter.RDX, 32);
            e.VmovupsStore(X86Emitter.YMM2, X86Emitter.RDX, 64);
            e.VmovupsStore(X86Emitter.YMM3, X86Emitter.RDX, 96);

            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);

            e.AddImm32(X86Emitter.RBX, 128);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);
        }

        EmitReLURemainder(e, length - simdCount);

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Generates a fused Add+ReLU kernel: dst[i] = max(a[i] + b[i], 0).
    /// Eliminates one full array traversal compared to Add then ReLU.
    /// </summary>
    private static ExecutableBuffer GenerateFusedAddReLUKernel(int length)
    {
        var e = new X86Emitter();
        e.Prologue();

        e.Vxorps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM15);

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            // Add b (RDX)
            e.Vaddps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.RDX, 0);
            e.Vaddps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.RDX, 32);
            e.Vaddps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.RDX, 64);
            e.Vaddps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.RDX, 96);

            // ReLU: max(result, 0)
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM15);
            e.Vmaxps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM15);

            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.R8, 0);
            e.VmovupsStore(X86Emitter.YMM1, X86Emitter.R8, 32);
            e.VmovupsStore(X86Emitter.YMM2, X86Emitter.R8, 64);
            e.VmovupsStore(X86Emitter.YMM3, X86Emitter.R8, 96);

            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);
            e.AddImm32(X86Emitter.R8, 128);

            e.AddImm32(X86Emitter.RBX, 128);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);
        }

        // Remainder in groups of 8
        int remaining = length - simdCount;
        int vec8Count = remaining / 8;
        for (int v = 0; v < vec8Count; v++)
        {
            int off = v * 32;
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, off);
            e.Vaddps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.RDX, off);
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM15);
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.R8, off);
        }

        // Scalar tail: process remaining <8 elements one at a time
        int scalarStart = vec8Count * 8;
        int scalarTail = remaining - scalarStart;
        if (scalarTail > 0)
        {
            int baseOff = vec8Count * 32;
            for (int s = 0; s < scalarTail; s++)
            {
                int off = baseOff + s * 4;
                // Load src0[i], add src1[i], max with zero, store to dst[i]
                EmitScalarLoad(e, X86Emitter.YMM0, X86Emitter.RCX, off);
                EmitScalarBinaryOp(e, JitBinaryOp.Add, X86Emitter.YMM0, X86Emitter.RDX, off);
                EmitScalarMaxWithZero(e, X86Emitter.YMM0, X86Emitter.YMM15);
                EmitScalarStore(e, X86Emitter.YMM0, X86Emitter.R8, off);
            }
        }

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Generates a JIT 6x16 GEMM micro-kernel.
    /// Register allocation:
    ///   YMM0-YMM11: 12 accumulators (6 rows x 2 vectors of 8 = 16 columns)
    ///   YMM12-YMM13: B row loads (2 vectors)
    ///   YMM14: A element broadcast (reused per row)
    ///   YMM15: (spare)
    ///
    /// Windows x64 ABI: RCX=packedA*, RDX=packedB*, R8=C*, R9=kc
    /// ldc is baked as immediate displacement for C row access.
    ///
    /// Mr=6 (A panel height), Nr=16 (B panel width)
    /// A packed layout: kc groups of Mr=6 contiguous floats (stride = 24 bytes)
    /// B packed layout: kc groups of Nr=16 contiguous floats (stride = 64 bytes)
    /// </summary>
    private static ExecutableBuffer GenerateGemmMicroKernel(int kc, int ldc)
    {
        var e = new X86Emitter();
        e.Prologue();

        int ldcBytes = ldc * sizeof(float);

        // Zero all 12 accumulators (YMM0-YMM11)
        for (int r = 0; r < 12; r++)
        {
            e.Vxorps(r, r, r);
        }

        if (kc > 0)
        {
            // Save R8 (C pointer) into R10 — we'll need R8 pristine for C stores later
            e.MovRR(X86Emitter.R10, X86Emitter.R8);

            // Loop counter: R9 = kc (counts down to 0)
            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            // Load B row: 2 vectors of 8 floats from packedB (RDX)
            e.VmovupsLoad(X86Emitter.YMM12, X86Emitter.RDX, 0);   // B[p, 0:7]
            e.VmovupsLoad(X86Emitter.YMM13, X86Emitter.RDX, 32);  // B[p, 8:15]

            // Row 0: broadcast A[p*6+0], FMA with B
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 0);
            e.Vfmadd231ps(X86Emitter.YMM0, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM1, X86Emitter.YMM14, X86Emitter.YMM13);

            // Row 1
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 4);
            e.Vfmadd231ps(X86Emitter.YMM2, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM3, X86Emitter.YMM14, X86Emitter.YMM13);

            // Row 2
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 8);
            e.Vfmadd231ps(X86Emitter.YMM4, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM5, X86Emitter.YMM14, X86Emitter.YMM13);

            // Row 3
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 12);
            e.Vfmadd231ps(X86Emitter.YMM6, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM7, X86Emitter.YMM14, X86Emitter.YMM13);

            // Row 4
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 16);
            e.Vfmadd231ps(X86Emitter.YMM8, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM9, X86Emitter.YMM14, X86Emitter.YMM13);

            // Row 5
            e.Vbroadcastss(X86Emitter.YMM14, X86Emitter.RCX, 20);
            e.Vfmadd231ps(X86Emitter.YMM10, X86Emitter.YMM14, X86Emitter.YMM12);
            e.Vfmadd231ps(X86Emitter.YMM11, X86Emitter.YMM14, X86Emitter.YMM13);

            // Advance A by Mr*4=24 bytes, B by Nr*4=64 bytes
            e.AddImm32(X86Emitter.RCX, 24);
            e.AddImm32(X86Emitter.RDX, 64);

            // Decrement kc and loop
            e.SubImm32(X86Emitter.R9, 1);
            e.Jne(loopLabel);

            // === Store accumulated results back to C (load-add-store) ===
            // R10 = C pointer, ldc baked as displacement
            // Row 0: C[0, 0:15]
            EmitGemmStoreRow(e, X86Emitter.R10, 0, X86Emitter.YMM0, X86Emitter.YMM1);
            // Row 1: C[1, 0:15] = R10 + ldcBytes
            EmitGemmStoreRow(e, X86Emitter.R10, ldcBytes, X86Emitter.YMM2, X86Emitter.YMM3);
            // Row 2
            EmitGemmStoreRow(e, X86Emitter.R10, ldcBytes * 2, X86Emitter.YMM4, X86Emitter.YMM5);
            // Row 3
            EmitGemmStoreRow(e, X86Emitter.R10, ldcBytes * 3, X86Emitter.YMM6, X86Emitter.YMM7);
            // Row 4
            EmitGemmStoreRow(e, X86Emitter.R10, ldcBytes * 4, X86Emitter.YMM8, X86Emitter.YMM9);
            // Row 5
            EmitGemmStoreRow(e, X86Emitter.R10, ldcBytes * 5, X86Emitter.YMM10, X86Emitter.YMM11);
        }

        e.Epilogue();
        return e.Build();
    }

    /// <summary>
    /// Emits load-add-store for one row of the 6x16 GEMM tile.
    /// Loads existing C[row], adds accumulator, stores back.
    /// </summary>
    private static void EmitGemmStoreRow(X86Emitter e, int baseReg, int disp,
        int accumLo, int accumHi)
    {
        // Load existing C values
        e.VmovupsLoad(X86Emitter.YMM14, baseReg, disp);
        e.VmovupsLoad(X86Emitter.YMM15, baseReg, disp + 32);

        // Add accumulators
        e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, accumLo);
        e.Vaddps(X86Emitter.YMM15, X86Emitter.YMM15, accumHi);

        // Store back
        e.VmovupsStore(X86Emitter.YMM14, baseReg, disp);
        e.VmovupsStore(X86Emitter.YMM15, baseReg, disp + 32);
    }

    /// <summary>
    /// Generates a JIT Sigmoid kernel using 5th-order polynomial approximation.
    /// All polynomial constants are baked into the executable buffer's data section
    /// via VBROADCASTSS from absolute addresses — zero register pressure for constants.
    ///
    /// sigmoid(x) ≈ 0.5 + x*(c1 + x²*(c3 + x²*c5))
    /// where c1=0.2156292, c3=-0.008921921, c5=0.0001585434
    /// Input clamped to [-5, 5], 4x unrolled (32 floats per iteration).
    ///
    /// Windows x64 ABI: RCX=src*, RDX=dst*, R8=length (int)
    /// </summary>
    private static ExecutableBuffer GenerateSigmoidKernel(int length)
    {
        var e = new X86Emitter();
        e.Prologue();

        // sigmoid(x) = 1 / (1 + exp(-x)) using FastExp approach (Cephes-style)
        // FastExp: range reduction x = n*ln2 + r, polynomial exp(r), scale by 2^n

        // Data section constants
        int idxClampLo  = e.EmitDataConstant(-87.3365f);
        int idxClampHi  = e.EmitDataConstant(88.7228f);
        int idxLog2e    = e.EmitDataConstant(1.44269504f);    // 1/ln2
        int idxLn2      = e.EmitDataConstant(0.693147181f);   // ln2
        int idxOne      = e.EmitDataConstant(1.0f);
        int idxP0       = e.EmitDataConstant(1.0f / 720.0f);  // 1/6! = r^6 coeff
        int idxP1       = e.EmitDataConstant(1.0f / 120.0f);  // 1/5!
        int idxP2       = e.EmitDataConstant(1.0f / 24.0f);   // 1/4!
        int idxP3       = e.EmitDataConstant(1.0f / 6.0f);    // 1/3!
        int idxHalf     = e.EmitDataConstant(0.5f);            // 1/2!
        int idx127      = e.EmitDataConstant(127.0f); // 127 as float, converted to int in loop

        // Load constants into YMM8-YMM15
        // We process 1 vector at a time (register-pressure limited by FastExp)
        e.VbroadcastssConst(X86Emitter.YMM8, idxClampLo);
        e.VbroadcastssConst(X86Emitter.YMM9, idxClampHi);
        e.VbroadcastssConst(X86Emitter.YMM10, idxLog2e);
        e.VbroadcastssConst(X86Emitter.YMM11, idxLn2);
        e.VbroadcastssConst(X86Emitter.YMM12, idxOne);
        e.VbroadcastssConst(X86Emitter.YMM13, idxHalf);

        int simdCount = (length / 8) * 8;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            // Load x from src, negate for sigmoid: -x
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.Vxorps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM1); // zero
            e.Vsubps(X86Emitter.YMM0, X86Emitter.YMM1, X86Emitter.YMM0); // -x

            // Clamp -x to [-87.3, 88.7]
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM8);
            e.Vminps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM9);

            // n = round(-x * log2e) → VCVTPS2DQ rounds to nearest int
            e.Vmulps(X86Emitter.YMM1, X86Emitter.YMM0, X86Emitter.YMM10); // -x * log2e
            e.Vcvtps2dq(X86Emitter.YMM2, X86Emitter.YMM1);                // n_int = round(...)
            e.Vcvtdq2ps(X86Emitter.YMM1, X86Emitter.YMM2);                // n_float = (float)n_int

            // r = -x - n * ln2 (fractional part in [-ln2/2, ln2/2])
            e.Vmulps(X86Emitter.YMM3, X86Emitter.YMM1, X86Emitter.YMM11); // n * ln2
            e.Vsubps(X86Emitter.YMM3, X86Emitter.YMM0, X86Emitter.YMM3);  // r = -x - n*ln2

            // exp(r) via Horner's method: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120 + r/720)))))
            // Load remaining polynomial coefficients from data section into temps
            // YMM4 = current accumulator
            e.VbroadcastssConst(X86Emitter.YMM4, idxP0);  // p = 1/720
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // p*r
            e.VbroadcastssConst(X86Emitter.YMM5, idxP1);
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM5); // + 1/120
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // *r
            e.VbroadcastssConst(X86Emitter.YMM5, idxP2);
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM5); // + 1/24
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // *r
            e.VbroadcastssConst(X86Emitter.YMM5, idxP3);
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM5); // + 1/6
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // *r
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM13); // + 0.5
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // *r
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM12); // + 1.0
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM3); // *r
            e.Vaddps(X86Emitter.YMM4, X86Emitter.YMM4, X86Emitter.YMM12); // + 1.0 → exp(r)

            // Scale by 2^n: reinterpret (n_int + 127) << 23 as float
            e.VbroadcastssConst(X86Emitter.YMM5, idx127);
            e.Vcvtps2dq(X86Emitter.YMM5, X86Emitter.YMM5); // 127 as int32
            e.Vpaddd(X86Emitter.YMM6, X86Emitter.YMM2, X86Emitter.YMM5); // n_int + 127
            e.VpslldImm(X86Emitter.YMM6, X86Emitter.YMM6, 23);            // << 23 → 2^n as float bits
            e.Vmulps(X86Emitter.YMM0, X86Emitter.YMM4, X86Emitter.YMM6); // exp(-x) = exp(r) * 2^n

            // sigmoid = 1 / (1 + exp(-x))
            e.Vaddps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM12); // 1 + exp(-x)
            // VDIVPS: YMM12(1.0) / YMM0(1+exp(-x)) → opcode 0x5E
            e.VbinaryPs(0x5E, X86Emitter.YMM0, X86Emitter.YMM12, X86Emitter.YMM0); // 1 / (1+exp(-x))

            // Store result
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.RDX, 0);

            // Advance pointers
            e.AddImm32(X86Emitter.RCX, 32);
            e.AddImm32(X86Emitter.RDX, 32);

            e.AddImm32(X86Emitter.RBX, 32);
            e.CmpImm32(X86Emitter.RBX, simdEndBytes);
            e.Jl(loopLabel);
        }

        e.Epilogue();
        return e.Build();
    }

    // ==================== Shared emit helpers (no switch statements) ====================

    /// <summary>
    /// Emits 4-wide binary op (YMM0-YMM3) with memory operand.
    /// Uses JitBinaryOp.Opcode directly via the generic VbinaryPs emitter —
    /// no switch/case needed. New operations are automatically supported.
    /// </summary>
    private static void EmitBinaryOp4Wide(X86Emitter e, JitBinaryOp op, int srcBaseReg)
    {
        e.VbinaryPs(op.Opcode, X86Emitter.YMM0, X86Emitter.YMM0, srcBaseReg, 0);
        e.VbinaryPs(op.Opcode, X86Emitter.YMM1, X86Emitter.YMM1, srcBaseReg, 32);
        e.VbinaryPs(op.Opcode, X86Emitter.YMM2, X86Emitter.YMM2, srcBaseReg, 64);
        e.VbinaryPs(op.Opcode, X86Emitter.YMM3, X86Emitter.YMM3, srcBaseReg, 96);
    }

    /// <summary>
    /// Emits remainder loop (groups of 8 floats) for binary ops, plus scalar tail.
    /// </summary>
    private static void EmitBinaryRemainder(X86Emitter e, JitBinaryOp op, int remaining)
    {
        int vec8Count = remaining / 8;
        for (int v = 0; v < vec8Count; v++)
        {
            int off = v * 32;
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, off);
            e.VbinaryPs(op.Opcode, X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.RDX, off);
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.R8, off);
        }

        // Scalar tail: process remaining elements one at a time using SSE scalar ops
        int scalarStart = vec8Count * 8;
        int scalarTail = remaining - scalarStart;
        if (scalarTail > 0)
        {
            int baseOff = vec8Count * 32;
            for (int s = 0; s < scalarTail; s++)
            {
                int off = baseOff + s * 4;
                // VMOVSS xmm0, [RCX+off] — load single float
                EmitScalarLoad(e, X86Emitter.YMM0, X86Emitter.RCX, off);
                // op xmm0, xmm0, [RDX+off] — apply scalar op
                EmitScalarBinaryOp(e, op, X86Emitter.YMM0, X86Emitter.RDX, off);
                // VMOVSS [R8+off], xmm0 — store single float
                EmitScalarStore(e, X86Emitter.YMM0, X86Emitter.R8, off);
            }
        }
    }

    /// <summary>
    /// Emits remainder loop for ReLU (groups of 8 floats), plus scalar tail.
    /// </summary>
    private static void EmitReLURemainder(X86Emitter e, int remaining)
    {
        int vec8Count = remaining / 8;
        for (int v = 0; v < vec8Count; v++)
        {
            int off = v * 32;
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, off);
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM15);
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.RDX, off);
        }

        // Scalar tail for ReLU
        int scalarStart = vec8Count * 8;
        int scalarTail = remaining - scalarStart;
        if (scalarTail > 0)
        {
            int baseOff = vec8Count * 32;
            for (int s = 0; s < scalarTail; s++)
            {
                int off = baseOff + s * 4;
                EmitScalarLoad(e, X86Emitter.YMM0, X86Emitter.RCX, off);
                // VMAXSS xmm0, xmm0, xmm15 for scalar ReLU
                EmitScalarMaxWithZero(e, X86Emitter.YMM0, X86Emitter.YMM15);
                EmitScalarStore(e, X86Emitter.YMM0, X86Emitter.RDX, off);
            }
        }
    }

    // Scalar helpers using VEX-encoded VMOVSS (4-byte load/store, no out-of-bounds reads)
    private static void EmitScalarLoad(X86Emitter e, int dst, int baseReg, int disp)
    {
        // VMOVSS xmm, [base+disp]: loads exactly 4 bytes (1 float)
        e.VmovssLoad(dst, baseReg, disp);
    }

    private static void EmitScalarBinaryOp(X86Emitter e, JitBinaryOp op, int dst, int srcBase, int disp)
    {
        // Use scalar (SS) variant to read exactly 4 bytes, not packed (PS) which reads 32
        e.VbinarySs(op.Opcode, dst, dst, srcBase, disp);
    }

    private static void EmitScalarMaxWithZero(X86Emitter e, int dst, int zero)
    {
        e.Vmaxps(dst, dst, zero);
    }

    private static void EmitScalarStore(X86Emitter e, int src, int baseReg, int disp)
    {
        // VMOVSS [base+disp], xmm: stores exactly 4 bytes (1 float)
        e.VmovssStore(src, baseReg, disp);
    }

    /// <summary>
    /// Disposes all cached kernels and frees executable memory.
    /// </summary>
    public static void ClearCache()
    {
        // Atomically swap the dictionary so concurrent GetOrAdd callers get a fresh cache
        // rather than accessing entries being disposed
        var old = new ConcurrentDictionary<long, Lazy<(ExecutableBuffer Buffer, Delegate Kernel)>>(_cache);
        _cache.Clear();

        foreach (var kvp in old)
        {
            if (kvp.Value.IsValueCreated)
                kvp.Value.Value.Buffer.Dispose();
        }
    }
}
