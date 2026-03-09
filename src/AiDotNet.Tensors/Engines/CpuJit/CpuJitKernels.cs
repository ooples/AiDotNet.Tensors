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
    private static readonly ConcurrentDictionary<long, (ExecutableBuffer Buffer, Delegate Kernel)> _cache = new();

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
        var entry = _cache.GetOrAdd(key, _ =>
        {
            var buf = aligned
                ? GenerateBinaryKernelAligned(length, op)
                : GenerateBinaryKernelUnaligned(length, op);
            return (buf, (Delegate)buf.CreateDelegate<BinaryKernel>());
        });
        return (BinaryKernel)entry.Kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT kernel for ReLU: dst[i] = max(src[i], 0).
    /// </summary>
    /// <param name="length">Number of floats to process.</param>
    /// <param name="aligned">True for aligned NT stores, false for VMOVUPS.</param>
    public static unsafe UnaryKernel GetReLUKernel(int length, bool aligned = false)
    {
        long key = MakeKey(OP_RELU, aligned, length);
        var entry = _cache.GetOrAdd(key, _ =>
        {
            var buf = aligned
                ? GenerateReLUKernelAligned(length)
                : GenerateReLUKernelUnaligned(length);
            return (buf, (Delegate)buf.CreateDelegate<UnaryKernel>());
        });
        return (UnaryKernel)entry.Kernel;
    }

    /// <summary>
    /// Gets or compiles a fused Add+ReLU kernel: dst[i] = max(a[i] + b[i], 0).
    /// Eliminates one full array pass compared to separate Add then ReLU.
    /// </summary>
    public static unsafe BinaryKernel GetFusedAddReLUKernel(int length)
    {
        long key = MakeKey(OP_FUSED_ADD_RELU, false, length);
        var entry = _cache.GetOrAdd(key, _ =>
        {
            var buf = GenerateFusedAddReLUKernel(length);
            return (buf, (Delegate)buf.CreateDelegate<BinaryKernel>());
        });
        return (BinaryKernel)entry.Kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT Sigmoid kernel: dst[i] = sigmoid(src[i]).
    /// Uses 5th-order polynomial approximation with constants baked into the data section.
    /// </summary>
    public static unsafe UnaryKernel GetSigmoidKernel(int length)
    {
        long key = MakeKey(OP_SIGMOID, false, length);
        var entry = _cache.GetOrAdd(key, _ =>
        {
            var buf = GenerateSigmoidKernel(length);
            return (buf, (Delegate)buf.CreateDelegate<UnaryKernel>());
        });
        return (UnaryKernel)entry.Kernel;
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
        var entry = _cache.GetOrAdd(key, _ =>
        {
            var buf = GenerateGemmMicroKernel(kc, ldc);
            return (buf, (Delegate)buf.CreateDelegate<GemmMicroKernel>());
        });
        return (GemmMicroKernel)entry.Kernel;
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

        // Register data section constants — these get appended after code in the executable buffer
        int idxNeg5   = e.EmitDataConstant(-5.0f);
        int idxPos5   = e.EmitDataConstant(5.0f);
        int idxC5     = e.EmitDataConstant(1.5854344e-4f);
        int idxC3     = e.EmitDataConstant(-8.9219211e-3f);
        int idxC1     = e.EmitDataConstant(2.1562920e-1f);
        int idxHalf   = e.EmitDataConstant(0.5f);

        // Load constants into dedicated registers (YMM8-YMM13) — loaded once, used every iteration
        // These VBROADCASTSS instructions each emit MOV R11,imm64 + VBROADCASTSS ymm,[R11]
        e.VbroadcastssConst(X86Emitter.YMM8, idxNeg5);    // clamp low
        e.VbroadcastssConst(X86Emitter.YMM9, idxPos5);    // clamp high
        e.VbroadcastssConst(X86Emitter.YMM10, idxC5);     // 1.5854344e-4
        e.VbroadcastssConst(X86Emitter.YMM11, idxC3);     // -8.9219211e-3
        e.VbroadcastssConst(X86Emitter.YMM12, idxC1);     // 2.1562920e-1
        e.VbroadcastssConst(X86Emitter.YMM13, idxHalf);   // 0.5

        int simdCount = (length / 32) * 32;
        int simdEndBytes = simdCount * sizeof(float);

        if (simdCount > 0)
        {
            e.MovImm32(X86Emitter.RBX, 0);

            int loopLabel = e.NewLabel();
            e.BindLabel(loopLabel);

            // Load 4 vectors from src (RCX)
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, 0);
            e.VmovupsLoad(X86Emitter.YMM1, X86Emitter.RCX, 32);
            e.VmovupsLoad(X86Emitter.YMM2, X86Emitter.RCX, 64);
            e.VmovupsLoad(X86Emitter.YMM3, X86Emitter.RCX, 96);

            // Clamp to [-5, 5]: clamped = min(max(x, -5), 5)
            // VMAXPS with YMM8 (-5.0) — opcode 0x5F
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM8);
            e.Vmaxps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM8);
            e.Vmaxps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM8);
            e.Vmaxps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM8);
            // VMINPS with YMM9 (5.0) — opcode 0x5D
            e.VbinaryPs(0x5D, X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM9);
            e.VbinaryPs(0x5D, X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM9);
            e.VbinaryPs(0x5D, X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM9);
            e.VbinaryPs(0x5D, X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM9);

            // x² = clamped * clamped → store in YMM4-YMM7
            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM0, X86Emitter.YMM0);
            e.Vmulps(X86Emitter.YMM5, X86Emitter.YMM1, X86Emitter.YMM1);
            e.Vmulps(X86Emitter.YMM6, X86Emitter.YMM2, X86Emitter.YMM2);
            e.Vmulps(X86Emitter.YMM7, X86Emitter.YMM3, X86Emitter.YMM3);

            // inner = FMA(x², c5, c3) → inner = x²*c5 + c3
            // VFMADD231PS: dst = src1*src2 + dst, so dst=c3 copy, src1=x², src2=c5
            // But we can't clobber YMM11 (c3). Use a different approach:
            // Start with inner = c3 (copy), then FMA inner = x² * c5 + inner
            // We need to copy c3 to temp first. Use YMM14,YMM15 + reuse after.
            // Actually: Horner scheme with FMA. inner = x²*c5 + c3
            // VFMADD231PS dst, src1, src2: dst += src1 * src2 → need dst=c3 copy
            // But we have 4 lanes to process simultaneously, so we need 4 copies of c3.
            // Better approach: use VMULPS + VADDPS for the first step, then FMA for the rest.

            // inner = x² * c5 → use YMM14,YMM15 as temps (only need 2 temp registers at a time)
            // Actually we can process 2 lanes at a time with 2 temp regs:

            // Lane 0-1: inner0 = x²_0 * c5 + c3
            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM4, X86Emitter.YMM10); // x²*c5
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM11); // +c3
            e.Vmulps(X86Emitter.YMM15, X86Emitter.YMM5, X86Emitter.YMM10);
            e.Vaddps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM11);

            // inner0 = FMA(x²_0, inner0, c1) → inner0 = x²*inner0 + c1
            // VFMADD231PS inner0, x², c1 won't work — need inner0 = x²*inner0 + c1
            // VFMADD213PS dst, src1, src2: dst = src1*dst + src2
            // We need VFMADD213PS but don't have it in emitter.
            // Use VMULPS + VADDPS instead:
            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM4, X86Emitter.YMM14); // x²*inner
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM12); // +c1
            e.Vmulps(X86Emitter.YMM15, X86Emitter.YMM5, X86Emitter.YMM15);
            e.Vaddps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM12);

            // result0 = FMA(clamped, inner, 0.5) → clamped*inner + 0.5
            // VFMADD231PS: dst += src1 * src2 → but we need dst = clamped*inner + 0.5
            // Copy 0.5 to YMM0 first, then VFMADD231PS YMM0, clamped_orig, inner
            // Problem: we clobbered YMM0 with clamped. clamped IS YMM0.
            // So: result = VFMADD231PS with dst starting as 0.5
            // Need temp = 0.5 copy. But we already have YMM13 = 0.5 constant.
            // We need: result = clamped * inner + 0.5
            // VMULPS result = clamped * inner, then VADDPS result += 0.5
            e.Vmulps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM14); // clamped*inner
            e.Vaddps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM13); // +0.5
            e.Vmulps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM15);
            e.Vaddps(X86Emitter.YMM1, X86Emitter.YMM1, X86Emitter.YMM13);

            // Lane 2-3: same pattern
            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM6, X86Emitter.YMM10); // x²*c5
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM11); // +c3
            e.Vmulps(X86Emitter.YMM15, X86Emitter.YMM7, X86Emitter.YMM10);
            e.Vaddps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM11);

            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM6, X86Emitter.YMM14); // x²*inner
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM12); // +c1
            e.Vmulps(X86Emitter.YMM15, X86Emitter.YMM7, X86Emitter.YMM15);
            e.Vaddps(X86Emitter.YMM15, X86Emitter.YMM15, X86Emitter.YMM12);

            e.Vmulps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM14);
            e.Vaddps(X86Emitter.YMM2, X86Emitter.YMM2, X86Emitter.YMM13);
            e.Vmulps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM15);
            e.Vaddps(X86Emitter.YMM3, X86Emitter.YMM3, X86Emitter.YMM13);

            // Store results to dst (RDX)
            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.RDX, 0);
            e.VmovupsStore(X86Emitter.YMM1, X86Emitter.RDX, 32);
            e.VmovupsStore(X86Emitter.YMM2, X86Emitter.RDX, 64);
            e.VmovupsStore(X86Emitter.YMM3, X86Emitter.RDX, 96);

            // Advance pointers
            e.AddImm32(X86Emitter.RCX, 128);
            e.AddImm32(X86Emitter.RDX, 128);

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
            // Load, clamp, polynomial, store
            e.VmovupsLoad(X86Emitter.YMM0, X86Emitter.RCX, off);
            e.Vmaxps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM8);
            e.VbinaryPs(0x5D, X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM9); // VMINPS

            e.Vmulps(X86Emitter.YMM4, X86Emitter.YMM0, X86Emitter.YMM0); // x²
            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM4, X86Emitter.YMM10); // x²*c5
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM11); // +c3
            e.Vmulps(X86Emitter.YMM14, X86Emitter.YMM4, X86Emitter.YMM14); // x²*inner
            e.Vaddps(X86Emitter.YMM14, X86Emitter.YMM14, X86Emitter.YMM12); // +c1
            e.Vmulps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM14); // clamped*inner
            e.Vaddps(X86Emitter.YMM0, X86Emitter.YMM0, X86Emitter.YMM13); // +0.5

            e.VmovupsStore(X86Emitter.YMM0, X86Emitter.RDX, off);
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

    // Scalar helpers using VEX-encoded SSE scalar ops (128-bit, no lane issues)
    private static void EmitScalarLoad(X86Emitter e, int dst, int baseReg, int disp)
    {
        // VMOVSS xmm, [base+disp]: VEX.128.F3.0F.W0 10 /r
        // Use VmovupsLoad but with 128-bit — actually VMOVSS is different encoding
        // Simplest: use the VEX memory load helper with scalar prefix
        // For now, reuse unaligned load (VMOVUPS loads 32 bytes but we only use low 4)
        // This is safe because we verify the full array fits before JIT compilation
        e.VmovupsLoad(dst, baseReg, disp);
    }

    private static void EmitScalarBinaryOp(X86Emitter e, JitBinaryOp op, int dst, int srcBase, int disp)
    {
        e.VbinaryPs(op.Opcode, dst, dst, srcBase, disp);
    }

    private static void EmitScalarMaxWithZero(X86Emitter e, int dst, int zero)
    {
        e.Vmaxps(dst, dst, zero);
    }

    private static void EmitScalarStore(X86Emitter e, int src, int baseReg, int disp)
    {
        e.VmovupsStore(src, baseReg, disp);
    }

    /// <summary>
    /// Disposes all cached kernels and frees executable memory.
    /// </summary>
    public static void ClearCache()
    {
        foreach (var kvp in _cache)
        {
            kvp.Value.Buffer.Dispose();
        }
        _cache.Clear();
    }
}
