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

    // Cache compiled kernels by (opId, aligned, length)
    private static readonly ConcurrentDictionary<long, (ExecutableBuffer Buffer, Delegate Kernel)> _cache = new();

    // Pack (opId, aligned, length) into a single long key
    private static long MakeKey(int opId, bool aligned, int length)
        => ((long)opId << 33) | ((aligned ? 1L : 0L) << 32) | (uint)length;

    // Separate key space for unary/fused ops (opId > 100 to avoid collisions)
    private const int OP_RELU = 101;
    private const int OP_FUSED_ADD_RELU = 102;

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
        if (_cache.TryGetValue(key, out var cached))
            return (BinaryKernel)cached.Kernel;

        var buffer = aligned
            ? GenerateBinaryKernelAligned(length, op)
            : GenerateBinaryKernelUnaligned(length, op);
        var kernel = buffer.CreateDelegate<BinaryKernel>();
        _cache.TryAdd(key, (buffer, kernel));
        return kernel;
    }

    /// <summary>
    /// Gets or compiles a JIT kernel for ReLU: dst[i] = max(src[i], 0).
    /// </summary>
    /// <param name="length">Number of floats to process.</param>
    /// <param name="aligned">True for aligned NT stores, false for VMOVUPS.</param>
    public static unsafe UnaryKernel GetReLUKernel(int length, bool aligned = false)
    {
        long key = MakeKey(OP_RELU, aligned, length);
        if (_cache.TryGetValue(key, out var cached))
            return (UnaryKernel)cached.Kernel;

        var buffer = aligned
            ? GenerateReLUKernelAligned(length)
            : GenerateReLUKernelUnaligned(length);
        var kernel = buffer.CreateDelegate<UnaryKernel>();
        _cache.TryAdd(key, (buffer, kernel));
        return kernel;
    }

    /// <summary>
    /// Gets or compiles a fused Add+ReLU kernel: dst[i] = max(a[i] + b[i], 0).
    /// Eliminates one full array pass compared to separate Add then ReLU.
    /// </summary>
    public static unsafe BinaryKernel GetFusedAddReLUKernel(int length)
    {
        long key = MakeKey(OP_FUSED_ADD_RELU, false, length);
        if (_cache.TryGetValue(key, out var cached))
            return (BinaryKernel)cached.Kernel;

        var buffer = GenerateFusedAddReLUKernel(length);
        var kernel = buffer.CreateDelegate<BinaryKernel>();
        _cache.TryAdd(key, (buffer, kernel));
        return kernel;
    }

    /// <summary>
    /// Check if the current CPU supports AVX2+FMA (required for our JIT kernels).
    /// </summary>
    public static bool IsSupported
    {
        get
        {
#if NET5_0_OR_GREATER
            return System.Runtime.Intrinsics.X86.Avx2.IsSupported &&
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
    /// Emits remainder loop (groups of 8 floats) for binary ops.
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
    }

    /// <summary>
    /// Emits remainder loop for ReLU (groups of 8 floats).
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
