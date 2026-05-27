using System;
#if NET5_0_OR_GREATER
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409) Phase J2 — JIT emitter for the AVX2 FP64 4×8 packed microkernel.
///
/// <para>
/// Emits a <see cref="System.Reflection.Emit.DynamicMethod"/> whose IL calls the
/// AVX2/FMA intrinsic methods (<c>Avx.LoadVector256</c>, <c>Fma.MultiplyAdd</c>,
/// <c>Vector256.Create</c>, <c>Avx.Store</c>) — RyuJIT intrinsifies those calls
/// when it compiles the method, so the result is real AVX2 machine code. The
/// payoff over the hand-written <see cref="Avx2Fp64_4x8.Run"/> is that the emitted
/// IL is <b>fully unrolled and straight-line for a specific Kc</b>, with constant
/// packed-A/packed-B offsets and no re-rollable loop. JIT-disasm (#409 S.3) proved
/// RyuJIT re-rolls a source-level <c>for</c> unroll back into a single-step loop,
/// blocking the scheduler from hoisting the next steps' loads; emitted straight-
/// line code can't be re-rolled, so the scheduler hides the load-to-use latency.
/// </para>
///
/// <para>
/// Same compute and op-order as <see cref="Avx2Fp64_4x8.Run"/> (4 rows × 8 cols,
/// each row split lo/hi, C read-modify-write, accumulate over Kc), so the result
/// is bit-identical. Guarded by <see cref="NativeAotDetector.IsDynamicCodeSupported"/>
/// at the call site (<see cref="JittedKernelCache"/>).
/// </para>
///
/// <para>
/// <b>Measured perf finding (#409 J2 — important).</b> Emitting the kernel
/// <i>fully unrolled</i> for a large Kc (here Kc=256 = ~2k FMAs straight-line)
/// REGRESSES badly — ~6.6 vs the hand-written ~21–35 GFLOPS — because RyuJIT
/// spills heavily across the giant single basic block. A <i>blocked</i> emit
/// (unroll-by-N inside a loop with runtime-k offsets) would instead be
/// instruction-for-instruction identical to the hand-written explicit unroll
/// (#409 S.3), so it can only MATCH it, not beat it. Net: <b>microkernel-level
/// JIT emission gives no throughput upside over the hand-written 4×8 kernel</b>
/// — the ~35 GFLOPS (80% of the managed FMA ceiling) hand-written form is the
/// practical .NET limit for this shape. The real remaining JIT lever is at the
/// whole-GEMM level (J3+: emit a shape-specialized full GEMM that removes the
/// per-tile dispatch/strategy overhead), not the microkernel. This emitter is
/// retained as the verified J2 foundation (correct, bit-exact) for that work;
/// full-unroll mode is therefore appropriate only for SMALL Kc.
/// </para>
/// </summary>
internal static class Fp64MicrokernelEmitter
{
    /// <summary>Row tile (matches <see cref="Avx2Fp64_4x8.Mr"/>).</summary>
    internal const int Mr = 4;
    /// <summary>Column tile (matches <see cref="Avx2Fp64_4x8.Nr"/>).</summary>
    internal const int Nr = 8;

    /// <summary>
    /// Emitted packed-B microkernel: accumulate packedA·packedB into the
    /// C[0..4, 0..8] tile over the Kc baked into the emitted method. C is
    /// read-modify-write. Pointers are raw (caller pins).
    /// </summary>
    internal unsafe delegate void PackedKernel(double* packedA, double* packedB, double* c, int ldc);

#if NET5_0_OR_GREATER
    private static readonly MethodInfo s_loadV =
        typeof(Avx).GetMethod(nameof(Avx.LoadVector256), new[] { typeof(double).MakePointerType() })
        ?? throw new MissingMethodException("Avx.LoadVector256(double*)");
    private static readonly MethodInfo s_storeV =
        typeof(Avx).GetMethod(nameof(Avx.Store), new[] { typeof(double).MakePointerType(), typeof(Vector256<double>) })
        ?? throw new MissingMethodException("Avx.Store(double*, Vector256<double>)");
    private static readonly MethodInfo s_fma =
        typeof(Fma).GetMethod(nameof(Fma.MultiplyAdd), new[] { typeof(Vector256<double>), typeof(Vector256<double>), typeof(Vector256<double>) })
        ?? throw new MissingMethodException("Fma.MultiplyAdd(Vector256<double> x3)");
    private static readonly MethodInfo s_createV =
        typeof(Vector256).GetMethod(nameof(Vector256.Create), new[] { typeof(double) })
        ?? throw new MissingMethodException("Vector256.Create(double)");

    private const int ElemBytes = sizeof(double); // 8

    /// <summary>
    /// Emit a fully-unrolled FP64 4×8 packed microkernel specialized to <paramref name="kc"/>.
    /// </summary>
    internal static PackedKernel Emit(int kc)
    {
        if (kc <= 0) throw new ArgumentOutOfRangeException(nameof(kc));

        var dm = new DynamicMethod(
            name: $"Fp64_4x8_packed_kc{kc}",
            returnType: typeof(void),
            parameterTypes: new[]
            {
                typeof(double).MakePointerType(), // 0: packedA
                typeof(double).MakePointerType(), // 1: packedB
                typeof(double).MakePointerType(), // 2: c
                typeof(int),                      // 3: ldc
            },
            restrictedSkipVisibility: true);

        var il = dm.GetILGenerator();

        // Locals 0..7: accumulators acc[r,h] = local[r*2 + h].
        var acc = new LocalBuilder[8];
        for (int i = 0; i < 8; i++) acc[i] = il.DeclareLocal(typeof(Vector256<double>));
        // Operand scratch.
        var bLo = il.DeclareLocal(typeof(Vector256<double>));
        var bHi = il.DeclareLocal(typeof(Vector256<double>));
        var a0 = il.DeclareLocal(typeof(Vector256<double>));
        var a1 = il.DeclareLocal(typeof(Vector256<double>));
        var a2 = il.DeclareLocal(typeof(Vector256<double>));
        var a3 = il.DeclareLocal(typeof(Vector256<double>));

        // --- Load 8 accumulators from C (read-modify-write). ---
        for (int r = 0; r < Mr; r++)
        {
            EmitCPtr(il, r, hOffsetElems: 0); il.Emit(OpCodes.Call, s_loadV); il.Emit(OpCodes.Stloc, acc[r * 2 + 0]);
            EmitCPtr(il, r, hOffsetElems: 4); il.Emit(OpCodes.Call, s_loadV); il.Emit(OpCodes.Stloc, acc[r * 2 + 1]);
        }

        // --- Unrolled K-loop: acc += broadcast(A[k,r]) * B[k, lo|hi]. ---
        for (int k = 0; k < kc; k++)
        {
            // bLo / bHi: packedB row k, halves at +0 and +4 elems.
            EmitConstPtr(il, argB: true, elemOffset: k * Nr + 0); il.Emit(OpCodes.Call, s_loadV); il.Emit(OpCodes.Stloc, bLo);
            EmitConstPtr(il, argB: true, elemOffset: k * Nr + 4); il.Emit(OpCodes.Call, s_loadV); il.Emit(OpCodes.Stloc, bHi);
            // a0..a3: broadcast packedA[k*Mr + r].
            EmitBroadcastA(il, k * Mr + 0); il.Emit(OpCodes.Stloc, a0);
            EmitBroadcastA(il, k * Mr + 1); il.Emit(OpCodes.Stloc, a1);
            EmitBroadcastA(il, k * Mr + 2); il.Emit(OpCodes.Stloc, a2);
            EmitBroadcastA(il, k * Mr + 3); il.Emit(OpCodes.Stloc, a3);

            EmitFma(il, a0, bLo, acc[0]); // acc0_lo
            EmitFma(il, a0, bHi, acc[1]); // acc0_hi
            EmitFma(il, a1, bLo, acc[2]);
            EmitFma(il, a1, bHi, acc[3]);
            EmitFma(il, a2, bLo, acc[4]);
            EmitFma(il, a2, bHi, acc[5]);
            EmitFma(il, a3, bLo, acc[6]);
            EmitFma(il, a3, bHi, acc[7]);
        }

        // --- Store 8 accumulators back to C. ---
        for (int r = 0; r < Mr; r++)
        {
            EmitCPtr(il, r, hOffsetElems: 0); il.Emit(OpCodes.Ldloc, acc[r * 2 + 0]); il.Emit(OpCodes.Call, s_storeV);
            EmitCPtr(il, r, hOffsetElems: 4); il.Emit(OpCodes.Ldloc, acc[r * 2 + 1]); il.Emit(OpCodes.Call, s_storeV);
        }

        il.Emit(OpCodes.Ret);

        return (PackedKernel)dm.CreateDelegate(typeof(PackedKernel));
    }

    /// <summary>acc = Fma.MultiplyAdd(mul1, mul2, acc).</summary>
    private static void EmitFma(ILGenerator il, LocalBuilder mul1, LocalBuilder mul2, LocalBuilder acc)
    {
        il.Emit(OpCodes.Ldloc, mul1);
        il.Emit(OpCodes.Ldloc, mul2);
        il.Emit(OpCodes.Ldloc, acc);
        il.Emit(OpCodes.Call, s_fma);
        il.Emit(OpCodes.Stloc, acc);
    }

    /// <summary>Push Vector256.Create(packedA[elemOffset]) onto the stack.</summary>
    private static void EmitBroadcastA(ILGenerator il, int elemOffset)
    {
        il.Emit(OpCodes.Ldarg_0);                 // packedA
        EmitAddConstElems(il, elemOffset);        // + elemOffset*8 bytes
        il.Emit(OpCodes.Ldind_R8);                // load double value
        il.Emit(OpCodes.Call, s_createV);         // -> Vector256<double>
    }

    /// <summary>Push (packedB + elemOffset) as a double* (for LoadVector256).</summary>
    private static void EmitConstPtr(ILGenerator il, bool argB, int elemOffset)
    {
        il.Emit(argB ? OpCodes.Ldarg_1 : OpCodes.Ldarg_0);
        EmitAddConstElems(il, elemOffset);
    }

    /// <summary>Push (c + (r*ldc + hOffsetElems) elements) as a double*.</summary>
    private static void EmitCPtr(ILGenerator il, int r, int hOffsetElems)
    {
        il.Emit(OpCodes.Ldarg_2);                 // c
        // element offset = r*ldc + hOffsetElems
        if (r != 0)
        {
            il.Emit(OpCodes.Ldarg_3);             // ldc
            EmitLdcI4(il, r);
            il.Emit(OpCodes.Mul);                 // r*ldc
            if (hOffsetElems != 0) { EmitLdcI4(il, hOffsetElems); il.Emit(OpCodes.Add); }
        }
        else
        {
            if (hOffsetElems == 0) return;        // offset 0 — pointer is c
            EmitLdcI4(il, hOffsetElems);
        }
        // bytes = elems * 8
        EmitLdcI4(il, ElemBytes);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Conv_I);                  // int -> native int
        il.Emit(OpCodes.Add);                     // c + byteOffset
    }

    /// <summary>Add a compile-time-constant element offset (in doubles) to the pointer on the stack.</summary>
    private static void EmitAddConstElems(ILGenerator il, int elemOffset)
    {
        if (elemOffset == 0) return;
        EmitLdcI4(il, elemOffset * ElemBytes);
        il.Emit(OpCodes.Conv_I);                  // int -> native int
        il.Emit(OpCodes.Add);
    }

    private static void EmitLdcI4(ILGenerator il, int v)
    {
        switch (v)
        {
            case 0: il.Emit(OpCodes.Ldc_I4_0); break;
            case 1: il.Emit(OpCodes.Ldc_I4_1); break;
            case 2: il.Emit(OpCodes.Ldc_I4_2); break;
            case 3: il.Emit(OpCodes.Ldc_I4_3); break;
            case 4: il.Emit(OpCodes.Ldc_I4_4); break;
            case 8: il.Emit(OpCodes.Ldc_I4_8); break;
            default:
                if (v >= sbyte.MinValue && v <= sbyte.MaxValue) il.Emit(OpCodes.Ldc_I4_S, (sbyte)v);
                else il.Emit(OpCodes.Ldc_I4, v);
                break;
        }
    }
#endif
}
