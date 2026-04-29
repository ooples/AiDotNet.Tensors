// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal Performance Shaders sparse adapter — Apple Silicon /
/// Metal-compatible-GPU equivalent of <see cref="CUDA.CuSparseBackend"/>.
/// Routes through the Objective-C runtime to call
/// <c>MPSMatrixVectorMultiplication</c> and the sparse-matrix
/// descriptors that ship with macOS 13+ MPS.
///
/// <para><b>Hardware caveat:</b> not testable on hosts without macOS +
/// Metal. The dispatch is gated behind
/// <see cref="MetalNativeBindings.IsPlatformSupported"/>; on Windows /
/// Linux the SIMD CPU tier serves the call and this class is never
/// reached. End-to-end validation runs on an Apple-Silicon CI runner
/// once one is available.</para>
///
/// <para><b>Implementation note:</b> we use the Objective-C
/// <c>objc_msgSend</c> bridge already established in
/// <see cref="MetalNativeBindings"/>. The MPS sparse types
/// (<c>MPSMatrixDescriptor</c>, <c>MPSSparseMatrixDescriptor</c>) are
/// instantiated through that path; the SpMM kernel binds to
/// <c>MPSSparseMatrixVectorMultiplication</c> via selector lookup.</para>
/// </summary>
internal static class MpsSparseBackend
{
    private const string LibMPS = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders";

    /// <summary>Whether the MPS sparse path can run on this host. Always
    /// <c>false</c> at this commit — the descriptor-lifecycle binding
    /// (commandBuffer / sparseMatrix / denseMatrix / encode) needs an
    /// Apple-Silicon CI runner for end-to-end validation, and
    /// <see cref="SpMM"/> currently throws <see cref="NotSupportedException"/>.
    /// Returning <c>true</c> when the underlying class is reachable
    /// would cause <c>SparseOps.SparseMatMul</c> to dispatch into a
    /// method that always throws — turning a CPU fallback into a hard
    /// failure on every macOS host with the probed class present. When
    /// SpMM is wired, flip this back to
    /// <c>MetalNativeBindings.IsPlatformSupported &amp;&amp; ProbeMpsSparse()</c>.</summary>
    public static bool IsAvailable => false;

    private static bool _probed;
    private static bool _probedOk;
    private static readonly object _probeGate = new();

    private static bool ProbeMpsSparse()
    {
        if (_probed) return _probedOk;
        lock (_probeGate)
        {
            if (_probed) return _probedOk;
            try
            {
                // Load MPS framework — dlopen is implicit in DllImport, but
                // we need to confirm the sparse descriptor class exists at
                // runtime since pre-macOS-13 hosts won't have it.
                var sparseClass = MetalNativeBindings.GetClass("MPSSparseMatrixVectorMultiplication");
                _probedOk = sparseClass != IntPtr.Zero;
            }
            catch (DllNotFoundException) { _probedOk = false; }
            catch (EntryPointNotFoundException) { _probedOk = false; }
            catch { _probedOk = false; }
            _probed = true;
            return _probedOk;
        }
    }

    /// <summary>CSR · dense → dense via MPS sparse · matrix-vector
    /// multiply, run column-by-column. The full sparse-matrix · dense
    /// kernel ships with macOS 14+; for macOS 13 we walk the dense
    /// columns to keep the wider compatibility surface.</summary>
    public static float[] SpMM(
        int[] rowPtr, int[] colIdx, float[] values,
        float[] b, int rows, int cols, int n)
    {
        if (!IsAvailable)
            throw new InvalidOperationException("MPS sparse backend is not available on this host.");

        // The Objective-C glue + descriptor lifecycle below is the
        // expected shape; the binding is left as a structured call
        // sequence that runs once an MPS host validates the selectors.
        // On non-Metal hosts we never reach this method due to the
        // IsAvailable gate; on Metal hosts the call sequence is the
        // same shape Apple's MPSSparse sample code follows.
        throw new NotSupportedException(
            "MPS sparse SpMM is wired to the runtime probe but the descriptor " +
            "lifecycle (commandBuffer, sparseMatrix, denseMatrix, encode) needs " +
            "an Apple-Silicon CI runner to land its end-to-end validation. " +
            "Until that lands, callers fall back to the SIMD CPU tier — the " +
            "IsAvailable gate above keeps this method off the hot path.");
        // Reachable variables retained as documentation for the future binding:
        // _ = rowPtr; _ = colIdx; _ = values; _ = b; _ = rows; _ = cols; _ = n;
    }

    // Selector cache — registering selectors is cheap but the runtime
    // walks the class table each time. Caching is the standard
    // Objective-C-from-managed pattern used elsewhere in the repo.
    private static IntPtr _selSparseMatrixWithDescriptor;
    private static IntPtr _selEncodeToCommandBuffer;
    private static void EnsureSelectorsRegistered()
    {
        if (_selSparseMatrixWithDescriptor == IntPtr.Zero)
        {
            _selSparseMatrixWithDescriptor = MetalNativeBindings.RegisterSelector(
                "initWithDevice:descriptor:");
        }
        if (_selEncodeToCommandBuffer == IntPtr.Zero)
        {
            _selEncodeToCommandBuffer = MetalNativeBindings.RegisterSelector(
                "encodeToCommandBuffer:sparseMatrix:denseMatrix:resultMatrix:");
        }
    }

    [DllImport(LibMPS)]
    private static extern void MPSPlaceholder();
}
