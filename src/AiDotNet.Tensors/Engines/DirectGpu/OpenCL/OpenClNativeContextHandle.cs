// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>The kind of an OpenCL object a context owns, in the order they are released.</summary>
    internal enum OpenClChildKind
    {
        Kernel = 0,
        Program = 1,
        MemObject = 2,
        CommandQueue = 3,
    }

    /// <summary>
    /// The native objects of one <c>cl_context</c>: the context itself and every kernel, program, memory object and
    /// command queue created in it that nobody has released yet. <see cref="DirectOpenClContext"/> releases what is left
    /// when it is disposed - including when it is reclaimed without being disposed (see its finalizer).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Ownership of each child is CLAIMED, not assumed: the wrapper that releases an object (a kernel's or program's
    /// Dispose, a buffer's retirement) first removes it from this table, and only the remover releases it. A context
    /// reclaimed while some of its buffers are still waiting in the deferred-free queue (a GPU buffer's finalizer hands
    /// its inner buffer to that queue) therefore never releases a memory object twice, whichever of the two runs first.
    /// </para>
    /// <para>
    /// This is deliberately not a <see cref="SafeHandle"/> with its own critical finalizer. A critical finalizer
    /// releases on the finalizer thread, concurrently with the deferred-free queue that the buffer finalizers resurrect
    /// their inner buffers into, and that race freed memory objects and queues still in use (an access violation in
    /// clReleaseMemObject). The context's finalizer instead hands the whole release to that same queue, so it runs on
    /// an ordinary thread, after or before - never during - the buffer releases.
    /// </para>
    /// </remarks>
    internal sealed class OpenClNativeContextHandle
    {
        private readonly ConcurrentDictionary<IntPtr, OpenClChildKind> _children = new();
        private readonly Action<IntPtr>? _onReleased;
        private IntPtr _context;

        internal OpenClNativeContextHandle(IntPtr context, Action<IntPtr>? onReleased)
        {
            _context = context;
            _onReleased = onReleased;
        }

        /// <summary>The <c>cl_context</c>, or zero once released.</summary>
        internal IntPtr Context => _context;

        /// <summary>Number of child objects still owned here (diagnostics and tests).</summary>
        internal int TrackedChildCount => _children.Count;

        /// <summary>Records a child object created in this context.</summary>
        internal void Track(IntPtr child, OpenClChildKind kind)
        {
            if (child != IntPtr.Zero) _children[child] = kind;
        }

        /// <summary>
        /// Claims a child for release by the caller: true when the caller now owns it (it was still tracked), false when
        /// it was already released with the context - the caller must then not touch it.
        /// </summary>
        internal bool TryClaim(IntPtr child) => child != IntPtr.Zero && _children.TryRemove(child, out _);

        /// <summary>Releases every child still owned, in dependency order, then the context. Idempotent.</summary>
        internal void ReleaseAll()
        {
            IntPtr context = _context;
            if (context == IntPtr.Zero) return;
            for (var kind = OpenClChildKind.Kernel; kind <= OpenClChildKind.CommandQueue; kind++)
            {
                foreach (var pair in _children)
                {
                    if (pair.Value != kind || !_children.TryRemove(pair.Key, out _)) continue;
                    // Best-effort: one failure must not leak the rest.
                    try
                    {
                        switch (kind)
                        {
                            case OpenClChildKind.Kernel: OpenClNativeBindings.ReleaseKernel(pair.Key); break;
                            case OpenClChildKind.Program: OpenClNativeBindings.ReleaseProgram(pair.Key); break;
                            case OpenClChildKind.MemObject:
                                DirectOpenClSubmission.Forget(pair.Key);
                                OpenClNativeBindings.ReleaseMemObject(pair.Key);
                                break;
                            case OpenClChildKind.CommandQueue:
                                DirectOpenClContext.LiveQueues.TryRemove(pair.Key, out _);
                                OpenClNativeBindings.ReleaseCommandQueue(pair.Key);
                                break;
                        }
                    }
                    catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or SEHException)
                    {
                    }
                }
            }

            _context = IntPtr.Zero;
            try
            {
                OpenClNativeBindings.ReleaseContext(context);
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or SEHException)
            {
            }
            _onReleased?.Invoke(context);
        }
    }
}
