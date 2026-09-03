// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// A native OpenCL memory object whose queue ownership is tracked at submission time.
    /// Queue handles and memory-object handles are process-local native identities, not user policy strings.
    /// </summary>
    internal interface IDirectOpenClMemoryObject
    {
        IntPtr NativeHandle { get; }
        DirectOpenClContext OwningContext { get; }

        /// <summary>
        /// Last queue on which a command using this memory object was submitted. Access is serialized by
        /// <see cref="DirectOpenClSubmission.Gate"/>.
        /// </summary>
        IntPtr LastSubmissionQueue { get; set; }
    }

    /// <summary>
    /// Serializes OpenCL submission bookkeeping with the native enqueue it describes. The existing kernel
    /// submit lock protected shared <c>cl_kernel</c> argument state, but buffer dependencies and retirement
    /// must be committed in that same critical section or another thread can reclaim a buffer between the
    /// bookkeeping update and <c>clEnqueue*</c>.
    /// </summary>
    internal static class DirectOpenClSubmission
    {
        internal static readonly object Gate = new object();

        private static readonly ConcurrentDictionary<IntPtr, IDirectOpenClMemoryObject> MemoryObjects = new();

        [ThreadStatic]
        private static List<IDirectOpenClMemoryObject>? _directSubmissionMemories;

        internal static void Register(IDirectOpenClMemoryObject memory)
        {
            IntPtr handle = memory.NativeHandle;
            if (handle == IntPtr.Zero)
                throw new ArgumentException("Cannot register a null OpenCL memory-object handle.", nameof(memory));
            if (!MemoryObjects.TryAdd(handle, memory))
                throw new InvalidOperationException("An OpenCL memory-object handle is already registered.");
        }

        internal static IDirectOpenClMemoryObject? Resolve(IntPtr handle)
            => handle != IntPtr.Zero && MemoryObjects.TryGetValue(handle, out var memory) ? memory : null;

        internal static void Unregister(IntPtr handle, IDirectOpenClMemoryObject memory)
        {
            if (handle == IntPtr.Zero) return;
            if (MemoryObjects.TryGetValue(handle, out var registered) && ReferenceEquals(registered, memory))
                MemoryObjects.TryRemove(handle, out _);
        }

        internal static List<IDirectOpenClMemoryObject> GetDirectSubmissionMemories(
            IDirectOpenClMemoryObject first,
            IDirectOpenClMemoryObject? second = null,
            IDirectOpenClMemoryObject? third = null)
        {
            var memories = _directSubmissionMemories
                ??= new List<IDirectOpenClMemoryObject>(2);
            if (memories.Count != 0)
                throw new InvalidOperationException("Nested direct OpenCL submission bookkeeping is not supported.");
            memories.Add(first);
            if (second is not null && !ReferenceEquals(first, second)) memories.Add(second);
            if (third is not null && !ReferenceEquals(first, third) && !ReferenceEquals(second, third))
                memories.Add(third);
            return memories;
        }

        internal static void ReleaseDirectSubmissionMemories(List<IDirectOpenClMemoryObject> memories)
            => memories.Clear();

        /// <summary>
        /// Creates device-side dependencies for memory objects moving between independent queues. The common
        /// same-queue path allocates nothing and enqueues no event. Call only while holding <see cref="Gate"/>.
        /// </summary>
        internal static SubmissionWaitList PrepareLocked(
            IntPtr destinationQueue,
            IReadOnlyList<IDirectOpenClMemoryObject> memories)
        {
            if (destinationQueue == IntPtr.Zero)
                throw new ArgumentException("A submission requires a valid OpenCL command queue.", nameof(destinationQueue));

            List<IntPtr>? sourceQueues = null;
            for (int i = 0; i < memories.Count; i++)
            {
                var memory = memories[i];
                IntPtr sourceQueue = memory.LastSubmissionQueue;
                if (sourceQueue == IntPtr.Zero || sourceQueue == destinationQueue
                    || memory.OwningContext.IsQueueKnownComplete(sourceQueue))
                    continue;

                sourceQueues ??= new List<IntPtr>(2);
                bool seen = false;
                for (int j = 0; j < sourceQueues.Count; j++)
                    if (sourceQueues[j] == sourceQueue) { seen = true; break; }
                if (!seen) sourceQueues.Add(sourceQueue);
            }

            if (sourceQueues is null)
                return default;

            var events = new IntPtr[sourceQueues.Count];
            int created = 0;
            try
            {
                for (int i = 0; i < sourceQueues.Count; i++)
                {
                    IntPtr sourceQueue = sourceQueues[i];
                    int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(
                        sourceQueue, 0, null, out IntPtr markerEvent);
                    if (err != OpenClNativeBindings.CL_SUCCESS || markerEvent == IntPtr.Zero)
                    {
                        // A queue wrapper synchronizes before it is released and records that fact with the
                        // context. If a foreign/driver queue disappeared without that notification, fail closed:
                        // silently omitting the dependency would allow the consumer to race its producer.
                        throw new InvalidOperationException(
                            $"Failed to fence OpenCL producer queue before a cross-queue buffer use: {err}.");
                    }

                    events[created++] = markerEvent;
                    // A marker held only by the host event reference is not guaranteed to leave the driver's
                    // submission batch promptly. Flush starts it without waiting and preserves GPU overlap.
                    OpenClNativeBindings.Flush(sourceQueue);
                }

                IntPtr waitList = Marshal.AllocHGlobal(checked(created * IntPtr.Size));
                for (int i = 0; i < created; i++)
                    Marshal.WriteIntPtr(waitList, i * IntPtr.Size, events[i]);
                return new SubmissionWaitList(events, created, waitList);
            }
            catch
            {
                for (int i = 0; i < created; i++)
                    OpenClNativeBindings.ReleaseEvent(events[i]);
                throw;
            }
        }

        /// <summary>Commits queue ownership after, and only after, a successful native enqueue.</summary>
        internal static void CommitLocked(
            IntPtr destinationQueue,
            IReadOnlyList<IDirectOpenClMemoryObject> memories)
        {
            for (int i = 0; i < memories.Count; i++)
                memories[i].LastSubmissionQueue = destinationQueue;
        }

        internal readonly struct SubmissionWaitList : IDisposable
        {
            private readonly IntPtr[]? _events;
            private readonly int _eventCount;

            internal uint Count => (uint)_eventCount;
            internal IntPtr Pointer { get; }

            internal SubmissionWaitList(IntPtr[] events, int eventCount, IntPtr pointer)
            {
                _events = events;
                _eventCount = eventCount;
                Pointer = pointer;
            }

            /// <summary>
            /// Inserts the wait list into a destination queue for native libraries, such as CLBlast, whose
            /// API accepts a queue but exposes no OpenCL event wait-list parameter. The following library
            /// enqueue is ordered after this bridge marker because AiDotNet creates in-order queues.
            /// </summary>
            internal void EnqueueBridgeMarker(IntPtr destinationQueue)
            {
                if (_eventCount == 0 || _events is null) return;
                int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(
                    destinationQueue, (uint)_eventCount, _events, out IntPtr bridgeEvent);
                if (err != OpenClNativeBindings.CL_SUCCESS || bridgeEvent == IntPtr.Zero)
                    throw new InvalidOperationException(
                        $"Failed to enqueue an OpenCL cross-queue dependency marker: {err}.");
                OpenClNativeBindings.ReleaseEvent(bridgeEvent);
            }

            public void Dispose()
            {
                if (Pointer != IntPtr.Zero)
                    Marshal.FreeHGlobal(Pointer);
                if (_events is null) return;
                for (int i = 0; i < _eventCount; i++)
                    if (_events[i] != IntPtr.Zero) OpenClNativeBindings.ReleaseEvent(_events[i]);
            }
        }
    }
}
