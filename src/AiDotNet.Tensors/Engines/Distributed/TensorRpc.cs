// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Remote tensor method invocation over a process group. Mirrors
/// <c>torch.distributed.rpc</c>'s <c>rpc_sync</c> / <c>rpc_async</c>
/// surface — register a named handler on every rank, then any rank
/// can invoke it on any other rank with tensor arguments and receive
/// the tensor result.
///
/// <para><b>Wire encoding (in-process backend):</b> we cheat — every
/// rank shares an in-process registry, and the RPC call directly
/// invokes the target rank's handler under a lock. A real Gloo / NCCL
/// transport would prepend a (handler-name, request-id, arg-count,
/// per-arg shape) header before the tensor payload; the in-process
/// path is byte-identical at the application layer.</para>
/// </summary>
public sealed class TensorRpc<T> : IDisposable
{
    private readonly IProcessGroup _group;
    private readonly ConcurrentDictionary<string, Func<Tensor<T>[], Tensor<T>>> _handlers = new();
    private readonly InProcessRpcRegistry? _registry;
    // Names this instance has published into the registry — used at
    // Dispose to unregister only OUR handlers, not every handler this
    // rank ever published. Without this, disposing one TensorRpc
    // instance wipes registrations owned by sibling instances on the
    // same rank.
    private readonly HashSet<string> _publishedNames = new();
    private bool _disposed;

    /// <summary>The process group this RPC channel runs over.</summary>
    public IProcessGroup Group => _group;

    /// <summary>Constructs an RPC channel.</summary>
    public TensorRpc(IProcessGroup group)
    {
        _group = group ?? throw new ArgumentNullException(nameof(group));
        _registry = group is InProcessGroup ? InProcessRpcRegistry.ForGroup(group) : null;
    }

    /// <summary>Registers a handler on this rank. Every rank must
    /// register the same handler name (collective contract) so any
    /// rank can dispatch to any other.</summary>
    public void Register(string name, Func<Tensor<T>[], Tensor<T>> handler)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorRpc<T>));
        _handlers[name] = handler;
        _registry?.PublishHandler<T>(_group.Rank, name, args => handler(args));
        lock (_publishedNames) _publishedNames.Add(name);
    }

    /// <summary>
    /// Synchronous remote call: dispatches <paramref name="name"/> on
    /// rank <paramref name="dst"/> with <paramref name="args"/>;
    /// returns the result. Blocks until the remote handler completes.
    /// </summary>
    public Tensor<T> RpcSync(int dst, string name, params Tensor<T>[] args)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorRpc<T>));
        if (_registry is not null) return _registry.Invoke(dst, name, args);

        // Network-transport path (placeholder; real implementation prepends
        // a header + sends args, then recvs the result).
        throw new NotSupportedException(
            "Network RPC transport not implemented in this build. Use the InProcessGroup backend " +
            "or wait for the NCCL/Gloo transport binding.");
    }

    /// <summary>Async variant — kicks off the call, returns a Task.</summary>
    public Task<Tensor<T>> RpcAsync(int dst, string name, params Tensor<T>[] args)
        => Task.Run(() => RpcSync(dst, name, args));

    /// <summary>Disposes — unregisters this instance's handlers from
    /// the in-process registry. Sibling TensorRpc instances on the
    /// same rank keep their registrations; only the names this
    /// instance published get removed.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_registry is null) return;
        string[] names;
        lock (_publishedNames) names = _publishedNames.ToArray();
        foreach (var n in names) _registry.Unregister(_group.Rank, n);
    }
}

/// <summary>
/// Process-shared registry of RPC handlers, keyed by group identity.
/// One registry per InProcessGroup coordinator — looked up by reference
/// equality on the IProcessGroup so isolated test groups don't share
/// handler tables.
/// </summary>
internal sealed class InProcessRpcRegistry
{
    private static readonly ConditionalWeakTable<object, InProcessRpcRegistry> _registries = new();
    private static readonly object _registryGate = new();

    private readonly Dictionary<(int Rank, string Name), Delegate> _handlers = new();
    private readonly object _gate = new();

    public static InProcessRpcRegistry ForGroup(IProcessGroup group)
    {
        // Key by the InProcessGroupCoordinator so every rank handle from
        // one InProcessGroup.Create() session shares one registry. Without
        // this all ranks would have isolated tables and Invoke would
        // never find a handler published by another rank.
        var coord = ((InProcessGroup)group).Coordinator;
        lock (_registryGate)
        {
            if (!_registries.TryGetValue(coord, out var reg))
            {
                reg = new InProcessRpcRegistry();
                _registries.Add(coord, reg);
            }
            return reg;
        }
    }

    public void PublishHandler<T>(int rank, string name, Func<Tensor<T>[], Tensor<T>> handler)
    {
        lock (_gate) _handlers[(rank, name)] = handler;
    }

    public Tensor<T> Invoke<T>(int dstRank, string name, Tensor<T>[] args)
    {
        Func<Tensor<T>[], Tensor<T>>? handler;
        lock (_gate)
        {
            _handlers.TryGetValue((dstRank, name), out var raw);
            handler = raw as Func<Tensor<T>[], Tensor<T>>;
        }
        if (handler is null)
            throw new InvalidOperationException(
                $"No RPC handler registered on rank {dstRank} with name '{name}'. " +
                "Every rank must call Register before any rank invokes.");
        return handler(args);
    }

    public void UnregisterRank(int rank)
    {
        lock (_gate)
        {
            var keys = new List<(int, string)>();
            foreach (var k in _handlers.Keys) if (k.Rank == rank) keys.Add(k);
            foreach (var k in keys) _handlers.Remove(k);
        }
    }

    /// <summary>Removes a single (rank, name) registration. Used by
    /// <see cref="TensorRpc{T}.Dispose"/> so sibling instances'
    /// registrations stay live.</summary>
    public void Unregister(int rank, string name)
    {
        lock (_gate) _handlers.Remove((rank, name));
    }
}
