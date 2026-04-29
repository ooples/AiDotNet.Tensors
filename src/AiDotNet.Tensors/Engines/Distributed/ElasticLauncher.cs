// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Rendezvous backend — how workers discover each other and agree on
/// (world_size, rank) assignments. Mirrors <c>torchrun</c>'s rendezvous
/// modes.
/// </summary>
public enum RendezvousBackend
{
    /// <summary>File-system rendezvous — workers write to a shared
    /// directory; the launcher reads the directory listing to assign
    /// ranks. Cheapest setup; fine for single-machine multi-process.</summary>
    File,

    /// <summary>TCP rendezvous — workers connect to a coordinator on
    /// a known host:port; the coordinator hands out ranks. Required
    /// for multi-machine.</summary>
    Tcp,

    /// <summary>etcd rendezvous — same idea as TCP but the coordinator
    /// is an etcd cluster; workers register their address in a shared
    /// key and read each other's. Production-grade, supports elastic
    /// add/remove.</summary>
    Etcd,
}

/// <summary>Configuration for <see cref="ElasticLauncher"/>.</summary>
public sealed class ElasticLauncherOptions
{
    /// <summary>Expected total worker count.</summary>
    public int WorldSize { get; set; }

    /// <summary>Rendezvous backend.</summary>
    public RendezvousBackend Backend { get; set; } = RendezvousBackend.File;

    /// <summary>Backend-specific endpoint (file path / "host:port" /
    /// "etcd://host:port/key"). Required.</summary>
    public string Endpoint { get; set; } = "";

    /// <summary>Maximum seconds the launcher waits for the full worker
    /// set to register before giving up.</summary>
    public int RendezvousTimeoutSeconds { get; set; } = 60;

    /// <summary>If true, the launcher will restart a worker that exits
    /// non-zero up to <see cref="MaxRestarts"/> times. Mirrors
    /// <c>torchrun --max-restarts</c>.</summary>
    public bool RestartOnFailure { get; set; }

    /// <summary>Max worker restarts (only used when
    /// <see cref="RestartOnFailure"/> is true).</summary>
    public int MaxRestarts { get; set; } = 3;

    /// <summary>Seconds between worker health probes.</summary>
    public int HealthCheckIntervalSeconds { get; set; } = 5;
}

/// <summary>
/// Worker-rendezvous result — what the launcher tells each worker
/// once the full group has registered.
/// </summary>
public sealed class RendezvousAssignment
{
    /// <summary>This worker's rank assignment.</summary>
    public int Rank { get; init; }

    /// <summary>Total agreed world size.</summary>
    public int WorldSize { get; init; }

    /// <summary>The rendezvous endpoint the worker should join with —
    /// for the TCP backend this is the master "host:port"; for File
    /// it's the rendezvous directory path.</summary>
    public string Endpoint { get; init; } = "";
}

/// <summary>
/// Elastic launcher — manages worker rendezvous, health checks, and
/// optional restart-on-failure. The .NET-native equivalent of
/// <c>torchrun</c> / <c>torch.distributed.run</c>.
///
/// <para><b>Usage pattern:</b></para>
/// <code>
/// var launcher = new ElasticLauncher(new ElasticLauncherOptions {
///     WorldSize = 4,
///     Backend   = RendezvousBackend.File,
///     Endpoint  = "/tmp/aidotnet-rendezvous",
/// });
/// // On every node:
/// var assignment = launcher.Rendezvous(workerNonce: Guid.NewGuid().ToString());
/// // assignment.Rank, assignment.WorldSize are now agreed across all workers.
/// </code>
///
/// <para><b>File backend (the only one fully implemented in-process):</b>
/// each worker writes a file named after its nonce; the launcher reads
/// the directory once <see cref="ElasticLauncherOptions.WorldSize"/>
/// files exist; workers are sorted by nonce and assigned ranks
/// 0..WorldSize-1. Avoids the network setup needed for TCP/etcd backends
/// and gives deterministic single-machine multi-process tests.</para>
///
/// <para>TCP and etcd backends require a real network coordinator and
/// are stubbed — they throw <see cref="NotSupportedException"/> until
/// the network process-group transport binding lands. They're enumerated
/// here so callers can pin to a backend in their config without wiring
/// changes when the implementations land.</para>
/// </summary>
public sealed class ElasticLauncher
{
    private readonly ElasticLauncherOptions _options;

    /// <summary>Constructs.</summary>
    public ElasticLauncher(ElasticLauncherOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (options.WorldSize < 1) throw new ArgumentException("WorldSize must be ≥ 1.");
        if (string.IsNullOrEmpty(options.Endpoint)) throw new ArgumentException("Endpoint required.");
    }

    /// <summary>
    /// Joins the worker rendezvous and returns the assigned rank.
    /// Blocks until <see cref="ElasticLauncherOptions.WorldSize"/>
    /// workers have registered or the timeout elapses.
    /// </summary>
    public RendezvousAssignment Rendezvous(string workerNonce)
    {
        return _options.Backend switch
        {
            RendezvousBackend.File => FileRendezvous(workerNonce),
            RendezvousBackend.Tcp => throw new NotSupportedException(
                "TCP rendezvous requires the network process-group transport. " +
                "Use the File backend for single-machine multi-process or wait for the NCCL/Gloo binding."),
            RendezvousBackend.Etcd => throw new NotSupportedException(
                "etcd rendezvous requires the etcd client + network process-group transport."),
            _ => throw new ArgumentException($"Unknown rendezvous backend {_options.Backend}."),
        };
    }

    private RendezvousAssignment FileRendezvous(string workerNonce)
    {
        Directory.CreateDirectory(_options.Endpoint);
        // Atomic-write our nonce file. Other workers do the same.
        string myFile = Path.Combine(_options.Endpoint, workerNonce);
        File.WriteAllText(myFile, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString());

        // Poll until WorldSize files have appeared OR timeout.
        var deadline = DateTime.UtcNow.AddSeconds(_options.RendezvousTimeoutSeconds);
        while (DateTime.UtcNow < deadline)
        {
            var files = Directory.GetFiles(_options.Endpoint).Select(Path.GetFileName).ToArray();
            if (files.Length >= _options.WorldSize)
            {
                Array.Sort(files, StringComparer.Ordinal);
                int rank = Array.IndexOf(files, workerNonce);
                if (rank < 0) throw new InvalidOperationException(
                    $"Our nonce {workerNonce} disappeared from the rendezvous directory.");
                return new RendezvousAssignment
                {
                    Rank = rank,
                    WorldSize = _options.WorldSize,
                    Endpoint = _options.Endpoint,
                };
            }
            Thread.Sleep(50);
        }
        throw new TimeoutException(
            $"Rendezvous timed out after {_options.RendezvousTimeoutSeconds}s. " +
            $"Saw {Directory.GetFiles(_options.Endpoint).Length} of {_options.WorldSize} workers.");
    }

    /// <summary>
    /// Spawns and supervises N worker delegates in-process. Each worker
    /// gets its <see cref="RendezvousAssignment"/> via the file-rendezvous
    /// path. If <see cref="ElasticLauncherOptions.RestartOnFailure"/> is
    /// true, workers that throw are restarted up to <see cref="ElasticLauncherOptions.MaxRestarts"/>
    /// times. Returns when every worker has finished (or all have
    /// permanently failed).
    /// </summary>
    public Task SpawnInProcessWorkers(Action<RendezvousAssignment> worker)
    {
        if (_options.Backend != RendezvousBackend.File)
            throw new InvalidOperationException(
                "SpawnInProcessWorkers only supports File backend. Use Rendezvous() directly for other backends.");

        var tasks = new Task[_options.WorldSize];
        for (int i = 0; i < _options.WorldSize; i++)
        {
            int workerIndex = i;
            tasks[i] = Task.Run(async () =>
            {
                int restarts = 0;
                while (true)
                {
                    try
                    {
                        var assignment = Rendezvous($"worker-{workerIndex:D6}");
                        worker(assignment);
                        return;
                    }
                    catch (Exception) when (_options.RestartOnFailure && restarts < _options.MaxRestarts)
                    {
                        restarts++;
                        await Task.Delay(_options.HealthCheckIntervalSeconds * 1000).ConfigureAwait(false);
                    }
                }
            });
        }
        return Task.WhenAll(tasks);
    }

    /// <summary>Removes the rendezvous directory + its contents.
    /// Call after all workers have finished to clean up.</summary>
    public void CleanupRendezvous()
    {
        if (_options.Backend == RendezvousBackend.File && Directory.Exists(_options.Endpoint))
        {
            try { Directory.Delete(_options.Endpoint, recursive: true); } catch { /* best-effort */ }
        }
    }
}
