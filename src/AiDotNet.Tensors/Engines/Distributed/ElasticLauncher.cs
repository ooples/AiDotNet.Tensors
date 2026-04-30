// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
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
        if (string.IsNullOrWhiteSpace(workerNonce))
            throw new ArgumentException("Worker nonce must be a non-empty / non-whitespace string.", nameof(workerNonce));
        // Reject control characters across every backend — empty / control
        // chars sort before everything and would let a hostile worker
        // grab rank 0 by passing "\0" or "" as its nonce.
        for (int i = 0; i < workerNonce.Length; i++)
        {
            if (char.IsControl(workerNonce[i]))
                throw new ArgumentException(
                    $"Worker nonce contains a control character at position {i}.", nameof(workerNonce));
        }
        return _options.Backend switch
        {
            RendezvousBackend.File => FileRendezvous(workerNonce),
            RendezvousBackend.Tcp => TcpRendezvous(workerNonce),
            RendezvousBackend.Etcd => EtcdRendezvous(workerNonce),
            _ => throw new ArgumentException($"Unknown rendezvous backend {_options.Backend}."),
        };
    }

    /// <summary>
    /// etcd rendezvous: each worker registers its nonce as a key under
    /// a shared prefix; once <see cref="ElasticLauncherOptions.WorldSize"/>
    /// keys land the deterministic sort decides ranks. We talk to etcd
    /// over its v3 HTTP/JSON gateway, so the implementation works
    /// against any standard etcd cluster without a client-library
    /// dependency.
    ///
    /// <para>Endpoint format: <c>etcd://host:port/prefix</c>. Workers
    /// PUT their nonce under <c>{prefix}/{nonce}</c>, then poll a
    /// RANGE over <c>{prefix}/</c> until the worker count is met.</para>
    /// </summary>
    private RendezvousAssignment EtcdRendezvous(string workerNonce)
    {
#if NET5_0_OR_GREATER
        var (baseUrl, prefix) = ParseEtcdEndpoint(_options.Endpoint);
        EtcdPut(baseUrl, $"{prefix}/{workerNonce}",
            DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString());

        var deadline = DateTime.UtcNow.AddSeconds(_options.RendezvousTimeoutSeconds);
        while (DateTime.UtcNow < deadline)
        {
            var keys = EtcdRangeKeys(baseUrl, prefix + "/");
            if (keys.Count >= _options.WorldSize)
            {
                keys.Sort(StringComparer.Ordinal);
                int rank = keys.IndexOf($"{prefix}/{workerNonce}");
                if (rank < 0)
                    throw new InvalidOperationException(
                        $"Our nonce {workerNonce} disappeared from etcd prefix {prefix}.");
                return new RendezvousAssignment
                {
                    Rank = rank,
                    WorldSize = _options.WorldSize,
                    Endpoint = _options.Endpoint,
                };
            }
            Thread.Sleep(100);
        }
        throw new TimeoutException(
            $"etcd rendezvous timed out after {_options.RendezvousTimeoutSeconds}s.");
#else
        _ = workerNonce;
        throw new NotSupportedException(
            "etcd rendezvous requires .NET 5+ (HttpClient). Use the File or TCP backend on net471.");
#endif
    }

#if NET5_0_OR_GREATER
    private static (string BaseUrl, string Prefix) ParseEtcdEndpoint(string endpoint)
    {
        if (endpoint.StartsWith("etcd://", StringComparison.OrdinalIgnoreCase))
            endpoint = endpoint.Substring("etcd://".Length);
        int slash = endpoint.IndexOf('/');
        if (slash <= 0)
            throw new ArgumentException($"etcd endpoint must be host:port/prefix; got '{endpoint}'.");
        string hostPort = endpoint.Substring(0, slash);
        string prefix = endpoint.Substring(slash);
        return ($"http://{hostPort}", prefix.TrimEnd('/'));
    }

    private static void EtcdPut(string baseUrl, string key, string value)
    {
        string body = "{\"key\":\"" + Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(key)) +
                      "\",\"value\":\"" + Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(value)) + "\"}";
        EtcdHttpPost($"{baseUrl}/v3/kv/put", body);
    }

    private static List<string> EtcdRangeKeys(string baseUrl, string prefix)
    {
        var prefixBytes = System.Text.Encoding.UTF8.GetBytes(prefix);
        var end = (byte[])prefixBytes.Clone();
        if (end.Length > 0) end[end.Length - 1] = (byte)(end[end.Length - 1] + 1);

        string body = "{\"key\":\"" + Convert.ToBase64String(prefixBytes) +
                      "\",\"range_end\":\"" + Convert.ToBase64String(end) +
                      "\",\"keys_only\":true}";
        string response = EtcdHttpPost($"{baseUrl}/v3/kv/range", body);
        return ParseEtcdRangeResponse(response);
    }

    // HttpClient is shared across calls — etcd rendezvous traffic is light.
    private static readonly System.Net.Http.HttpClient _etcdHttp = new() { Timeout = TimeSpan.FromSeconds(10) };

    private static string EtcdHttpPost(string url, string body)
    {
        using var content = new System.Net.Http.StringContent(body, System.Text.Encoding.UTF8, "application/json");
        using var resp = _etcdHttp.PostAsync(url, content).GetAwaiter().GetResult();
        if (!resp.IsSuccessStatusCode)
            throw new InvalidOperationException($"etcd HTTP {url} returned {resp.StatusCode}.");
        return resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
    }

    private static List<string> ParseEtcdRangeResponse(string json)
    {
        // Minimal extractor for the "kvs" array's "key" entries.
        var keys = new List<string>();
        int i = 0;
        while (true)
        {
            int idx = json.IndexOf("\"key\":\"", i, StringComparison.Ordinal);
            if (idx < 0) break;
            idx += "\"key\":\"".Length;
            int end = json.IndexOf('"', idx);
            if (end < 0) break;
            string b64 = json.Substring(idx, end - idx);
            try { keys.Add(System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(b64))); }
            catch { /* skip malformed */ }
            i = end + 1;
        }
        return keys;
    }
#endif

    /// <summary>
    /// TCP rendezvous: parses <see cref="ElasticLauncherOptions.Endpoint"/>
    /// as <c>host:port</c>. Each worker first attempts to bind a listener
    /// on that endpoint — the bind winner becomes the coordinator and
    /// the rest connect as clients. The coordinator collects every nonce,
    /// sorts deterministically, and replies to each client with its rank.
    /// Bit-equivalent rank assignment to file-rendezvous: the worker with
    /// the lexicographically-smallest nonce gets rank 0.
    /// </summary>
    private RendezvousAssignment TcpRendezvous(string workerNonce)
    {
        var (host, port) = ParseHostPort(_options.Endpoint);

        // Try to bind first. Whoever wins the bind acts as coordinator.
        var listener = TryBindListener(host, port);
        if (listener is not null)
        {
            try { return TcpCoordinator(listener, workerNonce, host, port); }
            finally { try { listener.Stop(); } catch { /* listener already closed */ } }
        }
        // Bind failed — someone else is the coordinator. Connect.
        return TcpClient(host, port, workerNonce);
    }

    private static (string Host, int Port) ParseHostPort(string endpoint)
    {
        // Accept "host:port" or "tcp://host:port".
        if (endpoint.StartsWith("tcp://", StringComparison.OrdinalIgnoreCase))
            endpoint = endpoint.Substring("tcp://".Length);
        int colon = endpoint.LastIndexOf(':');
        if (colon <= 0)
            throw new ArgumentException($"TCP rendezvous endpoint must be host:port; got '{endpoint}'.");
        string host = endpoint.Substring(0, colon);
        if (!int.TryParse(endpoint.Substring(colon + 1), out int port) || port <= 0 || port > 65535)
            throw new ArgumentException($"TCP rendezvous port must be 1..65535; got '{endpoint.Substring(colon + 1)}'.");
        return (host, port);
    }

    private static TcpListener? TryBindListener(string host, int port)
    {
        try
        {
            var address = ResolveBindAddress(host);
            var l = new TcpListener(address, port);
            l.ExclusiveAddressUse = true;
            l.Start();
            return l;
        }
        catch (SocketException)
        {
            return null;
        }
    }

    private static IPAddress ResolveBindAddress(string host)
    {
        // "0.0.0.0" / "*" → any; otherwise parse-or-resolve.
        if (host == "*" || host == "0.0.0.0") return IPAddress.Any;
        if (IPAddress.TryParse(host, out var ip)) return ip;
        var addrs = Dns.GetHostAddresses(host);
        if (addrs.Length == 0)
            throw new ArgumentException($"Could not resolve TCP rendezvous host '{host}'.");
        return addrs[0];
    }

    private RendezvousAssignment TcpCoordinator(TcpListener listener, string ownNonce, string host, int port)
    {
        var nonces = new List<string> { ownNonce };
        var clientReplies = new List<TcpReply>(_options.WorldSize - 1);
        var deadline = DateTime.UtcNow.AddSeconds(_options.RendezvousTimeoutSeconds);

        while (nonces.Count < _options.WorldSize)
        {
            if (DateTime.UtcNow >= deadline)
                throw new TimeoutException(
                    $"TCP rendezvous timed out after {_options.RendezvousTimeoutSeconds}s. " +
                    $"Saw {nonces.Count} of {_options.WorldSize} workers.");
            // Wait up to 250ms for the next connection so we can re-check the deadline.
            if (!WaitForConnection(listener, TimeSpan.FromMilliseconds(250))) continue;
            var client = listener.AcceptTcpClient();
            client.NoDelay = true;
            var stream = client.GetStream();
            string nonce = ReadLine(stream);
            nonces.Add(nonce);
            // Hold this client open until we've collected all nonces, then
            // reply. The stream + client are explicitly disposed in the
            // reply loop below — DO NOT use a `using` here.
            clientReplies.Add(new TcpReply(client, stream, nonce));
        }

        // Sort + assign ranks.
        var ordered = new List<string>(nonces);
        ordered.Sort(StringComparer.Ordinal);
        int myRank = ordered.IndexOf(ownNonce);

        // Write each client its rank.
        foreach (var reply in clientReplies)
        {
            int rank = ordered.IndexOf(reply.Nonce);
            try
            {
                WriteLine(reply.Stream, $"OK {rank} {_options.WorldSize}");
            }
            finally
            {
                reply.Stream.Dispose();
                reply.Client.Dispose();
            }
        }

        return new RendezvousAssignment
        {
            Rank = myRank,
            WorldSize = _options.WorldSize,
            Endpoint = $"{host}:{port}",
        };
    }

    private RendezvousAssignment TcpClient(string host, int port, string workerNonce)
    {
        var deadline = DateTime.UtcNow.AddSeconds(_options.RendezvousTimeoutSeconds);
        Exception? lastError = null;
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                using var client = new TcpClient();
                var connectTask = client.ConnectAsync(host, port);
                if (!connectTask.Wait((int)Math.Min(2000, (deadline - DateTime.UtcNow).TotalMilliseconds)))
                {
                    Thread.Sleep(50);
                    continue;
                }
                client.NoDelay = true;
                using var stream = client.GetStream();
                stream.ReadTimeout = (int)Math.Max(1000, (deadline - DateTime.UtcNow).TotalMilliseconds);
                WriteLine(stream, workerNonce);
                string reply = ReadLine(stream);
                // Format: "OK <rank> <worldSize>"
                var parts = reply.Split(' ');
                if (parts.Length != 3 || parts[0] != "OK")
                    throw new InvalidOperationException($"Unexpected coordinator reply: '{reply}'.");
                int rank = int.Parse(parts[1]);
                int ws = int.Parse(parts[2]);
                if (ws != _options.WorldSize)
                    throw new InvalidOperationException(
                        $"Coordinator advertised WorldSize {ws}, ours is {_options.WorldSize}.");
                return new RendezvousAssignment
                {
                    Rank = rank,
                    WorldSize = ws,
                    Endpoint = $"{host}:{port}",
                };
            }
            catch (SocketException ex) { lastError = ex; Thread.Sleep(50); }
            catch (IOException ex) { lastError = ex; Thread.Sleep(50); }
        }
        throw new TimeoutException(
            $"TCP rendezvous client could not reach coordinator at {host}:{port} within " +
            $"{_options.RendezvousTimeoutSeconds}s." +
            (lastError is null ? "" : $" Last error: {lastError.Message}"));
    }

    private static bool WaitForConnection(TcpListener listener, TimeSpan timeout)
    {
        // Pending() flips true when the OS has accepted a connection on our socket.
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (listener.Pending()) return true;
            Thread.Sleep(10);
        }
        return false;
    }

    private static void WriteLine(NetworkStream stream, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value + "\n");
        stream.Write(bytes, 0, bytes.Length);
        stream.Flush();
    }

    private static string ReadLine(NetworkStream stream)
    {
        var buf = new byte[1];
        var sb = new StringBuilder();
        while (true)
        {
            int n = stream.Read(buf, 0, 1);
            if (n <= 0) throw new IOException("Connection closed before line terminator.");
            if (buf[0] == (byte)'\n') return sb.ToString();
            sb.Append((char)buf[0]);
        }
    }

    private readonly struct TcpReply
    {
        public TcpReply(TcpClient client, NetworkStream stream, string nonce)
        {
            Client = client; Stream = stream; Nonce = nonce;
        }
        public TcpClient Client { get; }
        public NetworkStream Stream { get; }
        public string Nonce { get; }
    }

    /// <summary>
    /// Treat <paramref name="nonce"/> as untrusted: reject path separators,
    /// '..' segments, rooted paths, and any character that's invalid in a
    /// filename on the host OS. Without this guard a caller could escape
    /// the rendezvous directory or produce silently-failed writes.
    /// </summary>
    private static void ValidateNonceAsFilename(string nonce)
    {
        if (string.IsNullOrEmpty(nonce))
            throw new ArgumentException("Worker nonce must be non-empty.", nameof(nonce));
        if (nonce.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0
            || nonce.IndexOf('/') >= 0
            || nonce.IndexOf('\\') >= 0
            || nonce == "." || nonce == ".."
            || Path.IsPathRooted(nonce))
        {
            throw new ArgumentException(
                $"Worker nonce '{nonce}' contains path separators / invalid filename characters / a rooted path.",
                nameof(nonce));
        }
    }

    private RendezvousAssignment FileRendezvous(string workerNonce)
    {
        ValidateNonceAsFilename(workerNonce);
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
        return SpawnInProcessWorkersWithGenerationRestart(worker);
    }

    private async Task SpawnInProcessWorkersWithGenerationRestart(Action<RendezvousAssignment> worker)
    {
        // Per-task restart tore down only the failing rank, leaving
        // the healthy ranks at a stale rendezvous generation — the
        // restarted rank would rejoin and deadlock on the next
        // collective. Real elastic semantics restart the whole
        // generation: when ANY rank fails, we tear down the
        // rendezvous directory, the surviving ranks unwind out of
        // their delegate, and the next generation re-rendezvouses
        // from scratch.
        int generation = 0;
        while (true)
        {
            // Clean up any stale state at the start of each generation.
            // Cooperative cancellation across the worker delegates would
            // need an API change (the worker takes RendezvousAssignment,
            // no token); when the first task throws, Task.WhenAll
            // surfaces the exception and the catch below restarts the
            // whole generation. Surviving ranks complete naturally on
            // their next collective failure or natural end of work.
            CleanupRendezvous();

            var tasks = new Task[_options.WorldSize];
            for (int i = 0; i < _options.WorldSize; i++)
            {
                int workerIndex = i;
                int gen = generation;
                tasks[i] = Task.Run(() =>
                {
                    var assignment = Rendezvous($"worker-{workerIndex:D6}-gen{gen:D3}");
                    worker(assignment);
                });
            }

            try
            {
                await Task.WhenAll(tasks).ConfigureAwait(false);
                return; // every worker finished cleanly
            }
            catch
            {
                if (!_options.RestartOnFailure || generation >= _options.MaxRestarts)
                    throw;
                generation++;
                await Task.Delay(_options.HealthCheckIntervalSeconds * 1000).ConfigureAwait(false);
                // Loop around — next iteration cleans up + re-rendezvouses
                // the entire group at a fresh generation.
            }
        }
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
