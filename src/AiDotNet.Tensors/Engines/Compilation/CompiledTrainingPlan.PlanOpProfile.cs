using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private PlanOperationProfile? _planOperationProfile;

    private void InstallPlanOperationProfile(string[] backwardNames)
    {
        string? directory = Environment.GetEnvironmentVariable("AIDOTNET_PLAN_OP_PROFILE_DIR");
        if (string.IsNullOrEmpty(directory) || _engine.Unwrap().GetType() != typeof(CpuEngine)
            || !ReferenceEquals(_engine, _engine.Unwrap())) return;
        var profiler = new PlanOperationProfile(directory, _planId, _parameters.Length);
        _planOperationProfile = profiler;
        RebindForwardOperationProfile();
        profiler.WrapBackward(_backwardActions, backwardNames);
    }

    private void RebindForwardOperationProfile()
    {
        if (_planOperationProfile is not { } profiler) return;
        int[] map = ActionToStepIndex ?? throw new InvalidOperationException("Operation profile requires forward metadata.");
        if (_forwardSteps is null || map.Length != _forwardActions.Length)
            throw new InvalidOperationException("Operation profile forward metadata differs.");
        profiler.WrapForward(_forwardActions, map.Select(index => _forwardSteps[index].OpName).ToArray());
    }

    private sealed class PlanOperationProfile
    {
        private sealed class Counter
        {
            internal readonly string Name;
            internal int Calls;
            internal long Ticks;
            internal long AllocationEstimate;
            internal Counter(string name) => Name = name;
        }
        private readonly string _directory;
        private readonly long _id;
        private readonly int _parameters, _pid;
        private Counter[] _forward = Array.Empty<Counter>(), _backward = Array.Empty<Counter>();
        private long _wallTicks;
        private int _steps, _forwardGeneration;

        internal PlanOperationProfile(string directory, long id, int parameters)
        {
            _directory = Path.GetFullPath(directory); _id = id; _parameters = parameters;
            using var process = Process.GetCurrentProcess(); _pid = process.Id;
            Directory.CreateDirectory(_directory);
        }

        private static long Allocated()
        {
#if NETCOREAPP3_0_OR_GREATER
            return GC.GetTotalAllocatedBytes(false);
#else
            return 0;
#endif
        }

        private static Action<IEngine> Wrap(Action<IEngine> action, Counter counter) => engine =>
        {
            long allocated = Allocated(), start = Stopwatch.GetTimestamp();
            action(engine);
            counter.Ticks += Stopwatch.GetTimestamp() - start;
            counter.AllocationEstimate += Math.Max(0, Allocated() - allocated);
            counter.Calls++;
        };

        internal void WrapForward(Action<IEngine>[] actions, string[] names)
        {
            if (actions.Length != names.Length) throw new InvalidOperationException("Forward profile names differ.");
            if (_forward.Length != names.Length || !_forward.Select(row => row.Name).SequenceEqual(names))
            {
                _forward = names.Select(name => new Counter(name)).ToArray();
                _forwardGeneration++;
            }
            for (int i = 0; i < actions.Length; i++) actions[i] = Wrap(actions[i], _forward[i]);
        }

        internal void WrapBackward(Action<IEngine>[] actions, string[] names)
        {
            if (actions.Length != names.Length) throw new InvalidOperationException("Backward profile names differ.");
            _backward = names.Select(name => new Counter(name)).ToArray();
            for (int i = 0; i < actions.Length; i++) actions[i] = Wrap(actions[i], _backward[i]);
        }

        internal void CompleteStep(long start)
        {
            _wallTicks += Stopwatch.GetTimestamp() - start; _steps++;
            var output = new StringBuilder();
            output.AppendLine("# schema=cpu-plan-operation-profile-v1 diagnostic_only=true formal_performance_win=false");
            output.AppendLine($"# pid={_pid} plan={_id} parameters={_parameters} steps={_steps} forward_generation={_forwardGeneration} wall_ticks={_wallTicks} stopwatch_frequency={Stopwatch.Frequency}");
            output.AppendLine("# Allocation estimates are process-wide unflushed GC counters, not complete or exact native allocation bounds. Sequential diagnostic workload only.");
            output.AppendLine("phase\taction\top\tcalls\ttotal_ticks\ttotal_us\tmanaged_allocation_estimate_bytes");
            void Rows(string phase, Counter[] rows)
            {
                for (int i = 0; i < rows.Length; i++)
                {
                    var row = rows[i];
                    string name = row.Name.Replace('\t', ' ').Replace('\r', ' ').Replace('\n', ' ');
                    string us = (row.Ticks * (1_000_000.0 / Stopwatch.Frequency)).ToString("F3", CultureInfo.InvariantCulture);
                    output.AppendLine($"{phase}\t{i}\t{name}\t{row.Calls}\t{row.Ticks}\t{us}\t{row.AllocationEstimate}");
                }
            }
            Rows("forward", _forward); Rows("backward", _backward);
            string path = Path.Combine(_directory, $"pid{_pid}-plan{_id}-step{_steps:D2}.tsv");
            if (File.Exists(path)) throw new IOException("Preserve previous operation profile: " + path);
            File.WriteAllText(path, output.ToString(), new UTF8Encoding(false));
        }
    }
}
