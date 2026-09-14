using AiDotNet.Tensors.Helpers.Autotune;
using System.Runtime.InteropServices;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers.Autotune;

public sealed partial class EvolutionKernelAutotunerTests
{
    public enum ReceiptFailure { None, MoveDenied, DirectoryDenied, AncestorDenied, MissingNativeEntry }

    [Theory]
    [InlineData(Architecture.X86, 0x90000)]
    [InlineData(Architecture.X64, 0x90000)]
    [InlineData(Architecture.Arm64, 0x90000)]
    [InlineData(Architecture.Arm, 0x84000)]
    public void Quarantine_DirectoryFlagsMatchTheLinuxArchitecture(Architecture architecture, int expected)
    {
        Assert.True(QuarantineReceiptCommit.TryGetDirectoryFlags(architecture, out LinuxDirectoryOpenFlags flags));
        Assert.Equal(expected, (int)flags);
    }

    [Fact]
    public void Quarantine_UnknownArchitectureCannotGuessDirectoryFlags()
        => Assert.False(QuarantineReceiptCommit.TryGetDirectoryFlags((Architecture)int.MaxValue, out _));

    [Theory]
    [InlineData(ReceiptFailure.MoveDenied)]
    [InlineData(ReceiptFailure.DirectoryDenied)]
    [InlineData(ReceiptFailure.AncestorDenied)]
    [InlineData(ReceiptFailure.MissingNativeEntry)]
    public async Task Quarantine_FailedCommitBarrierCannotClaimDurability(ReceiptFailure failure)
    {
        var operations = new RecordedCommitOperations(QuarantineCommitMode.LinuxDirectorySync, failure);
        var inner = new MemoryStore();
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), inner, operations);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());

        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());

        Assert.True(result.WasApplied);
        Assert.False(result.WasPersisted);
        Assert.Null(result.ReceiptPath);
        Assert.Null(tuner.Deployment.Current);
        Assert.False(tuner.TryHydrate());
        Assert.True(operations.MoveAttempted);
        Assert.Contains("tensor-kernel-quarantine-v1", operations.ReceiptBeforeMove, StringComparison.Ordinal);
        if (failure == ReceiptFailure.MoveDenied)
        {
            Assert.Empty(operations.DirectoryBarriers);
            Assert.Empty(Directory.GetFiles(Journal(), "*.quarantine.json"));
            Assert.Single(Directory.GetFiles(Journal(), "*.pending"));
        }
        else
        {
            // A post-rename barrier failure must not remove the visible tombstone.
            string receipt = Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json"));
            Assert.Empty(Directory.GetFiles(Journal(), "*.pending"));
            Assert.Equal(failure == ReceiptFailure.AncestorDenied ? 2 : 1, operations.DirectoryBarriers.Count);
            Directory.CreateDirectory(Journal("other-process"));
            File.Copy(receipt, Path.Combine(Journal("other-process"), Path.GetFileName(receipt)));
            var fresh = CreateTuner(new(), new QuarantinedKernelTuningStore<FakeKernelConfiguration>(
                Journal("other-process"), inner), MeasurePassed);
            Assert.False(fresh.TryHydrate());
        }
    }

    [Fact]
    public async Task Quarantine_UnsupportedDurabilityRetainsOnlyBestEffortReceipt()
    {
        var operations = new RecordedCommitOperations(QuarantineCommitMode.Unsupported, ReceiptFailure.None);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore(), operations);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());

        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());

        Assert.True(result.WasApplied);
        Assert.False(result.WasPersisted);
        Assert.Null(result.ReceiptPath);
        Assert.True(operations.MoveAttempted);
        Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json"));
        Assert.Empty(Directory.GetFiles(Journal(), "*.pending"));
        Assert.Empty(operations.DirectoryBarriers);
        Assert.False(tuner.TryHydrate());
    }

    [Fact]
    public async Task Quarantine_RequiresEveryDirectoryBarrierAfterPublication()
    {
        var operations = new RecordedCommitOperations(QuarantineCommitMode.LinuxDirectorySync, ReceiptFailure.None);
        var store = new QuarantinedKernelTuningStore<FakeKernelConfiguration>(Journal(), new MemoryStore(), operations);
        var tuner = CreateTuner(new(), store, MeasurePassed);
        var run = await tuner.TuneAsync(Seeds());

        var result = await tuner.QuarantineAsync(run.ActiveDeployment, Regression());

        Assert.True(result.WasApplied);
        Assert.True(result.WasPersisted);
        Assert.Equal(Assert.Single(Directory.GetFiles(Journal(), "*.quarantine.json")), result.ReceiptPath);
        var expectedDirectories = new List<string>();
        for (string? directory = Journal(); !string.IsNullOrEmpty(directory); directory = Path.GetDirectoryName(directory))
            expectedDirectories.Add(directory);
        Assert.Equal(expectedDirectories, operations.DirectoryBarriers);
        Assert.Empty(Directory.GetFiles(Journal(), "*.pending"));
    }

    private sealed class RecordedCommitOperations : IQuarantineCommitOperations
    {
        private readonly ReceiptFailure _failure;
        private string? _destination;

        internal RecordedCommitOperations(QuarantineCommitMode mode, ReceiptFailure failure)
        {
            Mode = mode;
            _failure = failure;
        }

        public QuarantineCommitMode Mode { get; }
        internal bool MoveAttempted { get; private set; }
        internal string ReceiptBeforeMove { get; private set; } = string.Empty;
        internal List<string> DirectoryBarriers { get; } = new();

        public bool MoveNewFile(string source, string destination)
        {
            MoveAttempted = true;
            // The writer must have closed its exclusive file before attempting publication.
            ReceiptBeforeMove = File.ReadAllText(source);
            if (_failure == ReceiptFailure.MoveDenied) return false;
            File.Move(source, destination);
            _destination = destination;
            return true;
        }

        public bool FlushDirectory(string directory)
        {
            Assert.NotNull(_destination);
            Assert.True(File.Exists(_destination), "Directory barriers must follow publication.");
            DirectoryBarriers.Add(directory);
            return _failure switch
            {
                ReceiptFailure.DirectoryDenied => false,
                ReceiptFailure.AncestorDenied => DirectoryBarriers.Count < 2,
                ReceiptFailure.MissingNativeEntry => throw new EntryPointNotFoundException("The directory barrier is unavailable."),
                _ => true
            };
        }
    }
}
