using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Guards the property that the normalization kernels take their scalar
/// hyperparameters at launch instead of baking them into the instruction stream.
///
/// This needs its own check because neither of the two things that already run
/// can see a regression here. ptxas assembles a module whose epsilon register is
/// declared but never loaded exactly as happily as one that loads it - an
/// uninitialised register is legal PTX - so the offline cubin gate goes green on
/// a kernel that reads garbage. And every execution test in this area is gated on
/// an admitted NVIDIA device, so on a machine without one they all skip.
///
/// Both checks below run on the emitted text, on any machine, with no device.
/// </summary>
public class DirectPtxNormalizationScalarParameterTests
{
    /// <summary>
    /// Deliberately not a value any caller or default would pick, so that finding
    /// its bit pattern in the PTX can only mean it was baked in.
    /// </summary>
    private const float ProbeEpsilon = 3.7218e-3f;

    private const float ProbeMomentum = 0.31719f;

    /// <summary>
    /// The named registers the de-baked scalars live in. Named rather than
    /// numbered because the operation emitters declare f32 banks ranging from ten
    /// to forty-four registers, so no fixed index fits all of them.
    /// </summary>
    private static readonly string[] ScalarRegisters = { "%eps", "%mom", "%ommom" };

    private static string Bits(float value) =>
        "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0)
            .ToString("X8", CultureInfo.InvariantCulture);

    /// <summary>Every module the de-baked kernels can emit, with the probe values.</summary>
    private static IEnumerable<(string Name, string Ptx)> EmitAll()
    {
        foreach (DirectPtxRowNormalizationOperation op in
                 (DirectPtxRowNormalizationOperation[])Enum.GetValues(
                     typeof(DirectPtxRowNormalizationOperation)))
        foreach (int rows in new[] { 256, 2_048, 8_192 })
        {
            if (!PtxRowNormalizationD64Kernel.IsSupportedRows(rows)) continue;
            string ptx;
            try { ptx = PtxRowNormalizationD64Kernel.EmitPtx(8, 6, op, rows, ProbeEpsilon); }
            catch (ArgumentException) { continue; }
            yield return ($"row-{op}-r{rows}", ptx);
        }

        foreach (DirectPtxChannelNormalizationOperation op in
                 (DirectPtxChannelNormalizationOperation[])Enum.GetValues(
                     typeof(DirectPtxChannelNormalizationOperation)))
            yield return ($"channel-{op}",
                PtxChannelNormalizationD64Kernel.EmitPtx(8, 6, op, ProbeEpsilon, ProbeMomentum));

        foreach (int rows in new[] { 256, 2_048, 8_192 })
        {
            if (PtxFusedResidualBiasLayerNormGeluD64Kernel.IsSupportedRows(rows))
                yield return ($"gelu-r{rows}",
                    PtxFusedResidualBiasLayerNormGeluD64Kernel.EmitPtx(8, 6, rows, ProbeEpsilon));
            foreach (int warps in new[] { 1, 2, 4, 8 })
                yield return ($"rmsnorm-r{rows}-w{warps}",
                    PtxFusedResidualRmsNormD64Kernel.EmitPtx(8, 6, ProbeEpsilon, rows, warps));
        }

        foreach (bool causal in new[] { false, true })
        foreach (bool fuse in new[] { false, true })
            yield return ($"attention-c{causal}-f{fuse}",
                PtxOnlineFusedAttention128x64Kernel.EmitPtx(
                    8, 6, causal, fuse, scale: 0.125f, epsilon: ProbeEpsilon));
    }

    [Fact]
    public void EmittedPtx_NeverBakesTheScalarHyperparameters()
    {
        string epsilonBits = Bits(ProbeEpsilon);
        string momentumBits = Bits(ProbeMomentum);
        string complementBits = Bits(1f - ProbeMomentum);

        var offenders = new List<string>();
        int modules = 0;
        foreach ((string name, string ptx) in EmitAll())
        {
            modules++;
            if (ptx.IndexOf(epsilonBits, StringComparison.OrdinalIgnoreCase) >= 0)
                offenders.Add($"{name}: epsilon {epsilonBits}");
            if (ptx.IndexOf(momentumBits, StringComparison.OrdinalIgnoreCase) >= 0)
                offenders.Add($"{name}: momentum {momentumBits}");

            // The complement is derived on the device from the momentum register.
            // Seeing its literal would mean the host computed it and baked it,
            // which reintroduces exactly the unbounded module key the parameters
            // were moved to eliminate.
            if (ptx.IndexOf(complementBits, StringComparison.OrdinalIgnoreCase) >= 0)
                offenders.Add($"{name}: one-minus-momentum {complementBits}");
        }

        Assert.True(modules > 80, $"Only {modules} modules were emitted; the sweep is incomplete.");
        Assert.True(offenders.Count == 0,
            "These modules bake a scalar that must be a launch parameter:" +
            Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void EveryScalarParameterIsLoadedBeforeItIsUsed()
    {
        var offenders = new List<string>();
        foreach ((string name, string ptx) in EmitAll())
        {
            string[] lines = ptx.Split('\n');

            // Where each named scalar register is first written. Any instruction
            // counts, not just a load: epsilon and momentum arrive via ld.param,
            // but one-minus-momentum is derived on the device with sub.rn.f32, and
            // both are equally valid ways to fill a register.
            var loadedAt = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (line.StartsWith(".reg ", StringComparison.Ordinal)) continue;
                int space = line.IndexOf(' ');
                if (space < 0) continue;
                string destination = line.Substring(space + 1).TrimStart();
                int comma = destination.IndexOf(',');
                if (comma < 0) continue;
                destination = destination.Substring(0, comma).Trim();
                if (destination.Length > 0 && destination[0] == '%' && !loadedAt.ContainsKey(destination))
                    loadedAt[destination] = i;
            }

            // The failure this test exists for: a named scalar register reached by
            // an instruction that no ld.param filled, or filled only later. ptxas
            // accepts both - an uninitialised register is legal PTX - so the
            // kernel would run and read whatever was in the register file.
            //
            // A declaration line is not a use, and neither is the load itself.
            // An operation with no epsilon term declares the parameter anyway,
            // because the parameter position has to be identical across all of
            // them for the positional launch argument array to work; that is an
            // ignored parameter, not a dropped load, and is only a problem if
            // some instruction then reads the register.
            foreach (string register in ScalarRegisters)
            {
                loadedAt.TryGetValue(register, out int loadLine);
                bool isLoaded = loadedAt.ContainsKey(register);

                for (int i = 0; i < lines.Length; i++)
                {
                    string line = lines[i].Trim();
                    if (line.StartsWith(".reg ", StringComparison.Ordinal)) continue;
                    if (!line.Contains(register)) continue;

                    if (!isLoaded)
                        offenders.Add(
                            $"{name}: line {i} reads '{register}', which nothing ever fills");
                    else if (i < loadLine)
                        offenders.Add(
                            $"{name}: line {i} reads '{register}' before it is filled on line {loadLine}");
                    else
                        continue;
                    break;
                }
            }
        }

        Assert.True(offenders.Count == 0,
            "Scalar parameter wiring is broken in these modules:" +
            Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }
}
