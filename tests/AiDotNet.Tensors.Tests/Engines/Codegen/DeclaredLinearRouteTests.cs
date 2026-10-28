// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The declared linear-layer route (#890 phase 4c): CudaBackend.FusedLinear{ReLU,Sigmoid,Tanh,GELU,Swish} and their
// backwards are served by declared programs in the hand-written kernels' contract. Both routes -- declared, and the
// hand-written kernels with the switch off -- are checked against an fp64 reference written out here.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredLinearRouteTests
{
    private static CudaBackend OpenBackendOrSkip()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            throw new InvalidOperationException("CUDA driver was detected, but the CUDA backend failed to initialize.");
        }
        return backend;
    }

    private static float[] Fill(int count, int salt)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((((i * 37 + salt * 101) % 97) - 48) / 96.0);
        return values;
    }

    private const double GeluInner = 0.7978845608, GeluCubic = 0.044715;

    private static double Activate(string activation, double x) => activation switch
    {
        "relu" => Math.Max(0.0, x),
        "sigmoid" => 1.0 / (1.0 + Math.Exp(-x)),
        "tanh" => Math.Tanh(x),
        "gelu" => 0.5 * x * (1.0 + Math.Tanh(GeluInner * (x + GeluCubic * x * x * x))),
        _ => x / (1.0 + Math.Exp(-x)),   // swish
    };

    // The backward of the activation from the gradient and what the backend saves: the output for sigmoid and tanh,
    // the pre-activation for the others.
    private static double Derivative(string activation, double dy, double saved)
    {
        switch (activation)
        {
            case "relu": return saved > 0 ? dy : 0.0;
            case "sigmoid": return dy * saved * (1.0 - saved);
            case "tanh": return dy * (1.0 - saved * saved);
            case "gelu":
            {
                double t = Math.Tanh(GeluInner * (saved + GeluCubic * saved * saved * saved));
                double dt = GeluInner * (1.0 + 3.0 * GeluCubic * saved * saved) * (1.0 - t * t);
                return dy * (0.5 * (1.0 + t) + 0.5 * saved * dt);
            }
            default:
            {
                double sig = 1.0 / (1.0 + Math.Exp(-saved));
                return dy * (sig + saved * sig * (1.0 - sig));
            }
        }
    }

    private static void Forward(CudaBackend backend, string activation, IGpuBuffer x, IGpuBuffer w, IGpuBuffer b,
        IGpuBuffer y, int batch, int inFeatures, int outFeatures)
    {
        switch (activation)
        {
            case "relu": backend.FusedLinearReLU(x, w, b, y, batch, inFeatures, outFeatures); break;
            case "sigmoid": backend.FusedLinearSigmoid(x, w, b, y, batch, inFeatures, outFeatures); break;
            case "tanh": backend.FusedLinearTanh(x, w, b, y, batch, inFeatures, outFeatures); break;
            case "gelu": backend.FusedLinearGELU(x, w, b, y, batch, inFeatures, outFeatures); break;
            default: backend.FusedLinearSwish(x, w, b, y, batch, inFeatures, outFeatures); break;
        }
    }

    private static void Backward(CudaBackend backend, string activation, IGpuBuffer dy, IGpuBuffer x, IGpuBuffer w,
        IGpuBuffer saved, IGpuBuffer dx, IGpuBuffer dw, IGpuBuffer db, int batch, int inFeatures, int outFeatures)
    {
        switch (activation)
        {
            case "relu": backend.FusedLinearReLUBackward(dy, x, w, saved, dx, dw, db, batch, inFeatures, outFeatures); break;
            case "sigmoid": backend.FusedLinearSigmoidBackward(dy, x, w, saved, dx, dw, db, batch, inFeatures, outFeatures); break;
            case "tanh": backend.FusedLinearTanhBackward(dy, x, w, saved, dx, dw, db, batch, inFeatures, outFeatures); break;
            case "gelu": backend.FusedLinearGELUBackward(dy, x, w, saved, dx, dw, db, batch, inFeatures, outFeatures); break;
            default: backend.FusedLinearSwishBackward(dy, x, w, saved, dx, dw, db, batch, inFeatures, outFeatures); break;
        }
    }

    // The largest error relative to max(1, |reference|).
    private static double MaxError(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), "NaN at " + i);
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i])));
        }
        return worst;
    }

    /// <summary>
    /// Forward and backward, each activation, at an even, a ragged and a larger shape: the declared program serves the
    /// call, and on every output its error against the fp64 reference is within 2e-5 or within twice the hand-written
    /// kernels' own error on the same data (both compile their transcendentals with fast math, whose error a long
    /// reduction accumulates).
    /// </summary>
    [SkippableTheory]
    [InlineData("relu", 64, 128, 96)]
    [InlineData("sigmoid", 64, 128, 96)]
    [InlineData("tanh", 5, 33, 17)]
    [InlineData("gelu", 5, 33, 17)]
    [InlineData("swish", 64, 128, 96)]
    [InlineData("relu", 256, 512, 256)]
    [InlineData("gelu", 256, 512, 256)]
    public void TheDeclaredLinearMatchesTheReference(string activation, int batch, int inFeatures, int outFeatures)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled,
            "The declared route is not released on this architecture (see AIDOTNET_DECLARED_KERNELS_ARCH).");
        float[] x = Fill(batch * inFeatures, 1), w = Fill(inFeatures * outFeatures, 2), b = Fill(outFeatures, 3);
        float[] dy = Fill(batch * outFeatures, 4);
        var pre = new double[batch * outFeatures];
        var y = new double[pre.Length];
        for (int r = 0; r < batch; r++)
            for (int j = 0; j < outFeatures; j++)
            {
                double sum = b[j];
                for (int k = 0; k < inFeatures; k++) sum += (double)x[r * inFeatures + k] * w[k * outFeatures + j];
                pre[r * outFeatures + j] = sum;
                y[r * outFeatures + j] = Activate(activation, sum);
            }
        bool savesOutput = activation is "sigmoid" or "tanh";
        // The saved tensor goes in as fp32, so the reference derivative starts from the same rounded values.
        var saved = new float[pre.Length];
        for (int i = 0; i < saved.Length; i++) saved[i] = (float)(savesOutput ? y[i] : pre[i]);
        var dpre = new double[pre.Length];
        for (int i = 0; i < dpre.Length; i++) dpre[i] = Derivative(activation, dy[i], saved[i]);
        var dx = new double[x.Length];
        var dw = new double[w.Length];
        var db = new double[b.Length];
        for (int r = 0; r < batch; r++)
            for (int j = 0; j < outFeatures; j++)
            {
                double g = dpre[r * outFeatures + j];
                db[j] += g;
                for (int k = 0; k < inFeatures; k++)
                {
                    dx[r * inFeatures + k] += g * w[k * outFeatures + j];
                    dw[k * outFeatures + j] += x[r * inFeatures + k] * g;
                }
            }

        var errors = new System.Collections.Generic.Dictionary<(bool Declared, string Output), double>();
        foreach (bool declared in new[] { false, true })
        {
            backend.DeclaredKernelsSwitch = declared;
            try
            {
                using var bx = backend.AllocateBuffer(x);
                using var bw = backend.AllocateBuffer(w);
                using var bb = backend.AllocateBuffer(b);
                using var by = backend.AllocateBuffer(y.Length);
                using var bdy = backend.AllocateBuffer(dy);
                using var bsaved = backend.AllocateBuffer(saved);
                // Pre-filled, so an element a route fails to write cannot pass as a zero.
                using var bdx = backend.AllocateBuffer(Fill(dx.Length, 5));
                using var bdw = backend.AllocateBuffer(Fill(dw.Length, 6));
                using var bdb = backend.AllocateBuffer(Fill(db.Length, 7));
                long before = backend.DeclaredProgramLaunches;
                Forward(backend, activation, bx, bw, bb, by, batch, inFeatures, outFeatures);
                Backward(backend, activation, bdy, bx, bw, bsaved, bdx, bdw, bdb, batch, inFeatures, outFeatures);
                backend.Synchronize();
                long launches = backend.DeclaredProgramLaunches - before;
                string route = declared ? "declared" : "hand-written";
                Assert.True(launches == (declared ? 2 : 0), route + ": " + launches + " declared launches");
                errors[(declared, "y")] = MaxError(y, backend.DownloadBuffer(by));
                errors[(declared, "dx")] = MaxError(dx, backend.DownloadBuffer(bdx));
                errors[(declared, "dw")] = MaxError(dw, backend.DownloadBuffer(bdw));
                errors[(declared, "dbias")] = MaxError(db, backend.DownloadBuffer(bdb));
            }
            finally
            {
                backend.DeclaredKernelsSwitch = null;
            }
        }
        foreach (string output in new[] { "y", "dx", "dw", "dbias" })
        {
            double declaredError = errors[(true, output)], handWritten = errors[(false, output)];
            Assert.True(declaredError <= Math.Max(2e-5, 2 * handWritten),
                $"{output}: declared error {declaredError:E3}, hand-written {handWritten:E3}");
        }
    }

    /// <summary>
    /// The linear stages as one resident program -- layer 1 forward, layer 2 forward on its output, layer 2 backward --
    /// are bit-identical to the same stages launched separately, in one cooperative launch.
    /// </summary>
    [SkippableTheory]
    [InlineData(5, 33, 17)]
    [InlineData(64, 128, 96)]
    public void LinearStagesRunAsOneResidentLaunchBitIdenticalToSeparateLaunches(int batch, int inFeatures, int outFeatures)
    {
        using var backend = OpenBackendOrSkip();
        Skip.IfNot(backend.DeclaredKernelsEnabled && backend.SupportsResidentProgram,
            "A declared resident program needs the sm_75+ conveyor and a cooperative launch.");
        float[] x = Fill(batch * inFeatures, 1), w1 = Fill(inFeatures * outFeatures, 2), b1 = Fill(outFeatures, 3);
        float[] w2 = Fill(outFeatures * outFeatures, 4), b2 = Fill(outFeatures, 5), dy = Fill(batch * outFeatures, 6);
        float[] saved = Fill(batch * outFeatures, 7);
        using var barrier = backend.AllocateResidentBarrier();
        var results = new float[2][][];
        for (int mode = 0; mode < 2; mode++)
        {
            using var bx = backend.AllocateBuffer(x);
            using var bw1 = backend.AllocateBuffer(w1);
            using var bb1 = backend.AllocateBuffer(b1);
            using var bw2 = backend.AllocateBuffer(w2);
            using var bb2 = backend.AllocateBuffer(b2);
            using var bdy = backend.AllocateBuffer(dy);
            using var bsaved = backend.AllocateBuffer(saved);
            using var h = backend.AllocateBuffer(batch * outFeatures);
            using var y = backend.AllocateBuffer(batch * outFeatures);
            using var dh = backend.AllocateBuffer(batch * outFeatures);
            using var dw2 = backend.AllocateBuffer(outFeatures * outFeatures);
            using var db2 = backend.AllocateBuffer(outFeatures);
            var stages = new[]
            {
                backend.FusedLinearStage(bx, bw1, bb1, h, batch, inFeatures, outFeatures, CodegenActivationKind.ReLU),
                backend.FusedLinearStage(h, bw2, bb2, y, batch, outFeatures, outFeatures, CodegenActivationKind.Gelu),
                backend.FusedLinearBackwardStage(bdy, h, bw2, bsaved, dh, dw2, db2, batch, outFeatures, outFeatures,
                    CodegenActivationKind.Gelu),
            };
            long before = backend.DeclaredProgramLaunches;
            if (mode == 0) backend.RunDeclaredStagesSeparately(stages);
            else Assert.True(backend.RunDeclaredResidentProgram(stages, barrier) >= 1);
            backend.Synchronize();
            Assert.Equal(mode == 0 ? 3 : 1, backend.DeclaredProgramLaunches - before);
            results[mode] = new[]
            {
                backend.DownloadBuffer(h), backend.DownloadBuffer(y), backend.DownloadBuffer(dh),
                backend.DownloadBuffer(dw2), backend.DownloadBuffer(db2),
            };
        }
        for (int o = 0; o < results[0].Length; o++)
            for (int i = 0; i < results[0][o].Length; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(results[0][o][i]) ==
                    TestHelpers.MathCompat.SingleToInt32Bits(results[1][o][i]),
                    $"buffer {o}[{i}]: separate {results[0][o][i]} resident {results[1][o][i]}");
    }
}
