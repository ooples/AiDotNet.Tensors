// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1489: the register-resident training BatchNorm (IBatchNormVarianceBackend) on OpenCL, through PortableBatchNorm and
    // BatchNormRegisterKernelTemplate's kernels, compiled at initialization beside the Winograd modules.
    public sealed partial class OpenClBackend : IBatchNormVarianceBackend
    {
        // Every variant's reduction needs all 256 threads in one work-group: a driver that caps a kernel below that
        // (register pressure from the 32-value variants) leaves the composed path. Decided once.
        private int _batchNormRegisterReady;

        private bool BatchNormRegisterReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _batchNormRegisterReady);
                if (cached != 0) return cached == 1;
                bool ready = true;
                foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames()
                             .Concat(BatchNormRegisterKernelTemplate.GetUnpackedKernelNames()))
                {
                    int size = HasTunedKernel(name) ? KernelWorkGroupSize(name) : -1;
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    ready &= size >= BatchNormRegisterKernelTemplate.GroupSize;
                }
                System.Threading.Volatile.Write(ref _batchNormRegisterReady, ready ? 1 : 2);
                return ready;
            }
        }

        // Compiles the module; a driver that rejects it leaves the composed BatchNorm path, and says why.
        private void CompileBatchNormRegisterModule()
        {
            var context = _context;
            if (context is null) return;
            try
            {
                var program = CompileOrLoadCached(BatchNormRegisterKernelTemplate.GetModuleSourceWithUnpacked(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "register-resident BatchNorm kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in BatchNormRegisterKernelTemplate.GetKernelNames())
                        built.Add((name, new DirectOpenClKernel(context, program, name)));
                    foreach (var name in BatchNormRegisterKernelTemplate.GetUnpackedKernelNames())
                        built.Add((name, new DirectOpenClKernel(context, program, name)));
                }
                catch
                {
                    foreach (var (_, kernel) in built) kernel.Dispose();
                    program.Dispose();
                    throw;
                }
                _programs.Add(program);
                foreach (var (name, kernel) in built) _kernelCache[name] = kernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL register-resident BatchNorm kernels unavailable: {ex.Message}");
            }
        }

        private void LaunchBatchNormRegister(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * BatchNormRegisterKernelTemplate.GroupSize, BatchNormRegisterKernelTemplate.GroupSize);
        }

        bool IBatchNormVarianceBackend.TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output,
            IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance,
            int batch, int channels, int spatial, float epsilon, bool relu, IGpuBuffer? residual)
            => BatchNormRegisterReady && PortableBatchNorm.TryForward(this, LaunchBatchNormRegister, input, output, gamma, beta,
                saveMean, saveVariance, batch, channels, spatial, epsilon, relu, residual, unpacked: true);

        bool IBatchNormVarianceBackend.TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input,
            IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma,
            IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput,
            IGpuBuffer? gradResidual, IGpuBuffer? beta, IGpuBuffer? gradOutput2)
            => BatchNormRegisterReady && PortableBatchNorm.TryBackward(this, LaunchBatchNormRegister, gradOutput, input, gamma,
                saveMean, saveVariance, gradInput, gradGamma, gradBeta, batch, channels, spatial, epsilon, reluOutput,
                gradResidual, beta, gradOutput2, unpacked: true);
    }
}
