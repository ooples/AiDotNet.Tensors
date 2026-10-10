using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Launches for the PyTorch-parity elementwise kernels (Kernels/CudaTorchParityElementwiseKernels).
public sealed partial class CudaBackend : ITorchParityElementwiseBackend
{
    private static string UnaryKernelName(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Celu => "parity_celu",
        TorchParityUnaryOp.Hardtanh => "parity_hardtanh",
        TorchParityUnaryOp.LogSigmoid => "parity_logsigmoid",
        TorchParityUnaryOp.Softsign => "parity_softsign",
        TorchParityUnaryOp.Logit => "parity_logit",
        TorchParityUnaryOp.Sinc => "parity_sinc",
        TorchParityUnaryOp.Erf => "parity_erf",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private static string BinaryKernelName(TorchParityBinaryOp op) => op switch
    {
        TorchParityBinaryOp.Fmax => "parity_fmax",
        TorchParityBinaryOp.Fmin => "parity_fmin",
        TorchParityBinaryOp.Heaviside => "parity_heaviside",
        TorchParityBinaryOp.GreaterEqual => "parity_greater_equal",
        TorchParityBinaryOp.LessEqual => "parity_less_equal",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    unsafe void ITorchParityElementwiseBackend.TorchParityUnary(
        TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
    {
        if (n <= 0) return;
        string name = UnaryKernelName(op);
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        using var _ = PushContext();
        IntPtr x = input.Handle, y = output.Handle;
        void** args = stackalloc void*[5];
        args[0] = &x; args[1] = &y; args[2] = &n; args[3] = &p0; args[4] = &p1;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }

    unsafe void ITorchParityElementwiseBackend.TorchParityBinary(
        TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        string name = BinaryKernelName(op);
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {name}");
        using var _ = PushContext();
        IntPtr pa = a.Handle, pb = b.Handle, y = output.Handle;
        void** args = stackalloc void*[4];
        args[0] = &pa; args[1] = &pb; args[2] = &y; args[3] = &n;
        LaunchKernel(kernel, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, args);
    }
}
