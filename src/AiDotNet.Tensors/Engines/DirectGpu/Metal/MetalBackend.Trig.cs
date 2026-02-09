// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Trigonometric, Hyperbolic, and Additional Unary operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    #region Trigonometric Operations

    /// <summary>
    /// Sine: B = sin(A)
    /// </summary>
    public void Sin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("sin_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Cosine: B = cos(A)
    /// </summary>
    public void Cos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("cos_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Tangent: B = tan(A)
    /// </summary>
    public void Tan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("tan_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Arc sine: B = asin(A)
    /// </summary>
    public void Asin(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("asin_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Arc cosine: B = acos(A)
    /// </summary>
    public void Acos(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("acos_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Arc tangent: B = atan(A)
    /// </summary>
    public void Atan(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("atan_kernel", A, B, size, _trigLibrary);
    }

    #endregion

    #region Hyperbolic Operations

    /// <summary>
    /// Hyperbolic sine: B = sinh(A)
    /// </summary>
    public void Sinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("sinh_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Hyperbolic cosine: B = cosh(A)
    /// </summary>
    public void Cosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("cosh_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Inverse hyperbolic sine: B = asinh(A)
    /// </summary>
    public void Asinh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("asinh_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Inverse hyperbolic cosine: B = acosh(A)
    /// </summary>
    public void Acosh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("acosh_kernel", A, B, size, _trigLibrary);
    }

    /// <summary>
    /// Inverse hyperbolic tangent: B = atanh(A)
    /// </summary>
    public void Atanh(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("atanh_kernel", A, B, size, _trigLibrary);
    }

    #endregion

    #region Additional Unary Operations

    /// <summary>
    /// Reciprocal: B = 1/A
    /// </summary>
    public void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("reciprocal_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Cube root: B = cbrt(A)
    /// </summary>
    public void Cbrt(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        // cbrt(x) = sign(x) * |x|^(1/3)
        Power(A, B, 1.0f / 3.0f, size);
    }

    /// <summary>
    /// Base-10 logarithm: B = log10(A)
    /// </summary>
    public void Log10(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        // log10(x) = log(x) / log(10)
        Log(A, B, size);
        Scale(B, B, 1.0f / 2.302585092994046f, size);
    }

    /// <summary>
    /// Negate: B = -A
    /// </summary>
    public void Negate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        Scale(A, B, -1.0f, size);
    }

    /// <summary>
    /// Floor: B = floor(A)
    /// </summary>
    public void Floor(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("floor_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Ceiling: B = ceil(A)
    /// </summary>
    public void Ceiling(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("ceil_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Round: B = round(A)
    /// </summary>
    public void Round(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("round_kernel", A, B, size, _elementWiseLibrary);
    }

    /// <summary>
    /// Truncate: B = trunc(A)
    /// </summary>
    public void Truncate(IGpuBuffer A, IGpuBuffer B, int size)
    {
        ThrowIfDisposed();
        ExecuteUnaryOp("trunc_kernel", A, B, size, _elementWiseLibrary);
    }

    #endregion
}
