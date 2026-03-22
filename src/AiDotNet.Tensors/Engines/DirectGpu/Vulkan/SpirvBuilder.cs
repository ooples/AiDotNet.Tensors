using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Generates SPIR-V binary for simple compute shaders programmatically.
/// Supports element-wise unary (1 in, 1 out), binary (2 in, 1 out),
/// and scalar (1 in, 1 scalar, 1 out) operations.
/// </summary>
internal static class SpirvBuilder
{
    /// <summary>
    /// SPIR-V opcodes used in compute shader generation.
    /// </summary>
    private const uint OpCapability = 0x0011;
    private const uint OpMemoryModel = 0x000E;
    private const uint OpEntryPoint = 0x000F;
    private const uint OpExecutionMode = 0x0010;
    private const uint OpDecorate = 0x0047;
    private const uint OpMemberDecorate = 0x0048;
    private const uint OpTypeVoid = 0x0013;
    private const uint OpTypeFunction = 0x0021;
    private const uint OpTypeInt = 0x0015;
    private const uint OpTypeFloat = 0x0016;
    private const uint OpTypeVector = 0x0017;
    private const uint OpTypeBool = 0x0014;
    private const uint OpTypeRuntimeArray = 0x001D;
    private const uint OpTypeStruct = 0x001E;
    private const uint OpTypePointer = 0x0020;
    private const uint OpConstant = 0x002B;
    private const uint OpVariable = 0x003B;
    private const uint OpFunction = 0x0036;
    private const uint OpLabel = 0x00F8;
    private const uint OpAccessChain = 0x0041;
    private const uint OpLoad = 0x003D;
    private const uint OpStore = 0x003E;
    private const uint OpReturn = 0x00FD;
    private const uint OpFunctionEnd = 0x0038;
    private const uint OpULessThan = 0x00B0;
    private const uint OpSelectionMerge = 0x00F7;
    private const uint OpBranchConditional = 0x00FA;
    private const uint OpBranch = 0x00F9;
    // Float ops
    private const uint OpFAdd = 0x0081;
    private const uint OpFSub = 0x0083;
    private const uint OpFMul = 0x0085;
    private const uint OpFDiv = 0x0088;

    /// <summary>
    /// Builds a SPIR-V binary for a simple binary element-wise operation: C[i] = A[i] op B[i]
    /// </summary>
    public static uint[] BuildBinaryElementWise(uint floatOpcode)
    {
        // This produces the same SPIR-V structure as VectorAdd/Sub/Mul/Div
        // but with a configurable float operation opcode
        var words = new List<uint>();
        uint bound = 0x30;

        // Header
        words.AddRange(new uint[] { 0x07230203, 0x00010000, 0x00080001, bound, 0 });
        // Capability Shader
        Emit(words, OpCapability, 2, 0x00000001);
        // MemoryModel Logical GLSL450
        Emit(words, OpMemoryModel, 3, 0, 1);
        // EntryPoint GLCompute %1 "main" %2
        words.Add(MakeWord(OpEntryPoint, 6)); words.Add(5); words.Add(1);
        words.Add(0x6E69616D); words.Add(0); words.Add(2);
        // ExecutionMode LocalSize 256 1 1
        Emit(words, OpExecutionMode, 6, 1, 0x11, 256, 1, 1);

        // Decorations
        Decorate(words, 2, 0x0B, 0x1C); // GlobalInvocationId
        DecorateBinding(words, 3, 0, 0); // A
        DecorateBinding(words, 4, 0, 1); // B
        DecorateBinding(words, 5, 0, 2); // C
        Emit(words, OpDecorate, 4, 6, 6, 4); // ArrayStride 4
        Emit(words, OpDecorate, 3, 7, 3); // BufferBlock
        words.Add(MakeWord(OpMemberDecorate, 5)); words.Add(7); words.Add(0); words.Add(0x23); words.Add(0);
        Emit(words, OpDecorate, 3, 0x13, 2); // Push constant Block
        words.Add(MakeWord(OpMemberDecorate, 5)); words.Add(0x13); words.Add(0); words.Add(0x23); words.Add(0);

        // Types
        Emit(words, OpTypeVoid, 2, 8);
        Emit(words, OpTypeFunction, 3, 9, 8);
        words.Add(MakeWord(OpTypeInt, 4)); words.Add(0xA); words.Add(32); words.Add(0); // uint
        words.Add(MakeWord(OpTypeVector, 4)); words.Add(0xB); words.Add(0xA); words.Add(3); // uvec3
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0xC); words.Add(1); words.Add(0xB); // ptr Input uvec3
        words.Add(MakeWord(OpTypeInt, 4)); words.Add(0xD); words.Add(32); words.Add(1); // int
        words.Add(MakeWord(OpConstant, 4)); words.Add(0xD); words.Add(0xE); words.Add(0); // const 0
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0xF); words.Add(1); words.Add(0xA); // ptr Input uint
        words.Add(MakeWord(OpTypeFloat, 3)); words.Add(0x10); words.Add(32); // float
        words.Add(MakeWord(OpTypeBool, 2)); words.Add(0x1C);
        words.Add(MakeWord(OpTypeRuntimeArray, 3)); words.Add(6); words.Add(0x10); // float[]
        words.Add(MakeWord(OpTypeStruct, 3)); words.Add(7); words.Add(6); // struct { float[] }
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0x11); words.Add(2); words.Add(7);
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0x12); words.Add(2); words.Add(0x10);
        words.Add(MakeWord(OpTypeStruct, 3)); words.Add(0x13); words.Add(0xA); // push constant
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0x14); words.Add(9); words.Add(0x13);
        words.Add(MakeWord(OpTypePointer, 4)); words.Add(0x15); words.Add(9); words.Add(0xA);

        // Variables
        words.Add(MakeWord(OpVariable, 4)); words.Add(0xC); words.Add(2); words.Add(1);
        words.Add(MakeWord(OpVariable, 4)); words.Add(0x11); words.Add(3); words.Add(2);
        words.Add(MakeWord(OpVariable, 4)); words.Add(0x11); words.Add(4); words.Add(2);
        words.Add(MakeWord(OpVariable, 4)); words.Add(0x11); words.Add(5); words.Add(2);
        words.Add(MakeWord(OpVariable, 4)); words.Add(0x14); words.Add(0x16); words.Add(9);

        // Main function
        words.Add(MakeWord(OpFunction, 5)); words.Add(8); words.Add(1); words.Add(0); words.Add(9);
        words.Add(MakeWord(OpLabel, 2)); words.Add(0x17);
        // idx = gl_GlobalInvocationID.x
        words.Add(MakeWord(OpAccessChain, 5)); words.Add(0xF); words.Add(0x18); words.Add(2); words.Add(0xE);
        words.Add(MakeWord(OpLoad, 4)); words.Add(0xA); words.Add(0x19); words.Add(0x18);
        // size
        words.Add(MakeWord(OpAccessChain, 5)); words.Add(0x15); words.Add(0x1A); words.Add(0x16); words.Add(0xE);
        words.Add(MakeWord(OpLoad, 4)); words.Add(0xA); words.Add(0x1B); words.Add(0x1A);
        // if (idx < size)
        words.Add(MakeWord(OpULessThan, 5)); words.Add(0x1C); words.Add(0x1D); words.Add(0x19); words.Add(0x1B);
        words.Add(MakeWord(OpSelectionMerge, 3)); words.Add(0x1E); words.Add(0);
        words.Add(MakeWord(OpBranchConditional, 4)); words.Add(0x1D); words.Add(0x1F); words.Add(0x1E);
        words.Add(MakeWord(OpLabel, 2)); words.Add(0x1F);
        // a[idx]
        words.Add(MakeWord(OpAccessChain, 6)); words.Add(0x12); words.Add(0x20); words.Add(3); words.Add(0xE); words.Add(0x19);
        words.Add(MakeWord(OpLoad, 4)); words.Add(0x10); words.Add(0x21); words.Add(0x20);
        // b[idx]
        words.Add(MakeWord(OpAccessChain, 6)); words.Add(0x12); words.Add(0x22); words.Add(4); words.Add(0xE); words.Add(0x19);
        words.Add(MakeWord(OpLoad, 4)); words.Add(0x10); words.Add(0x23); words.Add(0x22);
        // op(a, b)
        words.Add(MakeWord(floatOpcode, 5)); words.Add(0x10); words.Add(0x24); words.Add(0x21); words.Add(0x23);
        // c[idx] = result
        words.Add(MakeWord(OpAccessChain, 6)); words.Add(0x12); words.Add(0x25); words.Add(5); words.Add(0xE); words.Add(0x19);
        words.Add(MakeWord(OpStore, 3)); words.Add(0x25); words.Add(0x24);
        // branch + return
        words.Add(MakeWord(OpBranch, 2)); words.Add(0x1E);
        words.Add(MakeWord(OpLabel, 2)); words.Add(0x1E);
        words.Add(MakeWord(OpReturn, 1));
        words.Add(MakeWord(OpFunctionEnd, 1));

        return words.ToArray();
    }

    /// <summary>Generates SPIR-V for C = A + B</summary>
    public static uint[] BinaryAdd() => BuildBinaryElementWise(OpFAdd);
    /// <summary>Generates SPIR-V for C = A - B</summary>
    public static uint[] BinarySub() => BuildBinaryElementWise(OpFSub);
    /// <summary>Generates SPIR-V for C = A * B</summary>
    public static uint[] BinaryMul() => BuildBinaryElementWise(OpFMul);
    /// <summary>Generates SPIR-V for C = A / B</summary>
    public static uint[] BinaryDiv() => BuildBinaryElementWise(OpFDiv);

    private static uint MakeWord(uint opcode, uint wordCount)
    {
        return (wordCount << 16) | opcode;
    }

    private static void Emit(List<uint> words, uint opcode, uint wordCount, params uint[] operands)
    {
        words.Add(MakeWord(opcode, (uint)(1 + operands.Length)));
        words.AddRange(operands);
    }

    private static void Decorate(List<uint> words, uint target, uint decoration, uint value)
    {
        words.Add(MakeWord(OpDecorate, 4));
        words.Add(target);
        words.Add(decoration);
        words.Add(value);
    }

    private static void DecorateBinding(List<uint> words, uint target, uint set, uint binding)
    {
        Decorate(words, target, 0x22, set); // DescriptorSet
        Decorate(words, target, 0x21, binding); // Binding
    }
}
