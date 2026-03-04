using System;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests;

public unsafe class VulkanStructDiagnostic
{
    private readonly ITestOutputHelper _output;
    public VulkanStructDiagnostic(ITestOutputHelper output) => _output = output;

    [Fact]
    public void PrintStructSizes()
    {
        _output.WriteLine($"sizeof(VkPipelineShaderStageCreateInfo) = {sizeof(VkPipelineShaderStageCreateInfo)}");
        _output.WriteLine($"sizeof(VkComputePipelineCreateInfo) = {sizeof(VkComputePipelineCreateInfo)}");
        _output.WriteLine($"sizeof(VkShaderModuleCreateInfo) = {sizeof(VkShaderModuleCreateInfo)}");
        _output.WriteLine($"sizeof(IntPtr) = {sizeof(IntPtr)}");
        _output.WriteLine($"sizeof(nuint) = {sizeof(nuint)}");

        VkComputePipelineCreateInfo ci = default;
        byte* pBase = (byte*)&ci;
        _output.WriteLine($"\nVkComputePipelineCreateInfo offsets:");
        _output.WriteLine($"  sType: {(byte*)&ci.sType - pBase}");
        _output.WriteLine($"  pNext: {(byte*)&ci.pNext - pBase}");
        _output.WriteLine($"  flags: {(byte*)&ci.flags - pBase}");
        _output.WriteLine($"  stage: {(byte*)&ci.stage - pBase}");
        _output.WriteLine($"  stage.module: {(byte*)&ci.stage.module - pBase}");
        _output.WriteLine($"  layout: {(byte*)&ci.layout - pBase}");
        _output.WriteLine($"  basePipelineHandle: {(byte*)&ci.basePipelineHandle - pBase}");
        _output.WriteLine($"  basePipelineIndex: {(byte*)&ci.basePipelineIndex - pBase}");
    }

    [SkippableFact]
    public void MinimalShaderPipelineTest()
    {
        var device = VulkanDevice.Instance;
        Skip.IfNot(device.Initialize(), "Vulkan device not available");

        _output.WriteLine($"Device: {device.DeviceName}");
        _output.WriteLine($"Device handle: 0x{device.Device:X}");

        // Minimal compute shader: just does nothing
        // #version 450
        // layout(local_size_x = 1) in;
        // void main() {}
        uint[] minimalSpirv = new uint[]
        {
            0x07230203, // Magic
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x00000006, // Bound = 6
            0x00000000, // Reserved

            // OpCapability Shader
            0x00020011, 0x00000001,
            // OpMemoryModel Logical GLSL450
            0x0003000E, 0x00000000, 0x00000001,
            // OpEntryPoint GLCompute %1 "main"
            0x0005000F, 0x00000005, 0x00000001, 0x6E69616D, 0x00000000,
            // OpExecutionMode %1 LocalSize 1 1 1
            0x00060010, 0x00000001, 0x00000011, 0x00000001, 0x00000001, 0x00000001,

            // %void = OpTypeVoid
            0x00020013, 0x00000002,
            // %func = OpTypeFunction %void
            0x00030021, 0x00000003, 0x00000002,

            // %1 = OpFunction %void None %func
            0x00050036, 0x00000002, 0x00000001, 0x00000000, 0x00000003,
            // %4 = OpLabel
            0x000200F8, 0x00000004,
            // OpReturn
            0x000100FD,
            // OpFunctionEnd
            0x00010038
        };

        // Create shader module directly
        var byteCode = new byte[minimalSpirv.Length * sizeof(uint)];
        Buffer.BlockCopy(minimalSpirv, 0, byteCode, 0, byteCode.Length);

        int uint32Count = byteCode.Length / 4;
        var alignedCode = new uint[uint32Count];
        Buffer.BlockCopy(byteCode, 0, alignedCode, 0, byteCode.Length);
        nuint alignedCodeSize = (nuint)(uint32Count * 4);

        IntPtr shaderModule;
        fixed (uint* pCode = alignedCode)
        {
            var smCreateInfo = new VkShaderModuleCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext = null,
                flags = 0,
                codeSize = alignedCodeSize,
                pCode = pCode
            };

            int smResult = VulkanNativeBindings.vkCreateShaderModule(
                device.Device, &smCreateInfo, IntPtr.Zero, out shaderModule);
            _output.WriteLine($"vkCreateShaderModule result: {smResult}");
            _output.WriteLine($"shaderModule handle: 0x{shaderModule:X}");
            Assert.Equal(0, smResult); // VK_SUCCESS
            Assert.NotEqual(IntPtr.Zero, shaderModule);
        }

        // Create empty descriptor set layout
        var dslCreateInfo = new VkDescriptorSetLayoutCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = null,
            flags = 0,
            bindingCount = 0,
            pBindings = null
        };

        IntPtr descriptorSetLayout;
        int dslResult = VulkanNativeBindings.vkCreateDescriptorSetLayout(
            device.Device, &dslCreateInfo, IntPtr.Zero, out descriptorSetLayout);
        _output.WriteLine($"vkCreateDescriptorSetLayout result: {dslResult}");
        _output.WriteLine($"descriptorSetLayout handle: 0x{descriptorSetLayout:X}");
        Assert.Equal(0, dslResult);

        // Create pipeline layout with no push constants
        var setLayout = descriptorSetLayout;
        var plCreateInfo = new VkPipelineLayoutCreateInfo
        {
            sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext = null,
            flags = 0,
            setLayoutCount = 1,
            pSetLayouts = &setLayout,
            pushConstantRangeCount = 0,
            pPushConstantRanges = null
        };

        IntPtr pipelineLayout;
        int plResult = VulkanNativeBindings.vkCreatePipelineLayout(
            device.Device, &plCreateInfo, IntPtr.Zero, out pipelineLayout);
        _output.WriteLine($"vkCreatePipelineLayout result: {plResult}");
        _output.WriteLine($"pipelineLayout handle: 0x{pipelineLayout:X}");
        Assert.Equal(0, plResult);

        // Create compute pipeline
        var entryPointBytes = Encoding.UTF8.GetBytes("main\0");
        fixed (byte* pEntryPoint = entryPointBytes)
        {
            var stageInfo = new VkPipelineShaderStageCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext = null,
                flags = 0,
                stage = VulkanNativeBindings.VK_SHADER_STAGE_COMPUTE_BIT,
                module = shaderModule,
                pName = pEntryPoint,
                pSpecializationInfo = null
            };

            _output.WriteLine($"\nAbout to create compute pipeline...");
            _output.WriteLine($"  stageInfo.sType: {stageInfo.sType}");
            _output.WriteLine($"  stageInfo.module: 0x{stageInfo.module:X}");
            _output.WriteLine($"  stageInfo.pName: 0x{(long)stageInfo.pName:X}");
            _output.WriteLine($"  pipelineLayout: 0x{pipelineLayout:X}");

            var cpCreateInfo = new VkComputePipelineCreateInfo
            {
                sType = VulkanNativeBindings.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext = null,
                flags = 0,
                stage = stageInfo,
                layout = pipelineLayout,
                basePipelineHandle = IntPtr.Zero,
                basePipelineIndex = -1
            };

            // Dump raw bytes of the create info
            byte* rawBytes = (byte*)&cpCreateInfo;
            _output.WriteLine($"\nRaw VkComputePipelineCreateInfo bytes ({sizeof(VkComputePipelineCreateInfo)} bytes):");
            for (int i = 0; i < sizeof(VkComputePipelineCreateInfo); i += 8)
            {
                string hex = "";
                for (int j = 0; j < 8 && i + j < sizeof(VkComputePipelineCreateInfo); j++)
                {
                    hex += $"{rawBytes[i + j]:X2} ";
                }
                _output.WriteLine($"  +{i:D3}: {hex}");
            }

            IntPtr pipeline;
            int cpResult = VulkanNativeBindings.vkCreateComputePipelines(
                device.Device, IntPtr.Zero, 1, &cpCreateInfo, IntPtr.Zero, &pipeline);
            _output.WriteLine($"vkCreateComputePipelines result: {cpResult}");
            _output.WriteLine($"pipeline handle: 0x{pipeline:X}");

            // Cleanup
            if (pipeline != IntPtr.Zero)
                VulkanNativeBindings.vkDestroyPipeline(device.Device, pipeline, IntPtr.Zero);
        }

        VulkanNativeBindings.vkDestroyPipelineLayout(device.Device, pipelineLayout, IntPtr.Zero);
        VulkanNativeBindings.vkDestroyDescriptorSetLayout(device.Device, descriptorSetLayout, IntPtr.Zero);
        VulkanNativeBindings.vkDestroyShaderModule(device.Device, shaderModule, IntPtr.Zero);
    }
}
