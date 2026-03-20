using static AiDotNet.Tensors.Helpers.ComputationGraph;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Target device for a computation subgraph.
/// </summary>
public enum DeviceTarget
{
    /// <summary>Execute on CPU with SIMD acceleration.</summary>
    CPU,

    /// <summary>Execute on GPU via DirectGpu backend.</summary>
    GPU,

    /// <summary>Automatically choose based on operation type and data size.</summary>
    Auto
}

/// <summary>
/// Splits a computation graph into CPU and GPU subgraphs based on operation type
/// and data size, inserting data transfer nodes at partition boundaries.
/// </summary>
/// <remarks>
/// <para>
/// The partitioner assigns each graph node to CPU or GPU based on:
/// 1. Operation type: MatMul/Conv2D/Attention prefer GPU for large sizes
/// 2. Data size: small tensors stay on CPU to avoid transfer overhead
/// 3. Neighbor affinity: prefer the same device as connected nodes to minimize transfers
/// </para>
/// </remarks>
public sealed class GraphPartitioner
{
    /// <summary>Minimum tensor elements to consider GPU execution.</summary>
    public int GpuThreshold { get; set; } = 65536; // 64K elements

    /// <summary>Whether GPU is available on this system.</summary>
    public bool GpuAvailable { get; set; }

    /// <summary>
    /// Result of graph partitioning.
    /// </summary>
    public sealed class PartitionPlan
    {
        /// <summary>Device assignment for each graph node.</summary>
        public DeviceTarget[] NodeDevices { get; }

        /// <summary>Number of nodes assigned to CPU.</summary>
        public int CpuNodeCount { get; }

        /// <summary>Number of nodes assigned to GPU.</summary>
        public int GpuNodeCount { get; }

        /// <summary>Number of device boundary crossings (data transfers needed).</summary>
        public int TransferCount { get; }

        internal PartitionPlan(DeviceTarget[] nodeDevices, int transfers)
        {
            NodeDevices = nodeDevices;
            CpuNodeCount = nodeDevices.Count(d => d == DeviceTarget.CPU);
            GpuNodeCount = nodeDevices.Count(d => d == DeviceTarget.GPU);
            TransferCount = transfers;
        }
    }

    /// <summary>
    /// Partitions a computation graph between CPU and GPU.
    /// </summary>
    public PartitionPlan Partition(ComputationGraph graph)
    {
        var nodes = graph.Nodes;
        int count = nodes.Count;
        var devices = new DeviceTarget[count];

        // Phase 1: Initial device assignment based on operation type and size
        for (int i = 0; i < count; i++)
        {
            var node = nodes[i];
            devices[i] = AssignDevice(node);
        }

        // Phase 2: Minimize transfers — flip nodes that are surrounded by the other device
        bool changed = true;
        int iterations = 0;
        while (changed && iterations < 10)
        {
            changed = false;
            iterations++;

            for (int i = 0; i < count; i++)
            {
                if (nodes[i].IsInput) continue;

                int cpuNeighbors = 0, gpuNeighbors = 0;

                // Count input device affiliations
                foreach (int inputId in nodes[i].InputIds)
                {
                    if (inputId >= 0 && inputId < count)
                    {
                        if (devices[inputId] == DeviceTarget.CPU) cpuNeighbors++;
                        else gpuNeighbors++;
                    }
                }

                // Count output consumers
                for (int j = 0; j < count; j++)
                {
                    if (j == i) continue;
                    foreach (int inp in nodes[j].InputIds)
                    {
                        if (inp == i)
                        {
                            if (devices[j] == DeviceTarget.CPU) cpuNeighbors++;
                            else gpuNeighbors++;
                        }
                    }
                }

                // Only flip if there's a strong majority on the other device (>= 2:1 ratio)
                // to prevent oscillation and respect initial device assignment
                int total = cpuNeighbors + gpuNeighbors;
                if (total >= 2)
                {
                    DeviceTarget preferred = cpuNeighbors > gpuNeighbors * 2 ? DeviceTarget.CPU
                        : gpuNeighbors > cpuNeighbors * 2 ? DeviceTarget.GPU
                        : devices[i]; // keep current if no strong majority
                    if (preferred != devices[i] && CanRunOnDevice(nodes[i], preferred))
                    {
                        devices[i] = preferred;
                        changed = true;
                    }
                }
            }
        }

        // Count transfers (boundaries where device changes between connected nodes)
        int transfers = 0;
        for (int i = 0; i < count; i++)
        {
            foreach (int inputId in nodes[i].InputIds)
            {
                if (inputId >= 0 && inputId < count && devices[inputId] != devices[i])
                    transfers++;
            }
        }

        return new PartitionPlan(devices, transfers);
    }

    private DeviceTarget AssignDevice(GraphNode node)
    {
        if (node.IsInput) return DeviceTarget.CPU; // inputs start on CPU

        if (!GpuAvailable) return DeviceTarget.CPU;

        // Compute output tensor size
        int size = 1;
        foreach (int dim in node.OutputShape)
            size *= dim;

        // Small tensors: keep on CPU to avoid transfer overhead
        if (size < GpuThreshold) return DeviceTarget.CPU;

        // Large compute-bound ops: prefer GPU
        return node.Type switch
        {
            OpType.MatMul or OpType.BatchMatMul or OpType.Linear => DeviceTarget.GPU,
            OpType.Conv2D or OpType.Conv2DInto => DeviceTarget.GPU,
            OpType.ScaledDotProductAttention or OpType.MultiHeadAttention => DeviceTarget.GPU,
            OpType.Softmax when size >= GpuThreshold * 4 => DeviceTarget.GPU,
            OpType.GroupNorm or OpType.BatchNorm or OpType.LayerNorm => DeviceTarget.GPU,
            OpType.FusedConv2DBiasActivation or OpType.FusedGroupNormActivation => DeviceTarget.GPU,
            _ => DeviceTarget.CPU // Element-wise ops: CPU is fine
        };
    }

    private static bool CanRunOnDevice(GraphNode node, DeviceTarget device)
    {
        // All ops can run on CPU
        if (device == DeviceTarget.CPU) return true;

        // GPU: only compute-bound ops are worth transferring
        return node.Type switch
        {
            OpType.MatMul or OpType.BatchMatMul or OpType.Linear => true,
            OpType.Conv2D or OpType.Conv2DInto => true,
            OpType.ScaledDotProductAttention or OpType.MultiHeadAttention => true,
            OpType.Softmax or OpType.GroupNorm or OpType.BatchNorm or OpType.LayerNorm => true,
            OpType.FusedConv2DBiasActivation or OpType.FusedGroupNormActivation => true,
            // Activations can run on GPU but are memory-bound, not worth transferring for
            OpType.ReLU or OpType.Sigmoid or OpType.Tanh or OpType.GELU => true,
            _ => false
        };
    }
}
