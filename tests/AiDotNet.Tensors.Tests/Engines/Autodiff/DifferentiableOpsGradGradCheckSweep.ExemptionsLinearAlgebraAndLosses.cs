namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>Documented second-order exclusions for the linear-algebra, recurrent-cell and box-loss ops.</summary>
public partial class DifferentiableOpsGradGradCheckSweep
{
    static partial void AddExemptionsLinearAlgebraAndLosses()
    {
        // A designed gradient: CIoU = DIoU + α·v with α = v / (1 - IoU + v) held constant during backpropagation (the CIoU
        // paper; torchvision's complete_box_iou_loss computes it under no_grad). Finite differences differentiate α, so
        // neither the second- nor the third-order numerical check applies. CIoULossHigherOrderTests pins the first and
        // second derivatives in both boxes exactly against DIoU + α₀·v with α₀ frozen.
        NotSecondOrderCheckable["TensorCIoULoss"] =
            "designed gradient: the aspect weight alpha is a stop-gradient (pinned by CIoULossHigherOrderTests)";
    }
}
