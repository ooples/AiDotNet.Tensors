using System.Linq;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue A (#369): gating contract for the BlasManaged bench catalog.
///
/// <para>
/// <see cref="Catalog.ShapeCatalog.All"/> must hold 50-80 unique <see cref="Catalog.Shape"/>
/// records merged from instrumentation (<see cref="Catalog.ShapeInstrumenter"/>) and
/// curated standard ML workloads (<see cref="Catalog.WorkloadShapes"/>). Until both
/// sources land (A.2 + A.3) and A.4 merges them, the count assertion fails. That
/// failure is the intended gate.
/// </para>
/// </summary>
public class ShapeCatalogTest
{
    [Fact]
    public void Catalog_Has_Between_50_And_80_Shapes()
    {
        var shapes = ShapeCatalog.All;
        Assert.InRange(shapes.Count, 50, 80);
    }

    [Fact]
    public void All_Shapes_Have_Positive_Dimensions()
    {
        foreach (var s in ShapeCatalog.All)
        {
            Assert.True(s.M > 0, $"{s.Name}: M={s.M} not > 0");
            Assert.True(s.N > 0, $"{s.Name}: N={s.N} not > 0");
            Assert.True(s.K > 0, $"{s.Name}: K={s.K} not > 0");
        }
    }

    [Fact]
    public void All_Shape_Names_Are_Unique()
    {
        var names = ShapeCatalog.All.Select(s => s.Name).ToList();
        Assert.Equal(names.Count, names.Distinct().Count());
    }

    [Fact]
    public void WorkloadShapes_Cover_Bert_Resnet_Gpt_MobileNet()
    {
        var w = WorkloadShapes.All;
        Assert.Contains(w, s => s.Source.Contains("BERT"));
        Assert.Contains(w, s => s.Source.Contains("ResNet"));
        Assert.Contains(w, s => s.Source.Contains("GPT"));
        Assert.Contains(w, s => s.Source.Contains("MobileNet"));
    }

    [Fact]
    public void WorkloadShapes_Include_Both_Single_And_Double_Precision()
    {
        var w = WorkloadShapes.All;
        Assert.Contains(w, s => s.Dtype == DType.Single);
        Assert.Contains(w, s => s.Dtype == DType.Double);
    }

    [Fact]
    public void WorkloadShapes_Include_Transposed_Backward_Shapes()
    {
        var w = WorkloadShapes.All;
        Assert.Contains(w, s => s.TransA || s.TransB);
    }
}
