using System.Diagnostics;
using System.Text.Json;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Configuration;
using AiDotNet.Dashboard.Dashboard;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TrainingMonitoring.Dashboard;

string mode = args[0], run = args[1];
string PathIn(string name) => Path.Combine(run, name);
if (mode == "display")
{
    using var monitor = new TrainingMonitorDashboard<float>(args[3]) { ShowCurves = false, ShowResources = false, UseColors = false };
    using var live = new LiveDashboard(PathIn("dashboard"), args[3], int.Parse(args[2])) { RefreshIntervalMs = 80 };
    live.Start();
    var session = monitor.StartSession(args[3]);
    File.WriteAllText(PathIn("url"), live.Url);
    while (!File.Exists(PathIn("live.jsonl"))) Thread.Sleep(20);
    using var input = new StreamReader(File.Open(PathIn("live.jsonl"), FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
    while (!File.Exists(PathIn("captured")))
    {
        var line = input.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) { Thread.Sleep(8); continue; }
        var row = JsonSerializer.Deserialize<Row>(line)!;
        monitor.LogMetrics(session, new() { ["Held-out accuracy (%)"] = 100f * row.correct / 320 }, row.step);
        live.LogScalars(monitor.GetCurrentMetrics(session).ToDictionary(x => x.Key, x => (double)x.Value), row.step);
    }
    monitor.EndSession(session);
    return;
}
string fixture = args[2];
const int batch = 64, width = 32, poolSize = 256, held = 5, warm = 8, steps = 4096;
AiDotNetEngine.AutoDetectAndConfigureGpu(verbose: true);
if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || !gpu.Name.Contains("CUDA")) throw new Exception("CUDA required");
int fused = 0;
TrainingDiagnosticsConfig.Level = TrainingDiagnosticLevel.PerStep;
TrainingDiagnosticsConfig.Sink = e => { if (e is FusedOptimizerPathEvent { Hit: true }) ++fused; };
var arch = new NeuralNetworkArchitecture<float>(inputType: InputType.TwoDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: width, inputWidth: width, outputSize: 10,
    layers: new List<ILayer<float>> { new LSTMLayer<float>(hiddenSize: 64),
        new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
        new DenseLayer<float>(10, activationFunction: null) });
var model = new LSTMNeuralNetwork<float>(arch, lossFunction: new CrossEntropyWithLogitsLoss<float>(), outputActivation: null);
model.SetBaseTrainOptimizer(new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3, Beta1 = .9, Beta2 = .999, Epsilon = 1e-8, WeightDecay = .01 }));
var data = new List<(Tensor<float> x, Tensor<float> y, byte[] labels)>();
using (var features = File.OpenRead(Path.Combine(fixture,"x.f32")))
using (var classes = File.OpenRead(Path.Combine(fixture,"y.u8")))
{
    for (int i = 0; i < poolSize + held; i++)
    {
        var values = new float[batch * width * width]; features.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()));
        var labels = new byte[batch]; classes.ReadExactly(labels);
        var targets = new float[batch * 10];
        for (int b = 0; b < batch; b++) targets[b * 10 + labels[b]] = 1;
        data.Add((new(values, [batch,width,width]), new(targets, [batch,10]), labels));
    }
}
model.Predict(data[0].x);
PyTorchStateDictImporter.Import(model, SafetensorsReader.Read(File.ReadAllBytes(args[3])), ["lstm","head"]);
int Evaluate() {
    int correct = 0;
    foreach (var item in data.Skip(poolSize)) {
        var scores = model.Predict(item.x).ToArray();
        for (int b = 0; b < batch; b++) {
            var s = scores.AsSpan(b*10,10);
            if (s.IndexOf(s.ToArray().Max()) == item.labels[b]) correct++;
        }
    }
    return correct;
}
using var output = new StreamWriter(PathIn("live.jsonl")) { AutoFlush = true };
void Report(int step, double seconds, int correct, double loss) => output.WriteLine(JsonSerializer.Serialize(new Row(step,seconds,correct,loss)));
var initial = Evaluate();
float firstLoss = 0;
for (int i = 0; i < warm; i++) { model.Train(data[i].x, data[i].y); if(i==0) firstLoss=model.GetLastLoss(); }
int correct = Evaluate();
gpu.SynchronizeStream();
Report(0, 0, correct, model.GetLastLoss());
File.WriteAllText(PathIn("ready"), JsonSerializer.Serialize(new { parameters=model.TrainableParameterCount, initial, firstLoss, device=gpu.Name }));
while (!File.Exists(PathIn("go"))) Thread.Sleep(1);
var timer = Stopwatch.StartNew();
for (int step=1; step<=steps; step++) {
    var item=data[(step-1+warm)%poolSize];
    model.Train(item.x,item.y);
    if (step%64==0) {
        var loss=model.GetLastLoss();
        correct=Evaluate();
        gpu.SynchronizeStream();
        Report(step,timer.Elapsed.TotalSeconds,correct,loss);
    }
}
if (fused!=steps+warm || model.TrainableParameterCount!=25482) throw new Exception("Training path mismatch");
File.WriteAllText(PathIn("done"), JsonSerializer.Serialize(new { fused, correct }));
record Row(int step, double seconds, int correct, double loss);
