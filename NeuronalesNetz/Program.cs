// See https://aka.ms/new-console-template for more information

using NeuronalesNetz.algo;

var nm = new NetworkModel(1, true);
var options = new NeuralNetworkTrainingOptions
{
    Epochs = 5,
    EpochSize = 20000,
    LearningRate = .1f,
    TrainingRateOptions = TrainingRateOptions.Logarithmic,
    Convolution = false,
    Layers = [196]
};

nm.Train(options);

for (var i = 0; i < 10; i++) nm.Assess(options.Convolution);