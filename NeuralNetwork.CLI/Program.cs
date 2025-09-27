using NeuralNetworkLib.Model;

var options = new NeuralNetworkTrainingOptions
{
    Epochs = 5,
    EpochSize = 20000,
    LearningRate = .1f,
    TrainingRateOptions = TrainingRateOptions.Logarithmic,
    Convolution = false,
    Layers = [196]
};

var nm = new NetworkModel(options);

nm.Train();

for (var i = 0; i < 10; i++) nm.Assess(options.Convolution);