using NeuralNetwork.Core.Model;
using NeuralNetwork.Model;

var options = new ModelOptions
{
    Epochs = 5,
    EpochSize = 800,
    LearningRate = .1f,
    TrainingRateOptions = TrainingRateOptions.Logarithmic,
    Convolution = false,
    ActivatorFunction = ActivatorFunctions.LeakyReLU,
    Layers = [196],
    Pooling = Pooling.Linear2D,
    BatchSize = 8
};

var nm = new NetworkModel(options);

nm.Train();

for (var i = 0; i < 10; i++) nm.Assess();