using NeuralNetworkLib.Model;

namespace Tests.NeuralNetwork.Model;

[TestFixture]
public class ModelOptionsTests
{
    [Test]
    public void Default_ReturnsOptionsWithDefaultValues()
    {
        var options = ModelOptions.Default;
        
        Assert.That(options, Is.Not.Null);
        Assert.That(options.Epochs, Is.EqualTo(5));
        Assert.That(options.EpochSize, Is.EqualTo(1750));
        Assert.That(options.LearningRate, Is.EqualTo(0.1f));
        Assert.That(options.TrainingRateOptions, Is.EqualTo(TrainingRateOptions.Constant));
        Assert.That(options.Convolution, Is.False);
        Assert.That(options.Layers, Is.Empty);
        Assert.That(options.ActivatorFunction, Is.EqualTo(ActivatorFunctions.ReLU));
        Assert.That(options.Pooling, Is.EqualTo(Pooling.None));
        Assert.That(options.InputWidth, Is.EqualTo(28));
        Assert.That(options.InputFeatures, Is.EqualTo(784));
        Assert.That(options.OutputFeatures, Is.EqualTo(10));
        Assert.That(options.BatchSize, Is.EqualTo(1));
    }

    [Test]
    public void Constructor_WithInitValues_SetsProperties()
    {
        var options = new ModelOptions
        {
            Epochs = 10,
            EpochSize = 2000,
            LearningRate = 0.05f,
            TrainingRateOptions = TrainingRateOptions.Logarithmic,
            Convolution = true,
            Layers = new[] { 128, 64 },
            ActivatorFunction = ActivatorFunctions.Sigmoid,
            Pooling = Pooling.Linear,
            InputWidth = 32,
            InputFeatures = 1024,
            OutputFeatures = 5,
            BatchSize = 4
        };

        Assert.That(options.Epochs, Is.EqualTo(10));
        Assert.That(options.EpochSize, Is.EqualTo(2000));
        Assert.That(options.LearningRate, Is.EqualTo(0.05f));
        Assert.That(options.TrainingRateOptions, Is.EqualTo(TrainingRateOptions.Logarithmic));
        Assert.That(options.Convolution, Is.True);
        Assert.That(options.Layers, Is.EqualTo(new[] { 128, 64 }));
        Assert.That(options.ActivatorFunction, Is.EqualTo(ActivatorFunctions.Sigmoid));
        Assert.That(options.Pooling, Is.EqualTo(Pooling.Linear));
        Assert.That(options.InputWidth, Is.EqualTo(32));
        Assert.That(options.InputFeatures, Is.EqualTo(1024));
        Assert.That(options.OutputFeatures, Is.EqualTo(5));
        Assert.That(options.BatchSize, Is.EqualTo(4));
    }

    [Test]
    public void LearningRate_CanBeModified()
    {
        var options = new ModelOptions { LearningRate = 0.2f };
        
        options.LearningRate = 0.3f;
        
        Assert.That(options.LearningRate, Is.EqualTo(0.3f));
    }

    [Test]
    public void InputWidth_CanBeModified()
    {
        var options = new ModelOptions { InputWidth = 32 };
        
        options.InputWidth = 64;
        
        Assert.That(options.InputWidth, Is.EqualTo(64));
    }
}

[TestFixture]
public class ActivatorFunctionsEnumTests
{
    [Test]
    public void ActivatorFunctions_HasExpectedValues()
    {
        var values = Enum.GetValues<ActivatorFunctions>();
        var expectedValues = new[] { 
            ActivatorFunctions.Sigmoid, 
            ActivatorFunctions.ReLU, 
            ActivatorFunctions.LeakyReLU, 
            ActivatorFunctions.Tanh 
        };

        Assert.That(values.Length, Is.EqualTo(4));
        foreach (var expected in expectedValues)
        {
            Assert.That(values, Contains.Item(expected));
        }
    }

    [Test]
    public void ActivatorFunctions_EnumValuesParseable()
    {
        Assert.That(Enum.Parse<ActivatorFunctions>("Sigmoid"), Is.EqualTo(ActivatorFunctions.Sigmoid));
        Assert.That(Enum.Parse<ActivatorFunctions>("ReLU"), Is.EqualTo(ActivatorFunctions.ReLU));
        Assert.That(Enum.Parse<ActivatorFunctions>("LeakyReLU"), Is.EqualTo(ActivatorFunctions.LeakyReLU));
        Assert.That(Enum.Parse<ActivatorFunctions>("Tanh"), Is.EqualTo(ActivatorFunctions.Tanh));
    }
}

[TestFixture]
public class TrainingRateOptionsEnumTests
{
    [Test]
    public void TrainingRateOptions_HasExpectedValues()
    {
        var values = Enum.GetValues<TrainingRateOptions>();
        var expectedValues = new[] { 
            TrainingRateOptions.Constant, 
            TrainingRateOptions.Logarithmic, 
            TrainingRateOptions.Linear 
        };

        Assert.That(values.Length, Is.EqualTo(3));
        foreach (var expected in expectedValues)
        {
            Assert.That(values, Contains.Item(expected));
        }
    }
}

[TestFixture]
public class PoolingEnumTests
{
    [Test]
    public void Pooling_HasExpectedValues()
    {
        var values = Enum.GetValues<Pooling>();
        var expectedValues = new[] { 
            Pooling.None, 
            Pooling.Linear, 
            Pooling.Parabolic, 
            Pooling.Linear2D 
        };

        Assert.That(values.Length, Is.EqualTo(4));
        foreach (var expected in expectedValues)
        {
            Assert.That(values, Contains.Item(expected));
        }
    }
}