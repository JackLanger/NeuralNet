using System.Diagnostics;
using IO;
using MathLib.Linalg;
using NeuralNetworkLib.extension;
using NeuralNetworkLib.Model.Activation;
using NeuralNetworkLib.Model.Data.Compression;

namespace NeuralNetworkLib.Model;

/// <summary>
///     A Simple feed forward neural network model with backpropagation learning.
///     The model supports multiple hidden layers and optional input compression via convolution.
///     The network is designed to work with the MNIST dataset for handwritten digit recognition.
///     It allows for customization of the number of hidden layers, learning rate, and training epochs.
/// </summary>
public class NetworkModel {
    private const double Expect = 1;

    private readonly IActivatorFunction _activator;

    /// <summary>
    ///     Tuple array of Vector and Matrix where the matrix is the respective incoming weighting matrix
    ///     for the layer and the
    ///     vector the output layer. if the matrix is null the layer is equal to the input layer and there
    ///     for the parsed
    ///     image.
    /// </summary>
    private readonly (Vector, Matrix?)[] _layers;

    private readonly ModelOptions _options;
    private readonly IPooling _pooling;

    public NetworkModel(ModelOptions? options = null)
    {
        _options = options ?? ModelOptions.Default;
        _activator = ActivatorFactory.Get(_options.ActivatorFunction);
        // create layers
        var n = _options.HiddenLayerCount + 1;
        _layers = new (Vector, Matrix?)[n];
        _pooling = _options.Pooling switch
        {
            Pooling.None => new GenericPooling(v => v),
            Pooling.Linear => new LinearPooling(),
            Pooling.Parabolic => new ParabolicPooling(),
            Pooling.Linear2D => new Linear2DPooling(28),
            _ => throw new ArgumentOutOfRangeException()
        };
        // initialize tuples of layers and weights
        Setup(n, _options.Layers.Length > 0 ? _options.Layers : null, _options.Convolution);
    }

    private void Setup(int layerCount, int[]? layers = null, bool compress = false, bool randomStart = true)
    {

        // TODO: refactor this function. It is not robust and will not be able to handle different types of input data. It is specific for the creation of the layers and weights needed for the MNIST data sets 
        var prev = compress ? 196 : 784;

        _layers[0] = (new Vector(prev), null);
        if (layers is null)
        {
            for (var i = 1; i < layerCount; i++)
            {
                prev = _layers[i - 1].Item1.Length;
                var n = (int)Math.Sqrt(prev * 10) + 10;
                _layers[i] = randomStart ? (new Vector(n), new Matrix(n, prev).WithRandom() * .1) : (new Vector(n), new Matrix(n, prev).WithValue(.05));
            }
        }
        else
        {
            if (layers.Length != _options.HiddenLayerCount)
            {
                Setup(layerCount);
            }
            for (var i = 1; i < layerCount; i++)
            {
                prev = _layers[i - 1].Item1.Length;
                var n = layers[i - 1];
                _layers[i] = randomStart ? (new Vector(n), new Matrix(n, prev).WithRandom() * .1) : (new Vector(n), new Matrix(n, prev).WithValue(.05));
            }
        }
        _layers[^1] = (new Vector(10), new Matrix(10, _layers[^2].Item1.Length).WithRandom() * .1);
    }

    private static Vector FromBytes(byte[] b)
    {
        var v = new Vector(b.Length);
        for (var i = 0; i < b.Length; i++) v[i] = b[i];

        return v;
    }

    /// <summary>
    ///     Begin training of the network model.
    /// </summary>
    public void Train()
    {
        Stopwatch sw = new();
        for (var i = 0; i < _options.Epochs; i++)
        {
            sw.Start();
            Console.ForegroundColor = ConsoleColor.Green;

            for (var j = 0; j < _options.EpochSize; j++)
            {
                var inputFeatures = FromBytes(MnistReader.TrainImage(j));
                var res = ForwardPass(_pooling.Pool(inputFeatures));
                var error = AssessError(MnistReader.TrainLabel(j), res);
                PropagateError(error, _options.LearningRate);
                Progress.PrintProgress(j + 1, _options.EpochSize, sw);
            }

            sw.Reset();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Assess(sw);
            MnistReader.Shuffle();

            // adjust learning rate if needed
            switch (_options.TrainingRateOptions)
            {
                case TrainingRateOptions.Constant: break;
                case TrainingRateOptions.Logarithmic: _options.LearningRate = _options.LearningRate / (1 + i); break;
                case TrainingRateOptions.Linear: _options.LearningRate = _options.LearningRate * (1 - (float)i / _options.Epochs); break;
                default: throw new ArgumentOutOfRangeException();
            }

        }
    }

    private void PropagateError(Vector error, float learningrate)
    {
        var err = new Vector[_layers.Length - 1];
        err[^1] = error;
        for (var i = _layers.Length - 2; i > 0; i--) err[i - 1] = _layers[i + 1].Item2!.T * err[i];
        for (var i = _layers.Length - 1; i > 0; i--)
        {
            var output = _layers[i].Item1;
            var next = _layers[i - 1].Item1;
            var diff = learningrate * _activator.Derivative(next, output, err[i - 1]);

            // CS8604: false positive, matrix will never be null, but must be nullable due to
            // the situation that each layer but the output layer needs a corresponding weight matrix 
            _layers[i].Item2 += diff;

        }
    }

    private Vector ForwardPass(Vector v)
    {
        NormalizeBytes(v);
        _layers[0] = (v, null);
        for (var i = 1; i < _layers.Length; i++)
            // item 1 is the layer vector while item 2 refers to the weight matrix.
        {
            var matrix = _layers[i].Item2;
            if (matrix != null) _layers[i].Item1 = _activator.Activate(matrix * _layers[i - 1].Item1);
        }

        return _layers[^1].Item1;
    }

    private Vector AssessError(int lable, Vector output)
    {
        var tmp = new Vector(output.Length);
        for (var i = 0; i < output.Length; i++)
        {
            var target = lable == i ? Expect : 0;
            tmp[i] = target - output[i];
        }

        return tmp;
    }

    /// <summary>
    ///     Evaluates the accuracy of the network using test data.
    /// </summary>
    /// <param name="convolutionActive">
    ///     Indicates whether input data should be compressed (e.g., using
    ///     convolution).
    /// </param>
    public void Assess(bool convolutionActive)
    {
        Stopwatch sw = new();
        Assess(sw);
    }

    private void Assess(Stopwatch sw, int iterations = 1000)
    {
        Random rand = new();
        var hits = 0;
        sw.Start();
        for (var i = 0; i < iterations; i++)
        {
            var n = rand.Next(10_000);
            var inputFeatures = FromBytes(MnistReader.Image(n));
            var res = ForwardPass(_pooling.Pool(inputFeatures));
            if (res.Max() == MnistReader.Label(n)) hits++;
            Progress.PrintProgress(i + 1, iterations, sw, hits);
        }
        Console.WriteLine();
    }
    private void NormalizeBytes(Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] /= 256.0;
    }


    private Vector Predict(byte[] inputFeatures, bool compress = false) => ForwardPass(_pooling.Pool(FromBytes(inputFeatures)));

    public int PredictLabel(byte[] inputFeatures, bool compress = false)
    {
        var res = Predict(inputFeatures, compress);

        return res.Max();
    }
}

public enum TrainingRateOptions {
    Constant,
    Logarithmic,
    Linear
}

public enum Pooling {
    None,
    Linear,
    Parabolic,
    Linear2D
}

public class ModelOptions {
    public int Epochs { get; init; } = 5;
    public int EpochSize { get; init; } = 1750;
    public float LearningRate { get; set; } = .1f;
    public TrainingRateOptions TrainingRateOptions { get; init; } = TrainingRateOptions.Constant;
    public bool Convolution { get; init; }
    public int[] Layers { get; init; } = [256];

    public ActivatorFunctions ActivatorFunction { get; init; } = ActivatorFunctions.ReLU;
    public int HiddenLayerCount { get; set; } = 1;

    public Pooling Pooling { get; init; } = Pooling.None;

    public static ModelOptions Default => new();
}