using System.Diagnostics;
using IO;
using MathLib.Linalg;
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

    private readonly MnistReader _mnist;

    private readonly ModelOptions _options;
    private readonly IPooling _pooling;

    /// <summary>
    ///     Tuple array of Vector and Matrix where the matrix is the respective incoming weighting matrix
    ///     for the layer and the
    ///     vector the output layer. if the matrix is null the layer is equal to the input layer and there
    ///     for the parsed
    ///     image.
    /// </summary>
    private (Matrix, Matrix?)[] _layers;

    public NetworkModel(ModelOptions? options = null, params (Matrix, Matrix?)[] layers)
    {
        _mnist = new MnistReader();
        _options = options ?? ModelOptions.Default;
        _activator = ActivatorFactory.Get(_options.ActivatorFunction);
        // create layers

        _pooling = _options.Pooling switch
        {
            Pooling.None => new GenericPooling(m => m),
            Pooling.Linear => new LinearPooling(),
            Pooling.Parabolic => new ParabolicFactorPooling(),
            Pooling.Linear2D => new Linear2DPooling(_options.InputWidth),
            _ => throw new ArgumentOutOfRangeException()
        };
        // initialize tuples of layers and weights
        InitLayers(_options, layers);
    }
    private void InitLayers(ModelOptions options, (Matrix, Matrix?)[]? layers)
    {
        if (layers is not null && layers.Length > 0)
        {
            _layers = layers;

            return;
        }

        var n = _options.Layers.Length + 2; // input and output layer considered
        _layers = new (Matrix, Matrix?)[n];
        var inputSize = _pooling.Pool(new Matrix(options.BatchSize, options.InputFeatures)).Cols;
        _layers[0] = (new Matrix(options.BatchSize, inputSize), null);

        var previousLayerSize = inputSize;
        for (var i = 1; i < n; i++)
        {
            var layerSize = i > _options.Layers.Length ? _layers[i - 1].Item1.Cols : _options.Layers[i - 1];
            _layers[i] = i == n - 1
                ? (new Matrix(options.BatchSize, _options.OutputFeatures), new Matrix(_options.OutputFeatures, previousLayerSize))
                : (new Matrix(options.BatchSize, layerSize), Matrix.Random(layerSize, previousLayerSize));
            previousLayerSize = _layers[i].Item1.Cols;
        }

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

            for (var j = 0; j < _options.EpochSize / _options.BatchSize; j++)
            {
                var imgBatch = _mnist.TrainImages(j, _options.BatchSize);

                var inputFeatureBatched = _pooling.Pool(Matrix.FromBytes(imgBatch.Select(x => x.Image).ToArray()));

                var res = ForwardPass(inputFeatureBatched);
                var error = AssessError(imgBatch.Select(x => (int)x.Label).ToArray(), res);

                PropagateError(error, _options.LearningRate);
                Progress.PrintProgress(j + 1, _options.EpochSize / _options.BatchSize, sw);
            }

            sw.Reset();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Assess(sw);
            _mnist.Shuffle();


            // adjust learning rate if needed
            switch (_options.TrainingRateOptions)
            {
                case TrainingRateOptions.Constant: break;
                case TrainingRateOptions.Logarithmic: _options.LearningRate /= 1 + i; break;
                case TrainingRateOptions.Linear: _options.LearningRate *= 1 - (float)i / _options.Epochs; break;
                default: throw new ArgumentOutOfRangeException();
            }

        }
    }

    private void PropagateError(Matrix error, float learningrate)
    {
        var err = new Matrix[_layers.Length - 1];
        err[^1] = error;
        for (var i = _layers.Length - 2; i > 0; i--)
        {
            var m = _layers[i + 1].Item2!.T;
            err[i - 1] = (m * err[i].T).T;
        }
        for (var i = _layers.Length - 1; i > 0; i--)
        {
            var output = _layers[i].Item1;
            var next = _layers[i - 1].Item1;
            var e = err[i - 1];
            var diff = learningrate * _activator.Derivative(next, output, e);

            // CS8604: false positive, matrix will never be null, but must be nullable due to
            // the situation that each layer but the output layer needs a corresponding weight matrix 
            _layers[i].Item2 += diff;

        }
    }

    private Matrix ForwardPass(Matrix m, bool assessment = false)
    {

        var n = assessment ? 1 : _options.BatchSize;
        for (var batch = 0; batch < n; batch++)
        {
            NormalizeBytes(m[batch]);
            _layers[0] = (m, null);
            // item 1 is the layer vector while item 2 refers to the weight matrix.
            for (var i = 1; i < _layers.Length; i++)
            {
                var weights = _layers[i].Item2;
                if (weights != null)
                {
                    var r = _layers[i - 1].Item1 * weights.T;

                    _layers[i].Item1 = _activator.Activate(r);
                }
            }
        }

        return _layers[^1].Item1;
    }

    private Matrix AssessError(int[] lable, Matrix output)
    {
        var tmp = new Matrix(_options.BatchSize, output[0].Length);
        for (var batch = 0; batch < _options.BatchSize; batch++)
        for (var i = 0; i < output[batch].Length; i++)
        {
            var target = lable[batch] == i ? Expect : 0;
            tmp[batch][i] = target - output[batch][i];
        }

        return tmp;
    }

    /// <summary>
    ///     Evaluates the accuracy of the network using test data.
    /// </summary>
    public void Assess()
    {
        Stopwatch sw = new();
        Assess(sw);
    }

    private void Assess(Stopwatch sw, int iterations = 1000)
    {
        Random rand = new();
        var hits = 0;
        sw.Start();
        var n = iterations / _options.BatchSize;
        for (var i = 0; i < n; i++)
        {
            // can run in parallel if needed
            var imageData = _mnist.Image();
            var inputFeatures = Vector.FromBytes(imageData.Image).ToMatrix();
            var res = ForwardPass(_pooling.Pool(inputFeatures), true);
            if (res[0].Max() == imageData.Label) hits++;
            Progress.PrintProgress(i, n, sw, hits);
        }
        _mnist.Shuffle();
        Console.WriteLine();
    }

    private void NormalizeBytes(Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] /= 256.0;
    }

    private Matrix Predict(byte[][] inputFeatures) => ForwardPass(_pooling.Pool(Matrix.FromBytes(inputFeatures)));

    public int PredictLabel(byte[][] inputFeatures) => Predict(inputFeatures)[0].Max();

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
    public int[] Layers { get; init; } = [];

    public ActivatorFunctions ActivatorFunction { get; init; } = ActivatorFunctions.ReLU;

    public Pooling Pooling { get; init; } = Pooling.None;
    public int InputWidth { get; set; } = 28; // default for MNIST data set
    public int InputFeatures { get; init; } = 784; // default for MNIST data set
    public int OutputFeatures { get; init; } = 10; // default for MNIST data set

    public int BatchSize { get; init; } = 1;

    public static ModelOptions Default => new();
}