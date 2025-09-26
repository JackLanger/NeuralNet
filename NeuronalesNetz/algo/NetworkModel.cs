using System.Diagnostics;
using IO;
using MathTools;
using NeuronalesNetz.extension;

namespace NeuronalesNetz.algo;

/// <summary>
///     A Simple feed forward neural network model with backpropagation learning.
///     The model supports multiple hidden layers and optional input compression via convolution.
///     The network is designed to work with the MNIST dataset for handwritten digit recognition.
///     It allows for customization of the number of hidden layers, learning rate, and training epochs.
///     TODO: Use Vector2 and Vector3 for convolution and pooling layers. This will allow for more
///     TODO: Implement saving and loading of the model to allow for persistence and sharing of trained
///     models.
///     TODO: Implement CUDA and ROCm support for GPU acceleration to handle larger datasets and
///     increase Performance.
///     TODO: Move Progress Updater to a separate Seperate Thread and use Observer Pattern to update
///     progress. This will allow for smoother UI updates and better separation of concerns.
///     TODO: Implement different activation functions and allow for customization of the activation
///     Funtion
///     TODO: Implement different loss functions and allow for customization of the loss function.
///     TODO: Implement regularization techniques such as dropout and weight decay to prevent
///     overfitting.
///     TODO: Implement batch training to improve training efficiency and stability.
///     TODO: Implement learning rate schedules to improve convergence and training efficiency.
///     TODO: Implement Dataproviders to allow for easy switching between different datasets and data
///     formats.
/// </summary>
public class NetworkModel {
    private const double Expect = 1;

    /// <summary>
    ///     Tuple array of Vector and Matrix where the matrix is the respective incoming weighting matrix
    ///     for the layer and the
    ///     vector the output layer. if the matrix is null the layer is equal to the input layer and there
    ///     for the parsed
    ///     image.
    /// </summary>
    private static (Vector, Matrix?)[] _layers;

    private readonly int _hiddenLayers;

    public NetworkModel() : this(1)
    {
    }
    public NetworkModel(int hiddenLayers, bool compressed = false)
    {
        _hiddenLayers = hiddenLayers;
    }

    private void Setup(int[]? layers = null, bool compress = false, bool randomStart = true)
    {
        var prev = compress ? 196 : 784;
        var layerCount = _hiddenLayers + 1;
        _layers = new (Vector, Matrix?)[layerCount];
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
            if (layers.Length != _hiddenLayers)
            {
                Setup();
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

    private static Vector FromBytes(ref byte[] b)
    {
        var v = new Vector(b.Length);
        for (var i = 0; i < b.Length; i++) v[i] = b[i];

        return v;
    }

    /// <summary>
    ///     Begin training of the network model.
    /// </summary>
    /// <param name="opt">Options for network training</param>
    public void Train(NeuralNetworkTrainingOptions opt)
    {
        Setup(opt.Layers?.Length > 0 ? opt.Layers : null, opt.Convolution);
        Random rand = new();
        Stopwatch sw = new();
        for (var i = 0; i < opt.Epochs; i++)
        {
            sw.Start();
            Console.ForegroundColor = ConsoleColor.Green;

            for (var j = 0; j < opt.EpochSize; j++)
            {
                var inputFeatures = MnistReader.TrainImage(j);
                var res = ProcessLayers(opt.Convolution ? Convolution.CompressFeatures(ref inputFeatures) : FromBytes(ref inputFeatures));
                var error = AssessError(MnistReader.TrainLabel(j), res);
                PropagetError(error, opt.LearningRate);
                Progress.PrintProgress(j + 1, opt.EpochSize, sw);
            }

            sw.Reset();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Assess(ref sw, opt.Convolution);
            MnistReader.Shuffle();

            // adjust learning rate if needed
            switch (opt.TrainingRateOptions)
            {
                case TrainingRateOptions.Constant: break;
                case TrainingRateOptions.Logarithmic: opt.LearningRate = opt.LearningRate / (1 + i); break;
                case TrainingRateOptions.Linear: opt.LearningRate = opt.LearningRate * (1 - (float)i / opt.Epochs); break;
                default: throw new ArgumentOutOfRangeException();
            }

        }
    }

    private void PropagetError(Vector error, float learningrate)
    {
        var err = new Vector[_layers.Length - 1];
        err[^1] = error;
        for (var i = _layers.Length - 2; i > 0; i--) err[i - 1] = _layers[i + 1].Item2!.T * err[i];
        for (var i = _layers.Length - 1; i > 0; i--)
        {
            var outp = _layers[i].Item1;
            var next = _layers[i - 1].Item1;
            // update matrix. matrix should never be null at this point
            var diff = learningrate * (err[i - 1] * outp * (1 - outp) * next.T());
            _layers[i].Item2 += diff;
        }
    }

    private Vector ProcessLayers(Vector v)
    {
        NormalizeBytes(ref v);
        _layers[0] = (v, null);
        for (var i = 1; i < _layers.Length; i++)
            // item 1 is the layer vector while item 2 refers to the weight matrix.
        {
            var matrix = _layers[i].Item2;
            if (matrix != null) _layers[i].Item1 = (matrix * _layers[i - 1].Item1).Activate();
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
        Assess(ref sw, convolutionActive);
    }

    private void Assess(ref Stopwatch sw, bool compress = false, int iterations = 1000)
    {
        Random rand = new();
        var hits = 0;
        sw.Start();
        for (var i = 0; i < iterations; i++)
        {
            var n = rand.Next(10_000);
            var inputFeatures = MnistReader.Image(n);
            var res = ProcessLayers(compress ? Convolution.CompressFeatures(ref inputFeatures) : FromBytes(ref inputFeatures));
            if (res.Max() == MnistReader.Label(n)) hits++;
            Progress.PrintProgress(i + 1, iterations, sw, hits);
        }
        Console.WriteLine();
    }
    private void NormalizeBytes(ref Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] /= 256.0;
    }


    private Vector Predict(ref byte[] inputFeatures, bool compress = false) => ProcessLayers(compress ? Convolution.CompressFeatures(ref inputFeatures) : FromBytes(ref inputFeatures));

    public int PredictLabel(ref byte[] inputFeatures, bool compress = false)
    {
        var res = Predict(ref inputFeatures, compress);

        return res.Max();
    }
}

public enum TrainingRateOptions {
    Constant,
    Logarithmic,
    Linear
}

public class NeuralNetworkTrainingOptions {
    public int Epochs { get; set; } = 5;
    public int EpochSize { get; set; } = 1750;
    public float LearningRate { get; set; } = .1f;
    public TrainingRateOptions TrainingRateOptions { get; set; } = TrainingRateOptions.Constant;
    public bool Convolution { get; set; } = false;
    public int[] Layers { get; set; } = [256];
    public static NeuralNetworkTrainingOptions Default => new();
}