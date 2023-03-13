using System.Diagnostics;
using IO;
using MachineLearn.extension;
using MathTools;
using NeuronalesNetz.extension;

namespace NeuronalesNetz.algo;

public class NetworkModel
{

    private int ImgSize = 784;
    /// <summary>
    ///     Tuple array of Vector and Matrix where the matrix is the respective incoming weighting matrix for the layer and the
    ///     vector the output layer. if the matrix is null the layer is equal to the input layer and there for the parsed
    ///     image.
    /// </summary>
    private static (Vector, Matrix?)[] _layers;

    private int _hiddenLayers;

    public NetworkModel():this(1)
    {
        
    }

    public NetworkModel(int hlayers)
    {
        _hiddenLayers = hlayers;
        
    }

    private void Setup(bool randomStart = true)
    {
        //todo reassess this setup function, it seems overly complicated.
        int layerCount = _hiddenLayers + 2;
        _layers = new (Vector, Matrix?)[layerCount];
        _layers[0] = (new Vector(784), null);
        var prev = 784;
        for (var i = 1; i < layerCount; i++)
        {
            prev = _layers[i - 1].Item1.Length;
            var n = (int)Math.Sqrt(prev * 10)+10;
            _layers[i] = randomStart 
                ? (new Vector(n), new Matrix(n, prev).WithRandom()*.1)
                : (new Vector(n), new Matrix(n, prev).WithValue(.05));
        }

        _layers[^1] = (new Vector(10), new Matrix(10, _layers[^2].Item1.Length).WithRandom()*.1);
    }

    public void Train(int gens = 5, int gensize = 1750, float learningrate = .1f)
    {
        Setup();
        Random rand = new();
        Stopwatch sw = new();
        
        for (var i = 0; i < gens; i++)
        {
            sw.Start();
            Console.ForegroundColor = ConsoleColor.Green;
            for (var j = 0; j < gensize; j++)
            {
                var n = rand.Next((int)6e4);
                var res = ProcessLayers(new Vector(ImgSize).FromBytes(MnistReader.TrainImage(n)));
                var error = AssessError(MnistReader.TrainLabel(n), res);
                PropagetError(error, learningrate);
                Progress.PrintProgress(j+1, gensize, sw);
            }
            sw.Reset();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Assess(ref sw);
        }
    }

    private void PropagetError(Vector error, float learningrate)
    {
        var err = new Vector[_layers.Length - 1];
        err[^1] = error;

        for (var i = _layers.Length-2; i > 0 ; i--) err[i-1] = _layers[i+1].Item2!.T * err[i];

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
            if (matrix != null)
                _layers[i].Item1 = (matrix * _layers[i - 1].Item1).Activate();
        }

        return _layers[^1].Item1;
    }

    private Vector AssessError(int lable, Vector output)
    {
        var tmp = new Vector(output.Length);
        for (var i = 0; i < output.Length; i++)
        {
            var target = lable == i ? 1 : 0;
            tmp[i] = (target - output[i]) ;
        }

        return tmp;
    }

    private void Assess(ref Stopwatch sw,int iterations = 1000)
    {
        Random rand = new();
        var hits = 0;
        sw.Start();
        for (var i = 0; i < iterations; i++)
        {
            var n = rand.Next((int)1e4);
            var res = ProcessLayers(new Vector(ImgSize).FromBytes(MnistReader.Image(n)));
            if (res.Max() == MnistReader.Label(n)) hits++;

            Progress.PrintProgress(i+1, iterations, sw, hits);
        }
        Console.WriteLine();
    }



    private void NormalizeBytes(ref Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] /= 256.0;
    }
}