using System.Diagnostics;
using IO;
using MathTools;

namespace MachineLearn.extension;

public class NetworkModel
{
    /// <summary>
    ///     Tuple array of Vector and Matrix where the matrix is the respective incoming weighting matrix for the layer and the
    ///     vector the output layer. if the matrix is null the layer is equal to the input layer and there for the parsed
    ///     image.
    /// </summary>
    private static (Vector, Matrix?)[] _layers;


    private void setup(int hiddenLayers = 1, bool randomStart = true)
    {
        //todo reassess this setup function, it seems overly complicated.
        _layers = new (Vector, Matrix?)[hiddenLayers + 2];
        _layers[0] = (new Vector(784), null);
        var prev = 784;
        for (var i = 1; i < _layers.Length - 1; i++)
        {
            prev = _layers[i - 1].Item1.Length;
            var n = (int)Math.Sqrt(prev * 10);
            _layers[i] = (new Vector(n), new Matrix(n, prev).WithRandom());
        }

        _layers[_layers.Length - 1] = (new Vector(10), new Matrix(10, prev).WithRandom());
    }

    public void Train(int gens = 5, int gensize = 1750, float learningrate = .1f)
    {
        setup();
        Random rand = new();
        int n;
        Stopwatch sw = new();
        for (var i = 0; i < gens; i++)
        {
            for (var j = 0; j < gensize; j++)
            {
                n = rand.Next((int)6e4);
                var res = ProcessLayers(FromBytes(MnistReader.TrainImage(n)));
                var error = AssessError(MnistReader.TrainLabel(n), res);
                PropagetError(error, learningrate);
                Progress.PrintProgress(j, gensize, sw);
            }

            Assess();
        }
    }

    private void PropagetError(Vector error, float learningrate)
    {
        var err = new Vector[_layers.Length - 1];
        err[^1] = error;

        for (var i = _layers.Length - 2; i > 0; i--) err[i - 1] = _layers[i].Item2!.T * err[i];

        for (var i = _layers.Length - 1; i > 0; i--)
        {
            var outp = _layers[i].Item1;
            var next = _layers[i - 1].Item1;

            // update matrix. matrix should never be null at this point
            _layers[i].Item2 += learningrate * (err[i] * outp * (1 - outp) * next.T());
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
        int target;
        for (var i = 0; i < output.Length; i++)
        {
            target = lable == i ? 1 : 0;
            tmp[i] = (target - output[i]) * (target - output[i]);
        }

        return tmp;
    }

    private void Assess(int iterations = 1000)
    {
        Random rand = new();
        Stopwatch sw = new();
        var hits = 0;
        sw.Start();
        for (var i = 0; i < iterations; i++)
        {
            var n = rand.Next((int)1e4);
            var res = ProcessLayers(FromBytes(MnistReader.Image(n)));
            if (res.Max() == MnistReader.Label(n)) hits++;

            Progress.PrintProgress(i, iterations, sw, hits);
        }
    }


    private Vector FromBytes(byte[] b)
    {
        var tmp = new double[b.Length];

        for (var i = 0; i < b.Length; i++) tmp[i] = b[i];

        return new Vector(tmp);
    }

    private void NormalizeBytes(ref Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] /= 256.0;
    }
}