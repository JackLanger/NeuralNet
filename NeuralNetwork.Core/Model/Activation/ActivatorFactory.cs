using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Activation;

internal static class ActivatorFactory {

    public static IActivatorFunction Get(ActivatorFunctions activator)
        => activator switch
        {
            ActivatorFunctions.Sigmoid => new Sigmoid(),
            ActivatorFunctions.ReLU => new ReLU(),
            ActivatorFunctions.LeakyReLU => new LeakyReLU(),
            ActivatorFunctions.Tanh => new Tanh(),
            _ => throw new NotImplementedException($"Activator function {activator} is not implemented.")
        };
}

internal class Tanh(int batchsize = 1) : IActivatorFunction {
    public Matrix Activate(Matrix z)
    {
        double ActivateTanh(double x) => (Math.Pow(Math.E, x) - Math.Pow(Math.E, -x)) / (Math.Pow(Math.E, x) + Math.Pow(Math.E, -x));
        for (var k = 0; k < batchsize; k++)
        for (var i = 0; i < z[k].Length; i++)
            z[k][i] = ActivateTanh(z[k][i]);

        return z;
    }
    public Matrix Derivative(Matrix inputLayer, Matrix activatedNextLayer, Matrix errorVector)
    {

        {
            Matrix result = new(batchsize, inputLayer[0].Length);

            result += errorVector.Hadamard(1 - activatedNextLayer.Hadamard(activatedNextLayer)) * inputLayer.T;

            return result;
        }
    }
}

internal class LeakyReLU(int batchsize = 1) : IActivatorFunction {
    public Matrix Activate(Matrix z)
    {
        double ActivateLeakyReLu(double x) => x > 0 ? x : x * 0.01;

        for (var k = 0; k < batchsize; k++)
        for (var i = 0; i < z[k].Length; i++)
            z[k][i] = ActivateLeakyReLu(z[k][i]);

        return z;
    }

    public Matrix Derivative(Matrix inputLayer, Matrix activatedNextLayer, Matrix errorVector)
    {
        Matrix result = new(activatedNextLayer.Cols, inputLayer.Cols);

        var left = errorVector.Hadamard(activatedNextLayer.Map(x => x > 0 ? 1 : 0.01));
        result += left.T * inputLayer;

        return result;
    }
}

internal class ReLU(int batchsize = 1) : IActivatorFunction {
    public Matrix Activate(Matrix z)
    {

        double ActivateReLu(double x) => x > 0 ? x : 0;
        for (var k = 0; k < batchsize; k++)
        for (var i = 0; i < z[k].Length; i++)
            z[k][i] = ActivateReLu(z[k][i]);

        return z;
    }
    public Matrix Derivative(Matrix inputLayer, Matrix activatedNextLayer, Matrix errorVector)
    {
        Matrix result = new(batchsize, inputLayer[0].Length);
        for (var k = 0; k < batchsize; k++)
            result += errorVector[k].Hadamard(activatedNextLayer[k].Map(x => x > 0 ? 1 : 0.0)) * inputLayer.T;

        return result;
    }
}

internal class Sigmoid(int batchsize = 1) : IActivatorFunction {


    public Matrix Activate(Matrix v)
    {
        double ActivateSig(double x) => 1 / (1 + Math.Pow(Math.E, -x));
        for (var batch = 0; batch < batchsize; batch++)
        for (var i = 0; i < v[batch].Length; i++)
            v[batch][i] = ActivateSig(v[batch][i]);

        return v;
    }

    public Matrix Derivative(Matrix inputLayer, Matrix activatedNextLayer, Matrix errorVector)
    {
        Matrix result = new(batchsize, inputLayer[0].Length);
        for (var k = 0; k < batchsize; k++)
            result += errorVector[k].Hadamard(activatedNextLayer[k].Hadamard(1 - activatedNextLayer[k])) * inputLayer.T;

        return result;
    }
}