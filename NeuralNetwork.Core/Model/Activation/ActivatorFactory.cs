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

internal class Tanh : IActivatorFunction {
    public Vector Activate(Vector z)
    {
        double ActivateTanh(double x) => (Math.Pow(Math.E, x) - Math.Pow(Math.E, -x)) / (Math.Pow(Math.E, x) + Math.Pow(Math.E, -x));

        for (var i = 0; i < z.Length; i++) z[i] = ActivateTanh(z[i]);

        return z;
    }
    public Matrix Derivative(Vector inputLayer, Vector activatedNextLayer, Vector errorVector)
        => errorVector.Hadamard(1 - activatedNextLayer.Hadamard(activatedNextLayer)) * inputLayer.T;
}

internal class LeakyReLU : IActivatorFunction {
    public Vector Activate(Vector z)
    {
        double ActivateLeakyReLu(double x) => x > 0 ? x : x * 0.01;
        for (var i = 0; i < z.Length; i++) z[i] = ActivateLeakyReLu(z[i]);

        return z;
    }

    public Matrix Derivative(Vector inputLayer, Vector activatedNextLayer, Vector errorVector)
        => errorVector.Hadamard(activatedNextLayer.Map(x => x > 0 ? 1 : 0.01)) * inputLayer.T;
}

internal class ReLU : IActivatorFunction {
    public Vector Activate(Vector z)
    {

        double ActivateReLu(double x) => x > 0 ? x : 0;
        for (var i = 0; i < z.Length; i++) z[i] = ActivateReLu(z[i]);

        return z;
    }
    public Matrix Derivative(Vector inputLayer, Vector activatedNextLayer, Vector errorVector)
        => errorVector.Hadamard(activatedNextLayer.Map(x => x > 0 ? 1 : 0)) * inputLayer.T;
}

internal class Sigmoid : IActivatorFunction {
    public Vector Activate(Vector v)
    {
        double ActivateSig(double x) => 1 / (1 + Math.Pow(Math.E, -x));

        for (var i = 0; i < v.Length; i++) v[i] = ActivateSig(v[i]);

        return v;
    }

    public Matrix Derivative(Vector inputLayer, Vector activatedNextLayer, Vector errorVector)
        => errorVector.Hadamard(activatedNextLayer.Hadamard(1 - activatedNextLayer)) * inputLayer.T;
}