using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

public class GenericPooling(Func<Vector, Vector> poolingFunction) : IPooling {

    public Vector Pool(Vector input) => poolingFunction(input);
}