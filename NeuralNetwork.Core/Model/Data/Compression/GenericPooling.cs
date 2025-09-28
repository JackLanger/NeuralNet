using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

public class GenericPooling(Func<Matrix, Matrix> poolingFunction) : IPooling {

    public Matrix Pool(Matrix input) => poolingFunction(input);

}