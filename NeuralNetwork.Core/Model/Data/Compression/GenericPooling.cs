using MathLib.Linalg;

namespace NeuralNetwork.Core.Model.Data.Compression;

public class GenericPooling(Func<Matrix, Matrix> poolingFunction) : IPooling {

    public Matrix Pool(Matrix input) => poolingFunction(input);
}