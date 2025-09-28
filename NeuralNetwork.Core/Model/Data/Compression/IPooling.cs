using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

/// <summary>
///     Pooling interface for reducing the dimensionality of input vectors.
/// </summary>
public interface IPooling {


    /// <summary>
    ///     Pools the input matrix, reducing its dimensionality.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public Matrix Pool(Matrix input);
}