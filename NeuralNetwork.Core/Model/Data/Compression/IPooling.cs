using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

/// <summary>
///     Pooling interface for reducing the dimensionality of input vectors.
/// </summary>
public interface IPooling {

    /// <summary>
    ///     Pools the input vector, reducing its dimensionality.
    /// </summary>
    /// <param name="input"></param>
    /// <returns></returns>
    public Vector Pool(Vector input);
}