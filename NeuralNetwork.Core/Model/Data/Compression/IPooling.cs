using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

/// <summary>
///     Pooling interface for reducing the dimensionality of input vectors.
/// </summary>
public interface IPooling {

    /// <summary>
    ///     Pools the input vector, reducing its dimensionality.
    /// </summary>
    /// <param name="input">The input vector to be pooled, typically representing features or activations.</param>
    /// <returns>The pooled result vector with reduced dimensionality.</returns>
    public Vector Pool(Vector input);
}