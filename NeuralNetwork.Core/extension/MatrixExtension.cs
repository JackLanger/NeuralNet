using MathLib.Linalg;

namespace NeuralNetwork.Core.extension;

public static class MatrixExtension {
    /// <summary>
    ///     Sets all the fields of the matrix to a given value val.
    /// </summary>
    /// <param name="m">The matrix</param>
    /// <param name="val">value to set for all values of the matrix backing field</param>
    /// <returns>the matrix with updated values.</returns>
    public static Matrix WithValue(this Matrix m, double val = .1)
    {
        for (var i = 0; i < m.Rows; i++)
        for (var j = 0; j < m.Cols; j++)
            m[i, j] = val;

        return m;
    }

    /// <summary>
    ///     Sets all the fields of the matrix to a random value.
    /// </summary>
    /// <param name="m">the matrix</param>
    /// <returns>Matrix with updated backing field.</returns>
    public static Matrix WithRandom(this Matrix m)
    {
        var rand = new Random();
        for (var i = 0; i < m.Rows; i++)
        for (var j = 0; j < m.Cols; j++)
            m[i, j] = rand.NextDouble();

        return m;
    }
}