using MathLib.Linalg;

namespace NeuronalesNetz.extension;

public static class VectorExtensions {

    /// <summary>
    ///     Activate a given input using the sigmoid function.
    /// </summary>
    /// <param name="x">the sum of all inputs respective to the input.</param>
    /// <returns>the activated value for the input</returns>
    private static double ActivateSig(double x) => 1 / (1 + Math.Pow(Math.E, -x));

    public static Vector Activate(this Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] = ActivateSig(v[i]);

        return v;
    }
}