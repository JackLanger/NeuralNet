using System;
using MathTools;

namespace NeuronalesNetz.extension;

public static class VectorExtensions
{
    public static Vector Activate(this Vector v)
    {
        for (var i = 0; i < v.Length; i++) v[i] = ActivateSig(v[i]);

        return v;
    }

    /// <summary>
    ///     Activate a given input using the sigmoid function.
    /// </summary>
    /// <param name="x">the sum of all inputs respective to the input.</param>
    /// <returns>the activated value for the input</returns>
    private static double ActivateSig(double x)
    {
        return 1 / (1 + Math.Pow(Math.E, -x));
    }


    public static int Max(this Vector v)
    {
        double max = long.MinValue;
        var index = -1;

        for (var i = 0; i < v.Length; i++)
            if (v[i] > max)
            {
                max = v[i];
                index = i;
            }

        return index;
    }
    
}