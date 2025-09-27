using MathLib.Linalg;

namespace NeuronalesNetz.algo;

public static class Convolution {


    /// <summary>
    ///     Compresses the input features from 784 to 196 using a simple 2x2 average pooling method.
    /// </summary>
    /// <param name="inputFeatures">the input features</param>
    /// <returns>compressed features as an array of size 196</returns>
    /// <exception cref="Exception"></exception>
    public static Vector CompressFeatures(byte[] inputFeatures)
    {
        if (inputFeatures.Length != 784) throw new Exception("Input features must be 784 to apply convolution.");

        var tmp = new double[196];

        for (var i = 0; i < 28; i += 2)
        for (var j = 0; j < 28; j += 2)
        {
            var xy = i * 28 + j; // start pixel
            var x = i * 28 + j + 1;
            var y = (i + 1) * 28 + j;

            var medx = (inputFeatures[xy] + inputFeatures[x]) / 2.0;
            var medy = (inputFeatures[xy] + inputFeatures[y]) / 2.0;

            tmp[i / 2 * 14 + j / 2] = (medx + medy) / 2.0;
        }

        return new Vector(tmp);
    }
}