using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class Linear2DPooling(int width) : IPooling {

    /// <summary>
    ///     Compresses the input features from 784 to 196 using a simple 2x2 average pooling method.
    /// </summary>
    /// <param name="input">the input features</param>
    /// <returns>compressed features as an array of size 196</returns>
    /// <exception cref="Exception"></exception>
    public Vector Pool(Vector input)
    {
        var height = input.Length / width;
        Vector v = new(height / 2 * (width / 2));
        var idx = 0;
        for (var i = 0; i < height; i += 2)
        {
            if (input.Length - i < 4)
            {
                for (var k = i; k < input.Length; k++)
                    v[idx++] = input[k];

                break;
            }

            for (var j = 0; j < width; j += 2)
            {
                var sum = input[i * width + j] + input[i * width + j + 1] + input[(i + 1) * width + j] +
                    input[(i + 1) * width + j + 1];
                v[idx++] = sum / 4.0;
            }
        }

        return v;
    }
}