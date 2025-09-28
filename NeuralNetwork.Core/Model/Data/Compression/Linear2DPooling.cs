using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class Linear2DPooling(int width) : IPooling {

    /// <summary>
    ///     Compresses the input features using a simple 2x2 average pooling method. The input is
    ///     interpreted as a 2D feature map of shape (height, width), and the output is a pooled feature
    ///     map of shape (height/2, width/2).
    /// </summary>
    /// <param name="input">the input features as a flat vector representing a 2D feature map</param>
    /// <returns>compressed features as a vector of size (height/2) * (width/2)</returns>
    /// <exception cref="Exception"></exception>
    public Vector Pool(Vector input)
    {
        var height = input.Length / width;
        var pooledHeight = height / 2;
        var pooledWidth = width / 2;
        Vector v = new(pooledHeight * pooledWidth);

        var idx = 0;

        for (var i = 0; i < height; i += 2)
        {

            if (i + 1 >= height)
            {
                for (var k = i * width; k < input.Length; k++)
                    v[idx++] = input[k];

                break;
            }

            for (var j = 0; j < width; j += 2)
            {

                if (j + 1 >= width)
                {
                    // Not enough columns left, copy remaining element(s) as-is
                    v[idx++] = input[i * width + j];
                    v[idx++] = input[(i + 1) * width + j];

                    continue;
                }

                var sum = input[i * width + j] + input[i * width + j + 1] + input[(i + 1) * width + j] +
                    input[(i + 1) * width + j + 1];
                v[idx++] = sum / 4.0;
            }

        }

        return v;
    }
}