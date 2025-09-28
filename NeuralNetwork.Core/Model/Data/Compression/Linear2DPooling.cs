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
        var v = ComputeOutputVector(height);

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
                if (j + 1 < width)
                {
                    var sum = input[i * width + j] + input[i * width + j + 1] + input[(i + 1) * width + j] +
                        input[(i + 1) * width + j + 1];
                    v[idx++] = sum / 4.0;
                }
                else
                {
                    // Handle the last column when width is odd
                    var sum = input[i * width + j] + input[(i + 1) * width + j];
                    v[idx++] = sum / 2.0;
                }

        }

        return v;
    }
    private Vector ComputeOutputVector(int height)
    {

        var pooledHeight = height / 2;
        var pooledWidth = width / 2;
        var outputSize = pooledHeight * pooledWidth;
        if (height % 2 != 0)
        {
            outputSize += width; // leftover row
        }
        if (width % 2 != 0)
        {
            outputSize += pooledHeight; // leftover column for each 2-row block
        }

        return new Vector(outputSize);

    }
}