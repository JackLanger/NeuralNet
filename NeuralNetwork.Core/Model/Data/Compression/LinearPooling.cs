using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class LinearPooling : IPooling {

    public Vector Pool(Vector input)
    {
        Vector v = new(input.Length / 4);

        var index = 0;
        var i = 0;
        for (; i < input.Length; i += 4)
            // always process a block of 4 if possible
            if (i < input.Length)
            {
                v[index++] = (input[i] + input[i + 1] + input[i + 2] + input[i + 3]) / 4.0;
            }

        if (i < input.Length)
        {
            var leftover = input.Length - i;
            v[index] = leftover switch
            {
                // If there are leftover elements, handle them
                1 => input[i],
                2 => (input[i] + input[i + 1]) / 2.0,
                3 => (input[i] + input[i + 1] + input[i + 2]) / 3.0,
                _ => throw new ArgumentOutOfRangeException("Something went wrong in LinearPooling, there should be 0-3 leftover elements but got " + leftover)
            };
        }

        return v;
    }
}