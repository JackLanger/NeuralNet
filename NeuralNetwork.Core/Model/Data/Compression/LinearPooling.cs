using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class LinearPooling : IPooling {

    public Vector Pool(Vector input)
    {
        Vector v = new(input.Length / 4);

        var index = 0;
        for (var i = 0; i < input.Length; i += 4)
        {
            switch ((input.Length - i) % 4)
            {
                // If there are leftover elements, handle them
                case 1:
                    v[index++] = input[i];
                    i++;

                    break;
                case 2:
                    v[index++] = (input[i] + input[i + 1]) / 2.0;
                    i += 2;

                    break;
                case 3:
                    v[index++] = (input[i] + input[i + 1] + input[i + 2]) / 3.0;
                    i += 3;

                    break;
            }
            // always process a block of 4 if possible
            if (i < input.Length)
            {
                v[index++] = (input[i] + input[i + 1] + input[i + 2] + input[i + 3]) / 4.0;
            }
        }

        return v;
    }
}