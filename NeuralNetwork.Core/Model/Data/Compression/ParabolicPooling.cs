using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class ParabolicPooling : IPooling {

    public Vector Pool(Vector input)
    {
        Vector v = new(input.Length / 4);

        for (var i = 0; i < input.Length; i += 4)
        {
            var avg = (input[i] + input[i + 1] + input[i + 2] + input[i + 3]) / 4.0;
            v[i / 4] = avg * avg;
        }

        return v;
    }
}