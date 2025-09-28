using MathLib.Linalg;

namespace NeuralNetworkLib.Model.Data.Compression;

internal class ParabolicFactorPooling : IPooling {

    public Vector Pool(Vector input)
    {
        Vector v = new(input.Length / 3);
        var index = 0;
        for (var i = 0; i <= input.Length - 3; i += 3)
            if (input.Length - i < 3)
            {
                for (var k = i; k < input.Length; k++)
                    v[index++] = input[k];
            }
            else
            {
                // Fit parabola to points (0, y0), (1, y1), (2, y2)
                var y0 = input[i];
                var y1 = input[i + 1];
                var y2 = input[i + 2];
                var a = FitParabolaA(y0, y1, y2);
                v[index++] = a;
            }

        return v;
    }

    // Helper method to fit parabola y = a x^2 + b x + c to three points (0, y0), (1, y1), (2, y2)
    private static double FitParabolaA(double y0, double y1, double y2) =>
        // System of equations:
        // y0 = c
        // y1 = a*1^2 + b*1 + c = a + b + c
        // y2 = a*4 + b*2 + c
        // Solve for a:
        // a = (y2 - 2*y1 + y0) / 2
        (y2 - 2 * y1 + y0) / 2.0;
}