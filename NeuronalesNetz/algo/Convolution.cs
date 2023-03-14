using MathTools;

namespace NeuronalesNetz.algo;

public static class Convolution
{
    public static Vector Compress(ref Vector arr)
    {
        Vector tmp = new(14*14);
        for (int i = 0; i < 28; i+=2)
        {
            for (int n = 0; n < 28; n++)
            {
                tmp[(i / 2) * (n/2) +(n/2)] = (int)((arr[i] + arr[i + 1] + arr[i * n + n] + arr[i * n + n + 1]) / 4.0);
            }
        }

        return tmp;
    }
}