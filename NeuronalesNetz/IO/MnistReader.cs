namespace IO;

public class MnistReader
{
    // todo replace paths with urls and call http request on setup.
    private static readonly string MnistTrainImages =
        @"D:\libs\dotnet\MathTools\MnistAssertion\data\train-images.idx3-ubyte";

    private static readonly string MnistTrainLabels =
        @"D:\libs\dotnet\MathTools\MnistAssertion\data\train-labels.idx1-ubyte";

    private static readonly string MnistImages = @"D:\libs\dotnet\MathTools\MnistAssertion\data\t10k-images.idx3-ubyte";
    private static readonly string MnistLabels = @"D:\libs\dotnet\MathTools\MnistAssertion\data\t10k-labels.idx1-ubyte";

    private static byte[]? _trainImages;
    private static byte[]? _trainlabels;
    private static byte[]? _images;
    private static byte[]? _labels;

    public static byte[] TrainImage(int n)
    {
        _trainImages ??= File.ReadAllBytes(MnistTrainImages);
        var offset = 16;
        var size = 784;

        return Splice(ref _trainImages, offset + n * size, size);
    }

    private static byte[] Splice(ref byte[] bytes, int offset, int size)
    {
        var i = 0;
        var tmp = new byte[size];
        for (var j = offset; j < offset + size; j++) tmp[i++] = bytes[j];

        return tmp;
    }

    public static byte TrainLabel(int n)
    {
        _trainlabels ??= File.ReadAllBytes(MnistTrainLabels);
        var offset = 8;
        return _trainlabels[offset + n];
    }

    public static byte[] Image(int n)
    {
        _images ??= File.ReadAllBytes(MnistImages);
        var offset = 16;
        var size = 784;

        return Splice(ref _images, offset + n * size, size);
    }

    public static byte Label(int n)
    {
        _labels ??= File.ReadAllBytes(MnistLabels);
        var offset = 8;
        return _labels[offset + n];
    }
}