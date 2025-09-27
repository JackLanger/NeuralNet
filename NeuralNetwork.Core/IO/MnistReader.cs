namespace IO;

public class MnistReader {
    // todo replace paths with urls and call http request on setup.
    private readonly static string MnistTrainImages = "train-images.idx3-ubyte";
    private readonly static string MnistTrainLabels = "train-labels.idx1-ubyte";
    private readonly static string MnistImages = "t10k-images.idx3-ubyte";
    private readonly static string MnistLabels = "t10k-labels.idx1-ubyte";
    private static Dictionary<int, byte[]>? _trainImages;
    private static Dictionary<int, byte[]>? _images;
    private static byte[]? _trainlabels;
    private static byte[]? _labels;

    public static byte[] TrainImage(int n)
    {
        if (_trainImages is null)
        {
            var offset = 16;
            var size = 784;
            _trainImages = new Dictionary<int, byte[]>();
            var data = GetDataAsync(MnistTrainImages).Result;
            for (var i = 0; i < 6e4; i++) _trainImages.Add(i, Splice(ref data, offset + i * size, size));

        }

        return _trainImages[n];
    }

    private async static Task<byte[]> GetDataAsync(string file)
    {
        var url =
            $"https://github.com/JackLanger/NeuralNet/raw/main/Resources/{file}";

        using HttpClient client = new();

        return await client.GetByteArrayAsync(url);
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
        _trainlabels ??= GetDataAsync(MnistTrainLabels).Result;
        var offset = 8;

        return _trainlabels[offset + n];
    }

    public static byte[] Image(int n)
    {
        if (_images is null)
        {
            var offset = 16;
            var size = 784;
            _images = new Dictionary<int, byte[]>();
            var data = GetDataAsync(MnistImages).Result;
            for (var i = 0; i < 1e4; i++) _images.Add(i, Splice(ref data, offset + i * size, size));
        }

        return _images[n];
    }

    public static byte Label(int n)
    {
        _labels ??= GetDataAsync(MnistLabels).Result;
        var offset = 8;

        return _labels[offset + n];
    }


    public static void Shuffle()
    {
        var rand = new Random();
        for (var i = 0; i < 6e4; i++)
        {
            var n = rand.Next((int)6e4);
            n = n == i ? rand.Next((int)6e4) : n;
            (_trainImages[i], _trainImages[n]) = (_trainImages[n], _trainImages[i]);
            (_trainlabels[8 + i], _trainlabels[8 + n]) = (_trainlabels[8 + n], _trainlabels[8 + i]);
        }
    }
}