namespace IO;

public class MnistReader
{
    // todo replace paths with urls and call http request on setup.
    private static readonly string MnistTrainImages = "train-images.idx3-ubyte";
    private static readonly string MnistTrainLabels = "train-labels.idx1-ubyte";
    private static readonly string MnistImages = "t10k-images.idx3-ubyte";
    private static readonly string MnistLabels = "t10k-labels.idx1-ubyte";
    private static byte[]? _trainImages;
    private static byte[]? _trainlabels;
    private static byte[]? _images;
    private static byte[]? _labels;

    public static byte[] TrainImage(int n)
    {
#if DEBUG
        _trainImages ??= File.ReadAllBytes(@$"D:\OSZIMT\Lehmann\NeuronalesNetz\NeuronalesNetz\data\{MnistTrainImages}");
#endif
        _trainImages ??= GetDataAsync(MnistTrainImages).Result;
        var offset = 16;
        var size = 784;

        return Splice(ref _trainImages, offset + n * size, size);
    }

    private static async Task<byte[]> GetDataAsync(string file)
    {
        
        var url =
            $"https://github.com/JackLanger/NeuralNet/raw/master/NeuronalesNetz/data/{file}";

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
        
#if DEBUG
        _trainImages ??= File.ReadAllBytes(@$"D:\OSZIMT\Lehmann\NeuronalesNetz\NeuronalesNetz\data\{MnistTrainLabels}");
#endif
        _trainlabels ??= GetDataAsync(MnistTrainLabels).Result;
        var offset = 8;
        return _trainlabels[offset + n];
    }

    public static byte[] Image(int n)
    {
        
#if DEBUG
        _trainImages ??= File.ReadAllBytes(@$"D:\OSZIMT\Lehmann\NeuronalesNetz\NeuronalesNetz\data\{MnistImages}");
#endif
        _images ??= GetDataAsync(MnistImages).Result;
        var offset = 16;
        var size = 784;

        return Splice(ref _images, offset + n * size, size);
    }

    public static byte Label(int n)
    {
        
#if DEBUG
        _trainImages ??= File.ReadAllBytes(@$"D:\OSZIMT\Lehmann\NeuronalesNetz\NeuronalesNetz\data\{MnistLabels}");
#endif
        _labels ??= GetDataAsync(MnistLabels).Result;
        var offset = 8;
        return _labels[offset + n];
    }
}