using System.Collections.Concurrent;

namespace IO;

public class MnistReader {
    private const int LabelOffset = 8;

    // todo replace paths with urls and call http request on setup.
    private readonly static string MnistTrainImages = "train-images.idx3-ubyte";
    private readonly static string MnistImages = "t10k-images.idx3-ubyte";
    private readonly static string MnistTrainLabels = "train-labels.idx1-ubyte";
    private readonly static string MnistLabels = "t10k-labels.idx1-ubyte";
    private readonly ConcurrentDictionary<int, byte[]> _images;
    private readonly byte[] _labels;
    private readonly ConcurrentDictionary<int, byte[]> _trainImages;
    private readonly byte[] _trainLabels;

    public MnistReader()
    {
        // initialize training images
        var offset = 16;
        var size = 784;

        _trainLabels = GetData(MnistTrainLabels);
        _labels = GetData(MnistLabels);
        var data = GetData(MnistTrainImages);
        var assesData = GetData(MnistImages);
        _trainImages = new ConcurrentDictionary<int, byte[]>();
        _images = new ConcurrentDictionary<int, byte[]>();
        List<Task> tasks = new();
        for (var split = 0; split < 6; split++)
        {
            var s = split;
            tasks.Add(Task.Run(() => {
                for (var i = 0; i < 10_000; i++)
                {
                    var key = s * 10000 + i;
                    _trainImages.TryAdd(key, Splice(ref data, offset + key * size, size));
                }
            }));
        }
        tasks.Add(Task.Run(() => {
            for (var i = 0; i < 1e4; i++) _images.TryAdd(i, Splice(ref assesData, offset + i * size, size));
        }));

        Task.WaitAll(tasks.ToArray());

    }

    private byte[] GetData(string filename)
    {
        try
        {
            return File.ReadAllBytes($"Mnist.Numbers/{filename}");
        }
        catch
        {
            // Data is missing so download it
            if (!Directory.Exists("Mnist.Numbers")) Directory.CreateDirectory("Mnist.Numbers");
            Task.Run(async () => await File.WriteAllBytesAsync($"Mnist.Numbers/{filename}", await GetDataAsync(filename))).Wait();

            return File.ReadAllBytes($"Mnist.Numbers/{filename}");
        }
    }

    public ImageData[] TrainImages(int i, int batchSize)
    {
        var images = new ImageData[batchSize];

        for (var k = 0; k < batchSize; k++) images[k] = new ImageData(_trainImages[i + k], _trainLabels[i + k]);

        return images;
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

    public void Shuffle()
    {
        var rand = new Random();
        // shuffle training data to avoid overfitting
        for (var i = 0; i < 60_000; i++)
        {
            var n = rand.Next(60_000);
            n = n == i ? rand.Next(60_000) : n;
            (_trainImages[i], _trainImages[n]) = (_trainImages[n], _trainImages[i]);
            (_trainLabels[8 + i], _trainLabels[8 + n]) = (_trainLabels[8 + n], _trainLabels[8 + i]);
        }
        // shuffle test data too to avoid overfitting
        for (var i = 0; i < 10_000; i++)
        {
            var n = rand.Next(10_000);
            (_images[i], _images[n]) = (_images[n], _images[i]);
            (_labels[8 + i], _labels[8 + n]) = (_labels[8 + n], _labels[8 + i]);
        }
    }

    public ImageData Image()
    {

        var rnd = Random.Shared;
        var n = rnd.Next(10_000);

        return new ImageData(_images[n], _labels[n]);
    }


    public record ImageData(byte[] Image, byte Label);
}