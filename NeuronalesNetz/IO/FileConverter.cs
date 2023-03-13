namespace IO;

public static class FileReader
{
    /// <summary>
    ///     Returns the contents of the File at the path provided. If an action is provided the action will be performed
    ///     against the Contents of the file. This can be used to eliminate comment lines or other metadata to extract the
    ///     actual data and or the meta data by itself.
    /// </summary>
    /// <param name="path">Path to file</param>
    /// <param name="action">Optional action to perform against the contents of the file</param>
    /// <returns>The file contents split by newlines, as an string array.</returns>
    public static string[]? GetFileContent(string path, Action<string[]>? action = null)
    {
        string[]? contents = null;
        try
        {
            contents = File.ReadAllLines(path);
            action?.Invoke(contents);
        }
        catch (IOException e)
        {
            Console.WriteLine(e.Message);
        }

        return contents;
    }
}