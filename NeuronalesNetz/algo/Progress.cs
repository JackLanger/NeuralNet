using System.Diagnostics;
using System.Text;
using MathTools;

namespace MachineLearn.extension;

public static class Progress
{
    public static void PrintProgress(int n, int total, Stopwatch sw, int hits = -1)
    {
        StringBuilder bars = new();
        var barsProg = (double)(n * 20) / total;
        for (var i = 0; i < 20; i++) bars.Append(i <= barsProg ? '=' : ' ');
        Console.Write(hits >= 0
            ? $"ASSESSING GEN : {n:0000}/{total}:[{bars}] {(double)hits / n:P} {sw.Elapsed:g}\r"
            : $"TRAINING GEN: {n:0000}/{total}:[{bars}] {sw.Elapsed:g}\r");
    }
    
}