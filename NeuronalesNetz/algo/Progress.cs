using System.Diagnostics;
using System.Text;

namespace MachineLearn.extension;

public static class Progress
{
    public static void PrintProgress(int n, int total, Stopwatch sw, int hits = -1)
    {
        StringBuilder bars = new();
        var barsProg = (double)(n * 20) / total;
        for (var i = 0; i < 20; i++)
            if (i <= barsProg)
                bars.Append("=");
            else
                bars.Append(" ");
        if (hits >= 0)
            Console.Write($"ASSESSING GEN : {n:0000}/{total}:[{bars}] {(double)hits / n:P} {sw.Elapsed}\r");
        else
            Console.Write($"TRAINING GEN: {n:0000}/{total}:[{bars}] {sw.Elapsed}\r");
    }
}