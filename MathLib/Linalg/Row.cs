namespace MathLib.Linalg;

internal class Row {
    public Row(params double[] data)
    {
        this.data = data;
    }

    public double[] data { get; set; }

    public int Length => data.Length;

    public double this[int i]
    {
        get => data[i];
        set => data[i] = value;
    }

    public static Row operator *(Row fst, double alph)
    {
        var numArray = new double[fst.Length];
        for (var i = 0; i < fst.Length; ++i)
            numArray[i] = fst[i] * alph;

        return new Row(numArray);
    }

    public static Row operator /(double alph, Row fst) => 1.0 / alph * fst;

    public static Row operator /(Row fst, double alph) => 1.0 / alph * fst;

    public static Row operator *(double alph, Row fst) => fst * alph;

    public static Row operator -(Row fst, Row snd) => fst + snd * -1.0;

    public static Row operator +(Row fst, Row snd)
    {
        if (fst.Length != snd.Length)
        {
            throw new InvalidOperationException("Rows of different size cannot be added");
        }
        var numArray = new double[fst.Length];
        for (var i = 0; i < fst.Length; ++i)
            numArray[i] = fst[i] + snd[i];

        return new Row(numArray);
    }
}