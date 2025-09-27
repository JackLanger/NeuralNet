namespace MathLib.Linalg;

public class Vector(double[] data) {
    private Matrix? _transpose;

    public Vector(int n)
        : this(new double[n])
    {
    }

    public int Length => data.Length;

    public double this[int i]
    {
        get => data[i];
        set => data[i] = value;
    }

    public static Matrix operator *(Vector v, Matrix m)
    {
        if (m.Rows > 1)
        {
            throw new ArgumentException($"Unable to multiply a vector from left to a matrix of size {m.Rows}x{m.Cols}");
        }
        var matrix = new Matrix(v.Length, m.Cols);
        for (var index = 0; index < v.Length; ++index)
        for (var col = 0; col < m.Cols; ++col)
            matrix[index, col] = v[index] * m[0, col];

        return matrix;
    }

    public static Vector operator -(Vector fst, Vector snd) => fst + -1.0 * snd;

    public static Vector operator -(Vector v, double n)
    {
        var vector = new Vector(v.Length);
        for (var i = 0; i < v.Length; ++i)
            vector[i] = v[i] - n;

        return vector;
    }

    public static Vector operator -(double n, Vector v)
    {
        var vector = new Vector(v.Length);
        for (var i = 0; i < v.Length; ++i)
            vector[i] = n - v[i];

        return vector;
    }

    public static Vector operator +(Vector fst, Vector snd)
    {
        if (fst.Length != snd.Length)
        {
            throw new ArgumentException("In order to perform this operation, both vectors need to be of the same size.");
        }
        var vector = new Vector(fst.Length);
        for (var i = 0; i < fst.Length; ++i)
            vector[i] = fst[i] + snd[i];

        return vector;
    }

    public static Vector operator /(double alpha, Vector v) => v * (1.0 / alpha);

    public static Vector operator /(Vector v, double alpha) => v * (1.0 / alpha);

    public static Vector operator *(double alpha, Vector v) => v * alpha;

    public static Vector operator *(Vector v, double alpha)
    {
        var vector = new Vector(v.Length);
        for (var i = 0; i < v.Length; ++i)
            vector[i] = v[i] * alpha;

        return vector;
    }

    public static Vector operator *(Vector a, Vector b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Operation invalid on vertices of different dimensions");
        }
        var vector = new Vector(a.Length);
        for (var i = 0; i < a.Length; ++i)
            vector[i] = a[i] * b[i];

        return vector;
    }

    public static Vector operator *(Matrix m, Vector v)
    {
        if (m.Cols != v.Length)
        {
            throw new ArgumentException($"Can not multiply matrix of size {m.Rows}x{m.Cols} with a Vector of size {v.Length}");
        }
        var vector = new Vector(m.Rows);
        for (var index1 = 0; index1 < m.Rows; ++index1)
        for (var index2 = 0; index2 < m.Cols; ++index2)
            vector[index1] += m[index1, index2] * v[index2];

        return vector;
    }

    public double Abs()
    {
        var d = 0.0;
        for (var index = 0; index < Length; ++index)
            d += data[index] * data[index];

        return Math.Sqrt(d);
    }

    public Matrix T()
    {
        if (_transpose != null)
        {
            return _transpose;
        }
        _transpose = new Matrix(1, Length);
        for (var col = 0; col < Length; ++col)
            _transpose[0, col] = data[col];

        return _transpose;
    }


    /// <summary>
    ///     Returns the index of the maximum value in the vector.
    /// </summary>
    /// <param name="v"></param>
    /// <returns></returns>
    public int Max()
    {
        double max = long.MinValue;
        var index = -1;

        for (var i = 0; i < Length; i++)
            if (this[i] > max)
            {
                max = this[i];
                index = i;
            }

        return index;
    }

    public Vector Normalize() => this / Abs();
}