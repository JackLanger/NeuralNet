namespace MathLib.Linalg;

public class Vector(double[] data) {
    private Matrix? _transpose;
    public Vector(int n)
        : this(new double[n])
    {
    }


    public int Length => data.Length;

    public Matrix T => (_transpose ??= Transpose())!;

    public double this[int i]
    {
        get => data[i];
        set => data[i] = value;
    }

    private Matrix? Transpose()
    {
        var transposed = new Matrix(1, Length);
        for (var col = 0; col < Length; ++col)
            transposed[0, col] = data[col];

        return transposed;
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

    public static double operator *(Vector a, Vector b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Operation invalid on vertices of different dimensions");
        }
        var dot = 0.0;
        for (var i = 0; i < a.Length; ++i)
            dot += a[i] * b[i];

        return dot;
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

    /// <summary>
    ///     Multiplies two vectors element wise, such that (a1, a2, a3) * (b1, b2, b3) = (a1*b1, a2*b2,
    ///     a3*b3)
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public Vector Hadamard(Vector other)
    {
        if (Length != other.Length)
        {
            throw new ArgumentException("Operation invalid on vertices of different dimensions");
        }
        var vec = new Vector(Length);
        for (var i = 0; i < Length; ++i)
            vec[i] = this[i] * other[i];

        return vec;
    }

    public double Abs()
    {
        var d = 0.0;
        for (var index = 0; index < Length; ++index)
            d += data[index] * data[index];

        return Math.Sqrt(d);
    }


    /// <summary>
    ///     Returns the index of the maximum value in the vector.
    /// </summary>
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

    public override bool Equals(object? obj)
    {
        if (obj is not Vector other || other.Length != Length)
        {
            return false;
        }


        for (var i = 0; i < Length; i++)
            if (Math.Abs(this[i] - other[i]) > .000001)
            {
                return false;
            }

        return true;

    }

    public override int GetHashCode() => HashCode.Combine(data, Length);
    public Vector Map(Func<double, double> func)
    {
        for (var i = 0; i < Length; i++)
            data[i] = func(data[i]);

        return this;
    }
}