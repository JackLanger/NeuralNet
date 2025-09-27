namespace MathLib.Linalg;

public class Matrix(double[,] data) {
    private readonly MatrixComparator _comparer = new();
    private double? _det;
    private Matrix? _transpose;

    private Matrix(double[,] data, Matrix trans) : this(data)
    {
        _transpose = trans;
    }
    private Matrix(int n)
        : this(n, n)
    {
    }

    public Matrix(int n, int m) : this(new double[n, m]) {}


    public double D
    {
        get
        {
            var valueOrDefault = _det.GetValueOrDefault();
            if (_det.HasValue)
            {
                return valueOrDefault;
            }
            var d = Determinant();
            _det = d;

            return d;
        }
    }


    public int Rows => data.GetLength(0);

    public int Cols => data.GetLength(1);

    public int Rank => data.Rank;

    public Matrix T
    {
        get
        {
            if (_transpose != null)
            {
                return _transpose;
            }
            var transposed = new double[Cols, Rows];
            for (var row = 0; row < Rows; ++row)
            for (var col = 0; col < Cols; ++col)
                transposed[col, row] = this[row, col];
            _transpose = new Matrix(transposed, this);

            return _transpose;
        }
    }

    public double this[int row, int col]
    {
        get => data[row, col];
        set => data[row, col] = value;
    }

    private Row this[int row]
    {
        get => GetRow(row);
        set => SetRow(value, row);
    }

    public double[] GetColumn(int col)
    {
        return Enumerable.Range(0, data.GetLength(0)).Select(x => data[x, col]).ToArray();
    }

    private Row GetRow(int row)
    {
        return new Row(Enumerable.Range(0, data.GetLength(1)).Select(x => data[row, x]).ToArray());
    }

    private void SetRow(Row rowdata, int row)
    {
        for (var i = 0; i < Cols; ++i)
            data[row, i] = rowdata[i];
    }

    public override bool Equals(object? obj) => obj is Matrix other && Equals(other);

    private bool Equals(Matrix other) => _comparer.Equals(this, other);

    public override int GetHashCode() => HashCode.Combine(data);

    public void SetCol(double[] colData, int col)
    {
        for (var index = 0; index < Cols; ++index)
            data[index, col] = colData[index];
    }

    public static Matrix operator +(Matrix left, Matrix right)
    {
        if (left.Rows != right.Rows || left.Cols != right.Cols)
        {
            throw new ArgumentException($"Cannot add matrices of different sizes {left.Rows}x{left.Cols} and {right.Rows}x{right.Cols}");
        }
        var matrix = new Matrix(left.Rows, left.Cols);
        for (var row = 0; row < left.Rows; ++row)
        for (var col = 0; col < left.Cols; ++col)
            matrix[row, col] = left[row, col] + right[row, col];

        return matrix;
    }

    public static Matrix operator /(Matrix m, double alpha) => m * (1.0 / alpha);

    public static Matrix operator /(double alpha, Matrix m) => 1.0 / alpha * m;

    public static Matrix operator *(Matrix left, Matrix right)
    {
        if (left.Cols != right.Rows)
        {
            throw new ArgumentException($"Cannot multiply matrices of sizes {left.Rows}x{left.Cols} and {right.Rows}x{right.Cols}");
        }
        var matrix = new Matrix(left.Rows, right.Cols);
        for (var row = 0; row < left.Rows; ++row)
        for (var col = 0; col < right.Cols; ++col)
        for (var index = 0; index < right.Rows; ++index)
            matrix[row, col] += left[row, index] * right[index, col];

        return matrix;
    }

    public static Matrix operator *(Matrix matrix, double alpha)
    {
        var matrix1 = new Matrix(matrix.Rows, matrix.Cols);
        for (var row = 0; row < matrix.Rows; ++row)
        for (var col = 0; col < matrix.Cols; ++col)
            matrix1[row, col] = matrix[row, col] * alpha;

        return matrix1;
    }

    public static Matrix operator *(double alpha, Matrix matrix) => matrix * alpha;

    public static Matrix operator -(Matrix lft, Matrix rgt) => lft + rgt * -1.0;

    public void PivotRows(int n, int m)
    {
        try
        {
            var row1 = n;
            var row2 = m;
            var row3 = this[m];
            var row4 = this[n];
            this[row1] = row3;
            this[row2] = row4;
        }
        catch (IndexOutOfRangeException ex)
        {
            throw new ArgumentException(n > Rows ? $"Cannot pivot row {n} as the matrix has only {Rows} rows." : $"Cannot pivot row {m} as the matrix has only {Rows} rows.", ex);
        }
    }

    private double Determinant()
    {
        if (Rows != Cols)
        {
            throw new ArgumentException("The can not be calculate the determinant of a non square matrix.");
        }
        var length = data.Length;
        if (length < 3)
        {
            if (length == 1)
            {
                return data[0, 0];
            }

            if (length == 2)
            {
                return data[0, 0] * data[1, 1] - data[1, 0] * data[0, 1];
            }

            throw new ArgumentException("Matrix has size of 0");
        }
        if (length == 3)
        {
            return DeterminantSarrus();
        }
        var num1 = 0.0;
        for (var i = 0; i < length; ++i)
        for (var j = 0; j < length; ++j)
        {
            var num2 = (int)Math.Pow(-1.0, i + j);
            num1 += num2 * data[i, j] * GetSubArray(i, j).Determinant();
        }

        return num1;
    }

    private Matrix GetSubArray(int i, int j)
    {
        var n = data.Length - 1;
        var subArray = n >= 2 ? new Matrix(n) : throw new ArgumentException("Invalid exception Sub Matrix of size 1 cannot be created as it is a single value");
        for (var row = 0; row < n; ++row)
        for (var col = 0; col < n; ++col)
            if (row > i && col > j)
            {
                subArray[row - 1, col - 1] = data[row, col];
            }
            else if (row > n)
            {
                subArray[row - 1, col] = data[row, col];
            }
            else if (col > n)
            {
                subArray[row, col - 1] = data[row, col];
            }
            else
            {
                subArray[row, col] = data[row, col];
            }

        return subArray;
    }

    private double DeterminantSarrus() => data[0, 0] * data[1, 1] * data[2, 2] + data[1, 0] * data[2, 1] * data[0, 2] + data[2, 0] * data[0, 1] * data[1, 2] - data[2, 0] * data[1, 1] * data[0, 2] - data[2, 1] * data[1, 2] * data[0, 0] - data[2, 2] * data[1, 0] * data[0, 1];

    private double SmallDetermining() => data[0, 0] * data[1, 1] + data[0, 1] * data[1, 0] - data[1, 0] * data[0, 1] - data[0, 1] * data[0, 0];
}