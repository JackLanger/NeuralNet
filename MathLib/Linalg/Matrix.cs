namespace MathLib.Linalg;

public class Matrix(double[,] data) {
    private readonly MatrixComperator _comparer = new();
    private double? _det;
    private Matrix? _transpose;

    private Matrix(double[,] data, Matrix trans) : this(data)
    {
        _transpose = trans;
    }

    public Matrix(int n)
        : this(n, n)
    {
    }

    public Matrix(int n, int m) : this(new double[n, m]) {}

    public Matrix(double[] data, int n, int m)
        : this(n, m)
    {
        for (var index1 = 0; index1 < n; ++index1)
        for (var index2 = 0; index2 < m; ++index2)
            Data[index1, index2] = data[index1 * index2 + index2];
    }

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

    private double[,] Data { get; }

    public int Rows => Data.GetLength(0);

    public int Cols => Data.GetLength(1);

    public int Rank => Data.Rank;

    public Matrix T
    {
        get
        {
            if (_transpose != null)
            {
                return _transpose;
            }
            var data = new double[Cols, Rows];
            for (var row = 0; row < Rows; ++row)
            for (var col = 0; col < Cols; ++col)
                data[col, row] = this[row, col];
            _transpose = new Matrix(data, this);

            return _transpose;
        }
    }

    public double this[int row, int col]
    {
        get => Data[row, col];
        set => Data[row, col] = value;
    }

    private Row this[int row]
    {
        get => GetRow(row);
        set => SetRow(value, row);
    }

    public double[] GetColumn(int col)
    {
        return Enumerable.Range(0, Data.GetLength(0)).Select(x => Data[x, col]).ToArray();
    }

    private Row GetRow(int row)
    {
        return new Row(Enumerable.Range(0, Data.GetLength(1)).Select(x => Data[row, x]).ToArray());
    }

    private void SetRow(Row rowdata, int row)
    {
        for (var i = 0; i < Cols; ++i)
            Data[row, i] = rowdata[i];
    }

    public override bool Equals(object? obj) => obj is Matrix other && Equals(other);

    private bool Equals(Matrix other) => _comparer.Equals(this, other);

    public override int GetHashCode() => HashCode.Combine(Data);

    public void SetCol(double[] colData, int col)
    {
        for (var index = 0; index < Cols; ++index)
            Data[index, col] = colData[index];
    }

    public static Matrix operator +(Matrix left, Matrix right)
    {
        if (left.Rows != right.Rows && left.Cols != right.Cols)
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
        var length = Data.Length;
        if (length < 3)
        {
            if (length == 1)
            {
                return Data[0, 0];
            }

            if (length == 2)
            {
                return Data[0, 0] * Data[1, 1] - Data[1, 0] * Data[0, 1];
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
            num1 += num2 * Data[i, j] * GetSubArray(i, j).Determinant();
        }

        return num1;
    }

    private Matrix GetSubArray(int i, int j)
    {
        var n = Data.Length - 1;
        var subArray = n >= 2 ? new Matrix(n) : throw new ArgumentException("Invalid exception Sub Matrix of size 1 cannot be created as it is a single value");
        for (var row = 0; row < n; ++row)
        for (var col = 0; col < n; ++col)
            if (row > i && col > j)
            {
                subArray[row - 1, col - 1] = Data[row, col];
            }
            else if (row > n)
            {
                subArray[row - 1, col] = Data[row, col];
            }
            else if (col > n)
            {
                subArray[row, col - 1] = Data[row, col];
            }
            else
            {
                subArray[row, col] = Data[row, col];
            }

        return subArray;
    }

    private double DeterminantSarrus() => Data[0, 0] * Data[1, 1] * Data[2, 2] + Data[1, 0] * Data[2, 1] * Data[0, 2] + Data[2, 0] * Data[0, 1] * Data[1, 2] - Data[2, 0] * Data[1, 1] * Data[0, 2] - Data[2, 1] * Data[1, 2] * Data[0, 0] - Data[2, 2] * Data[1, 0] * Data[0, 1];

    private double SmallDetermining() => Data[0, 0] * Data[1, 1] + Data[0, 1] * Data[1, 0] - Data[1, 0] * Data[0, 1] - Data[0, 1] * Data[0, 0];
}