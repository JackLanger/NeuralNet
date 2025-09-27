namespace MathLib.Linalg;

public class MatrixComperator : IEqualityComparer<Matrix> {
    public bool Equals(Matrix? x, Matrix? y)
    {
        if (x == null || y == null || x.Rows != y.Rows || x.Cols != y.Cols)
        {
            return false;
        }
        for (var row = 0; row < x.Rows; ++row)
        for (var col = 0; col < x.Cols; ++col)
            if (Math.Abs(x[row, col] - y[row, col]) > 1E-06)
            {
                return false;
            }

        return true;
    }

    public int GetHashCode(Matrix obj) => obj.GetHashCode();
}