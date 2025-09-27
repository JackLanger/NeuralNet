using MathLib.Linalg;

namespace Tests.Math.Linalg;

[TestFixture]
public class MatrixTest {
    [Test]
    public void Constructor_CreatesCorrectSize()
    {
        var m = new Matrix(2, 3);
        Assert.That(m.Rows, Is.EqualTo(2));
        Assert.That(m.Cols, Is.EqualTo(3));
    }

    [Test]
    public void Indexer_SetsAndGetsValue()
    {
        var m = new Matrix(2, 2);
        m[0, 1] = 5;
        Assert.That(m[0, 1], Is.EqualTo(5));
    }

    [Test]
    public void Add_AddsMatricesCorrectly()
    {
        var a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });
        var result = a + b;
        Assert.That(result[0, 0], Is.EqualTo(6));
        Assert.That(result[0, 1], Is.EqualTo(8));
        Assert.That(result[1, 0], Is.EqualTo(10));
        Assert.That(result[1, 1], Is.EqualTo(12));
    }

    [Test]
    public void Subtract_SubtractsMatricesCorrectly()
    {
        var a = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });
        var b = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var result = a - b;
        Assert.That(result[0, 0], Is.EqualTo(4));
        Assert.That(result[0, 1], Is.EqualTo(4));
        Assert.That(result[1, 0], Is.EqualTo(4));
        Assert.That(result[1, 1], Is.EqualTo(4));
    }

    [Test]
    public void Multiply_MultipliesMatricesCorrectly()
    {
        var a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = new Matrix(new double[,] { { 2, 0 }, { 1, 2 } });
        var result = a * b;
        Assert.That(result[0, 0], Is.EqualTo(4));
        Assert.That(result[0, 1], Is.EqualTo(4));
        Assert.That(result[1, 0], Is.EqualTo(10));
        Assert.That(result[1, 1], Is.EqualTo(8));
    }

    [Test]
    public void Transpose_TransposesMatrix()
    {
        var m = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var t = m.T;
        Assert.That(t[0, 0], Is.EqualTo(1));
        Assert.That(t[0, 1], Is.EqualTo(4));
        Assert.That(t[1, 0], Is.EqualTo(2));
        Assert.That(t[1, 1], Is.EqualTo(5));
        Assert.That(t[2, 0], Is.EqualTo(3));
        Assert.That(t[2, 1], Is.EqualTo(6));
    }

    [Test]
    public void Scale_ScalesMatrixCorrectly()
    {
        var m = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var s = m * 2;
        Assert.That(s[0, 0], Is.EqualTo(2));
        Assert.That(s[0, 1], Is.EqualTo(4));
        Assert.That(s[1, 0], Is.EqualTo(6));
        Assert.That(s[1, 1], Is.EqualTo(8));
    }

    [Test]
    public void Equals_ReturnsTrueForEqualMatrices()
    {
        var a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        Assert.That(a.Equals(b), Is.True);
    }

    [Test]
    public void Equals_ReturnsFalseForDifferentMatrices()
    {
        var a = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var b = new Matrix(new double[,] { { 1, 2 }, { 4, 3 } });
        Assert.That(a.Equals(b), Is.False);
    }
}