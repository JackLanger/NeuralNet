using MathLib.Linalg;

namespace Tests.Math.Linalg;

[TestFixture]
public class VectorTest {
    [Test]
    public void Constructor_CreatesCorrectSize()
    {
        var v = new Vector(3);
        Assert.That(v.Length, Is.EqualTo(3));
    }

    [Test]
    public void Indexer_SetsAndGetsValue()
    {
        var v = new Vector(2);
        v[1] = 5;
        Assert.That(v[1], Is.EqualTo(5));
    }

    [Test]
    public void Add_AddsVectorsCorrectly()
    {
        var a = new Vector([1, 2, 3]);
        var b = new Vector([4, 5, 6]);
        var result = a + b;
        Assert.That(result[0], Is.EqualTo(5));
        Assert.That(result[1], Is.EqualTo(7));
        Assert.That(result[2], Is.EqualTo(9));
    }

    [Test]
    public void Subtract_SubtractsVectorsCorrectly()
    {
        var a = new Vector([5, 7, 9]);
        var b = new Vector([1, 2, 3]);
        var result = a - b;
        Assert.That(result[0], Is.EqualTo(4));
        Assert.That(result[1], Is.EqualTo(5));
        Assert.That(result[2], Is.EqualTo(6));
    }

    [Test]
    public void DotProduct_ComputesCorrectly()
    {
        var a = new Vector([1, 2, 3]);
        var b = new Vector([4, 5, 6]);
        var dot = a * b;
        Assert.That(dot, Is.EqualTo(32));
    }

    [Test]
    public void ScalarMultiplication_MultipliesCorrectly()
    {
        var v = new Vector([1, 2, 3]);
        var result = v * 2;
        Assert.That(result[0], Is.EqualTo(2));
        Assert.That(result[1], Is.EqualTo(4));
        Assert.That(result[2], Is.EqualTo(6));
    }


    [Test]
    public void Equals_ReturnsTrueForEqualVectors()
    {
        var a = new Vector([1, 2, 3]);
        var b = new Vector([1, 2, 3]);
        Assert.That(a.Equals(b), Is.True);
    }

    [Test]
    public void Equals_ReturnsFalseForDifferentVectors()
    {
        var a = new Vector([1, 2, 3]);
        var b = new Vector([3, 2, 1]);
        Assert.That(a.Equals(b), Is.False);
    }

    [Test]
    public void Transpose_ReturnsColumnVector()
    {
        var v = new Vector([1, 2, 3]);
        var t = v.T;
        Assert.That(t.Rows, Is.EqualTo(1));
        Assert.That(t.Cols, Is.EqualTo(3));
        Assert.That(t[0, 0], Is.EqualTo(1));
        Assert.That(t[0, 1], Is.EqualTo(2));
        Assert.That(t[0, 2], Is.EqualTo(3));
    }
}