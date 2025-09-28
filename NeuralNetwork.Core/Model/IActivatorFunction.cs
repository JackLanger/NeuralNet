using MathLib.Linalg;

namespace NeuralNetworkLib.Model;

public interface IActivatorFunction {
    /// <summary>
    ///     Activation function to introduce non-linearity into the model.
    ///     Activation functions are crucial in neural networks as they help the model learn complex
    ///     patterns
    ///     and relationships in the data by allowing it to capture non-linearities.
    ///     Common activation functions include Sigmoid, Tanh, ReLU (Rectified Linear Unit), and Leaky
    ///     ReLU.
    ///     The choice of activation function can significantly impact the performance and convergence of
    ///     the
    ///     neural network during training.
    /// </summary>
    /// <param name="m">Vector to be activated</param>
    /// <returns>A new Activated Vector</returns>
    Matrix Activate(Matrix m);
    /// <summary>
    ///     Derivative of the activation function.
    ///     The derivative is essential for the backpropagation algorithm, which is used to train neural
    ///     networks.
    ///     During backpropagation, the derivative of the activation function is used to compute the
    ///     gradients needed to update the weights of the network.
    ///     This helps the model learn by adjusting the weights in a way that minimizes the error between
    ///     the predicted output and the actual target values.
    /// </summary>
    /// <param name="inputLayer">Input features or hidden layer before activation (e.g., raw features)</param>
    /// <param name="activatedNextLayer">
    ///     Activated outputs of the next layer (e.g., output layer after
    ///     activation)
    /// </param>
    /// <param name="errorVector">Error vector between prediction and target values</param>
    /// <returns>Matrix of derivatives for backpropagation</returns>
    Matrix Derivative(Matrix inputLayer, Matrix activatedNextLayer, Matrix errorVector);
}