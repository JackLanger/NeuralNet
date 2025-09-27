# NeuralNet

A simple feedforward neural network implementation in C# designed for handwritten digit recognition using the MNIST dataset.

## 🔥 Features

- **Feedforward Neural Network**: Multi-layer perceptron with customizable hidden layers
- **MNIST Dataset Integration**: Built-in support for loading and processing MNIST handwritten digit data
- **Backpropagation Learning**: Gradient descent optimization with configurable learning rates
- **Convolution Support**: Optional 2x2 average pooling for input compression (784 → 196 features)
- **Flexible Training Options**: Multiple training rate schedules (Constant, Logarithmic, Linear)
- **Real-time Progress Tracking**: Visual progress bars during training and assessment
- **Sigmoid Activation**: Standard sigmoid activation function for neural computation
- **Assessment Tools**: Built-in accuracy evaluation with test data

## 🚀 Quick Start

### Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later
- Internet connection (for downloading MNIST dataset)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JackLanger/NeuralNet.git
cd NeuralNet
```

2. Build the project:
```bash
dotnet build
```

3. Run the neural network:
```bash
dotnet run --project NeuronalesNetz
```

## 📖 Usage

### Basic Example

The default configuration trains a neural network with the following settings:

```csharp
var options = new ModelOptions
{
    Epochs = 5,
    EpochSize = 20000,
    LearningRate = .1f,
    TrainingRateOptions = TrainingRateOptions.Logarithmic,
    Convolution = false,
    Layers = [196]
};

var nm = new NetworkModel(options);

nm.Train();

for (var i = 0; i < 10; i++) nm.Assess(options.Convolution);
```

### Training Configuration

Customize your neural network training with these options:

| Property | Type | Default  | Description |
|----------|------|----------|-------------|
| `Epochs` | int | 5        | Number of complete training cycles |
| `EpochSize` | int | 1750     | Number of training samples per epoch |
| `LearningRate` | float | 0.1f     | Learning rate for weight updates |
| `TrainingRateOptions` | enum | Constant | Learning rate schedule |
| `Convolution` | bool | false    | Enable input compression via pooling |
| `Layers` | int[] | [256]    | Hidden layer sizes |
| `ActivatorFunction` | Func<float, float> | ReLU     | Activation function for neurons |

### Training Rate Options

- **Constant**: Fixed learning rate throughout training
- **Logarithmic**: Decreasing learning rate based on logarithmic decay
- **Linear**: Linear decay of learning rate

### Prediction

Make predictions on new handwritten digits:

```csharp
var networkModel = new NetworkModel();
// ... train the model ...

byte[] imageData = GetImageData(); // 784 bytes (28x28 pixels)
int predictedDigit = networkModel.PredictLabel(imageData, compress: false);
```

## 🏗️ Architecture

### Project Structure

```
NeuronalesNetz/
├── algo/
│   ├── NetworkModel.cs      # Main neural network implementation
│   ├── Convolution.cs       # 2x2 average pooling for input compression
│   └── Progress.cs          # Training progress visualization
├── extension/
│   ├── VectorExtensions.cs  # Sigmoid activation and utility functions
│   └── MatrixExtension.cs   # Matrix initialization helpers
├── IO/
│   └── MnistReader.cs       # MNIST dataset loader and utilities
├── lib/
│   └── Linalg.dll          # Linear algebra operations library
└── data/
    ├── train-images.idx3-ubyte  # MNIST training images
    ├── train-labels.idx1-ubyte  # MNIST training labels
    ├── t10k-images.idx3-ubyte   # MNIST test images
    └── t10k-labels.idx1-ubyte   # MNIST test labels
```

### Network Architecture

The neural network consists of:

1. **Input Layer**: 784 neurons (28×28 pixel images) or 196 with convolution
2. **Hidden Layers**: Configurable number and size of hidden layers
3. **Output Layer**: 10 neurons (digits 0-9)
4. **Activation Function**: Sigmoid function with derivative support
5. **Loss Function**: Mean squared error
6. **Optimization**: Standard backpropagation with gradient descent

### Data Flow

1. **Input Processing**: Raw MNIST images (28×28 pixels) are normalized to [0,1]
2. **Optional Convolution**: 2×2 average pooling reduces 784 features to 196
3. **Forward Propagation**: Data flows through hidden layers with sigmoid activation
4. **Output Generation**: Final layer produces probability distribution over 10 digits
5. **Backpropagation**: Error gradients propagate backward to update weights

## 📊 MNIST Dataset

The project automatically downloads and processes the MNIST handwritten digit dataset:

- **Training Set**: 60,000 images with labels
- **Test Set**: 10,000 images with labels
- **Image Format**: 28×28 grayscale pixels
- **Labels**: Digits 0-9

Dataset files are cached locally in the `data/` directory and loaded via HTTP requests to the GitHub repository.

## 🔧 Customization

### Creating Custom Network Architectures

```csharp
// Single hidden layer with 128 neurons
var options1 = new NeuralNetworkTrainingOptions
{
    Layers = [128]
};

// Multiple hidden layers
var options2 = new NeuralNetworkTrainingOptions
{
    Layers = [512, 256, 128]
};

// Deep network with convolution
var options3 = new NeuralNetworkTrainingOptions
{
    Layers = [256, 128, 64],
    Convolution = true
};
```

### Advanced Training Schedules

```csharp
// Long training with logarithmic decay
var options = new NeuralNetworkTrainingOptions
{
    Epochs = 10,
    EpochSize = 50000,
    LearningRate = 0.5f,
    TrainingRateOptions = TrainingRateOptions.Logarithmic,
    ActivatorFunction = ActivatorFunctions.LeakyReLU
};

```

## 🎯 Performance

Typical accuracy results after 5 epochs:
- **Basic Network (256 hidden)**: ~85-90% accuracy
- **With Convolution**: ~80-85% accuracy (faster training)
- **Deep Network**: ~90-95% accuracy (longer training time)

Training time depends on:
- Number of epochs and epoch size
- Network architecture complexity
- Hardware specifications

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

### Planned Features (TODOs)
- [ ] GPU acceleration with CUDA/ROCm support
- [ ] Model persistence (save/load trained networks)
- [ ] Different activation functions (ReLU, Tanh, etc.)
- [ ] Additional loss functions (Cross-entropy, etc.)
- [ ] Regularization techniques (Dropout, Weight decay)
- [ ] Batch training for improved efficiency
- [ ] Learning rate schedules
- [ ] Data provider abstraction for different datasets
- [ ] Multi-threading for progress updates
- [ ] Vector2/Vector3 support for advanced convolution layers

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

### Code Style

- Follow C# naming conventions
- Add XML documentation for public methods
- Include unit tests for new features
- Maintain compatibility with .NET 8.0+

## 📄 License

This project is open source. See the repository for license details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JackLanger/NeuralNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JackLanger/NeuralNet/discussions)

## 🙏 Acknowledgments

- MNIST dataset creators and maintainers
- .NET community for excellent tooling

---

**Happy Learning!** 🧠✨
