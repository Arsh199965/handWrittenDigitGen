# Handwritten Digit Generator

A conditional GAN (Generative Adversarial Network) that generates handwritten digits (0-9) using PyTorch and displays them through an interactive Streamlit web application.

## Features

- **Conditional Generation**: Generate specific digits (0-9) on demand
- **Interactive Web UI**: Easy-to-use Streamlit interface with digit selection
- **Pre-trained Model**: Includes a trained generator model ready for use
- **Real-time Generation**: Generate 5 sample images instantly with a button click

## Demo

The application allows you to:
1. Select any digit from 0-9 using a slider
2. Generate 5 different variations of that digit
3. View the generated images in grayscale format

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd HandWrittenDigitGenerator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Training Your Own Model

If you want to train a new model from scratch:

```bash
python train_model.py
```

**Note**: Training requires:
- MNIST dataset (automatically downloaded)
- Approximately 30-60 minutes on CPU (much faster with GPU)
- The trained model will be saved as `generator.pth`

## Model Architecture

### Generator
- **Input**: 100-dimensional noise vector + digit label (0-9)
- **Architecture**: 
  - Embedding layer for class labels (10 classes → 10 dimensions)
  - Fully connected layers: 110 → 256 → 512 → 1024 → 784 (28×28)
  - Activations: ReLU for hidden layers, Tanh for output
  - Includes dropout (0.3) and batch normalization for stability
- **Output**: 28×28 grayscale image in range [-1, 1]

### Discriminator
- **Input**: 28×28 image + digit label
- **Architecture**:
  - Embedding layer for class labels (10 classes → 10 dimensions)
  - Fully connected layers: 794 (784+10) → 512 → 256 → 1
  - Activations: LeakyReLU (α=0.2) for hidden layers, Sigmoid for output
- **Output**: Binary probability (real/fake)

## Training Details

### Hyperparameters
- **Latent Dimension**: 100
- **Batch Size**: 128
- **Epochs**: 50
- **Learning Rate**: 0.0002
- **Beta1**: 0.5 (Adam optimizer momentum)
- **Beta2**: 0.999 (Adam optimizer momentum)

### Loss Functions

#### Adversarial Loss
Both generator and discriminator use **Binary Cross Entropy (BCE) Loss**:

```
L_BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

#### Generator Loss
The generator aims to fool the discriminator:
```
L_G = BCE(D(G(z, c)), 1)
```
Where:
- `G(z, c)` = Generated image from noise `z` and class `c`
- `D(G(z, c))` = Discriminator's prediction on generated image
- Target = 1 (generator wants discriminator to think images are real)

#### Discriminator Loss
The discriminator learns to distinguish real from fake:
```
L_D = [BCE(D(x_real, c), 1) + BCE(D(G(z, c), c), 0)] / 2
```
Where:
- `x_real` = Real MNIST images
- First term: Loss on real images (target = 1)
- Second term: Loss on fake images (target = 0)
- Average of both losses

### Training Process

1. **Data Preprocessing**: 
   - Images normalized to [-1, 1] range using `transforms.Normalize([0.5], [0.5])`
   - Matches Tanh output activation range

2. **Alternating Training**:
   - Train Generator: Minimize ability of discriminator to detect fakes
   - Train Discriminator: Maximize ability to distinguish real from fake
   - Use `.detach()` on fake images when training discriminator to prevent gradients flowing back to generator

3. **Optimization**:
   - Adam optimizer with β₁=0.5, β₂=0.999
   - Learning rate: 0.0002 for both networks
   - Batch size: 128 samples

### Loss Monitoring
Training progress is logged every 200 batches:
```
[Epoch 1/50] [Batch 200/469] [D loss: 0.6234] [G loss: 1.2567]
```

**Ideal Loss Behavior**:
- **Discriminator Loss**: Should stabilize around 0.5-0.7
- **Generator Loss**: Should decrease initially, then stabilize around 0.8-1.5
- Both losses should not reach 0 (indicates mode collapse)

## Project Structure

```
HandWrittenDigitGenerator/
├── app.py              # Streamlit web application
├── train_model.py      # GAN training script with loss functions
├── generator.pth       # Pre-trained generator model
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## Technical Implementation

### Image Generation Pipeline
1. **Input**: Random noise vector (100D) + digit label
2. **Processing**: 
   - Label embedding converts class to 10D vector
   - Concatenate noise + embedding → 110D input
   - Pass through generator network
3. **Output**: 28×28 image in [-1, 1] range
4. **Display**: Normalize to [0, 1] and convert to 8-bit grayscale

### Conditional Generation
The model uses label embeddings to generate specific digits:
- Each digit (0-9) has a learnable 10-dimensional embedding
- Embeddings are concatenated with noise vector
- Allows controlled generation of desired digit classes

## Requirements

See `requirements.txt` for dependencies:
- streamlit - Web application framework
- torch - Deep learning framework
- numpy - Numerical computations
- torchvision - Computer vision utilities (MNIST dataset)

## GPU Support

The training script automatically detects and uses GPU if available (NVIDIA CUDA). 

**Performance Comparison**:
- **CPU**: ~45-60 minutes for 50 epochs
- **GPU**: ~5-10 minutes for 50 epochs

To enable GPU support:
1. Install CUDA-compatible PyTorch version
2. Ensure NVIDIA drivers are installed
3. The script will automatically use GPU when available

## Training Tips

### Successful Training Indicators
- **Balanced losses**: Neither generator nor discriminator should dominate
- **Visual quality**: Generated digits should become clearer over epochs
- **Stability**: Losses should not oscillate wildly

### Common Issues
- **Mode Collapse**: Generator produces limited variety → Reduce learning rate
- **Training Instability**: Losses oscillate → Add noise to discriminator inputs
- **Poor Quality**: Images remain blurry → Increase training epochs or adjust architecture

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- MNIST dataset creators
- PyTorch community for excellent deep learning framework
- Streamlit team for the intuitive web framework
- Original GAN paper by Ian Goodfellow et al.

## Troubleshooting

### Common Issues

1. **"No module named 'train_model'"**: Ensure you're running the app from the project root directory
2. **"generator.pth not found"**: Run `python train_model.py` to train and save the model first
3. **Poor image quality**: Model may need more training epochs or different hyperparameters
4. **Training divergence**: Try reducing learning rate or adding noise to inputs

### Getting Help

If you encounter any issues:
1. Check the console output for error messages
2. Monitor loss values during training for stability
3. Ensure all dependencies are correctly installed
4. Verify you're using Python 3.7 or higher

---

**Made with ❤️ using PyTorch and Streamlit**