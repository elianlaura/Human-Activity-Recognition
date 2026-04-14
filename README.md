# Human Activity Recognition (HAR) Classifier

This project implements a Human Activity Recognition (HAR) classifier using a two-stream transformer-based architecture with attention mechanisms for classifying human activities based on accelerometer and gyroscope sensor data.

## Features

- **Two-Stream Architecture**: Dual-input model processing accelerometer and gyroscope data separately
- **Advanced Feature Extraction**: Convolutional layers with SiLU activation for initial feature extraction from sensor data
- **Attention Mechanisms**: Multi-head self-attention within each stream and cross-attention between accelerometer and gyroscope streams
- **Positional Embeddings**: Custom positional encoding for temporal sequence understanding
- **Fusion Layers**: Configurable number of fusion layers combining information from both sensor streams
- **Regularization**: L2 regularization, dropout, and layer normalization for robust training
- **Data Preprocessing**: Comprehensive preprocessing including normalization, segmentation, and overlap handling
- **Cross-Validation**: Support for k-fold cross-validation with user-based splitting
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1-score, balanced accuracy, confusion matrices, sensitivity, specificity, and ROC curves
- **TensorBoard Integration**: Real-time monitoring of training progress and model performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Har_Classifier
```

2. Create and activate a conda environment:
```bash
conda create -n har-env python=3.9
conda activate har-env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- TensorFlow 2.15.0
- PyTorch 1.13.1+cu117
- NumPy 1.23.5
- Pandas 2.3.3
- Matplotlib 3.7.1
- Scikit-learn 1.3.0
- SciPy 1.9.3

## Usage

### Training

Run the main training script:

```bash
python src/main.py
```

### Synthetic Data Generation (PaD-TS Diffusion)

This repository now includes the PaD-TS diffusion pipeline under `pad_ts/` for synthetic time-series generation.

Example:

```bash
python pad_ts/run.py -data drinkeat
```

### Configuration

Modify the parameters in `main.py` to configure:

- `modeltype`: Type of model (e.g., 'har_classifier')
- `dataset`: Dataset name (e.g., 'data_5')
- `n_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `sensors`: Number of sensors (2 or 3)
- Other hyperparameters

### Data Preparation

The project expects data in CSV format with sensor readings. Place your data files in the appropriate directory as specified in the code.

## Project Structure

```
Har_Classifier/
├── src/
│   ├── har_classifier.py      # Main training and evaluation script
│   ├── main.py               # Entry point script
│   ├── models.py             # Neural network model definitions
│   └── utils.py              # Utility functions for data processing and metrics
├── pad_ts/                   # Imported PaD-TS diffusion module for synthetic data generation
│   ├── run.py                # Main PaD-TS training/generation script
│   ├── Model.py              # PaD-TS architecture
│   ├── configs/              # Dataset and training configs for diffusion runs
│   └── data_preprocessing/   # Data loaders and sampling utilities
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── LICENSE                  # MIT License
├── saved_models/            # Directory for saved models and results (gitignored)
└── README.md                # This file
```

## Model Architecture

The HAR classifier uses a sophisticated two-stream transformer-based architecture:

- **Dual Input Streams**: Separate processing pipelines for accelerometer and gyroscope sensor data
- **Feature Extraction**: Convolutional layers (1D convolutions with kernel sizes 5 and 3) followed by max pooling and dense layers for initial feature extraction
- **Tokenization**: Projection to embedding space with positional embeddings for temporal sequence understanding
- **Attention Layers**: Multi-head self-attention within each stream and cross-attention between streams for information fusion
- **Fusion Mechanism**: Configurable number of fusion layers that alternate between self-attention and cross-attention
- **Global Pooling**: Average pooling to obtain stream-level representations
- **Classification Head**: Concatenation of both streams, followed by dense layers with dropout and softmax activation
- **Regularization**: L2 weight decay, dropout, and layer normalization throughout the network
- **Loss Function**: Sparse categorical cross-entropy
- **Optimizer**: Adam with weight decay

The model is designed to capture both intra-sensor temporal patterns and inter-sensor relationships for robust activity classification.

## Evaluation

The model is evaluated using:

- Accuracy
- Balanced Accuracy
- F1-Score (weighted, macro, micro)
- Precision and Recall
- Confusion Matrix
- ROC Curves
- Sensitivity and Specificity per class

Results are saved to text files and plots are generated for loss/accuracy curves and confusion matrices.

## TensorBoard

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir saved_models/recurrent_models_*/logs/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

mailto: e265685@dac.unicamp.br