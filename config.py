import torch
import sys

# --- General Parameters ---
DEBUG = True
USE_GRADIENT = True
LAYERS = 5
TINY = True
IDIM=784
DIM = 1
SHRINK = 10
BATCH_SIZE = 128

try:
    SEED = int(sys.argv[1])
except (IndexError, ValueError):
    SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


# --- Architecture Definitions ---

# 1D architecture for flattened MNIST
mnist_conv1d_architecture = [
    {"type": "Conv1d", "out_channels": 2, "kernel_size": 32, "stride": 2, "padding": 0},
    {"type": "ReLU"},
    {"type": "Conv1d", "out_channels": 2, "kernel_size": 32, "stride": 2, "padding": 0},
    {"type": "ReLU"},
    {"type": "Conv1d", "out_channels": 1, "kernel_size": 32, "stride": 1, "padding": 0},
    {"type": "ReLU"},
    {"type": "Flatten"},
    {"type": "Linear", "out_features": 10}
]

# mnist_conv2d_architecture = [
#     # Input: (1, 28, 28)
#     # Use stride to reduce dimensions instead of pooling
#     {"type": "Conv2d", "out_channels": 2, "kernel_size": 5, "stride": 2, "padding": 2},
#     {"type": "ReLU"},
#
#     {"type": "Conv2d", "out_channels": 2, "kernel_size": 5, "stride": 2, "padding": 2},
#     {"type": "ReLU"},
#
#     {"type": "Flatten"},
#     # Flattened features = 32 * 7 * 7 = 1568
#     {"type": "Linear", "out_features": 120},
#     {"type": "ReLU"},
#     {"type": "Linear", "out_features": 84},
#     {"type": "ReLU"},
#     {"type": "Linear", "out_features": 10}
# ]
mnist_conv2d_architecture = [
    # Input: (1, 28, 28)
    {"type": "Conv2d", "out_channels": 6, "kernel_size": 5, "stride": 1, "padding": 2},  # (1,28,28) -> (6,28,28)
    {"type": "ReLU"},
    {"type": "AvgPool2d", "kernel_size": 2, "stride": 2},  # (6,28,28) -> (6,14,14)

    {"type": "Conv2d", "out_channels": 16, "kernel_size": 5, "stride": 1, "padding": 0},  # (6,14,14) -> (16,10,10)
    {"type": "ReLU"},
    {"type": "AvgPool2d", "kernel_size": 2, "stride": 2},  # (16,10,10) -> (16,5,5)

    {"type": "Flatten"},  # 16*5*5 = 400
    {"type": "Linear", "out_features": 120},
    {"type": "ReLU"},
    {"type": "Linear", "out_features": 84},
    {"type": "ReLU"},
    {"type": "Linear", "out_features": 10}
]

# --- Dataset-Specific Configurations ---

DATASET_CONFIGS = {
    'mnist_1d': {
        'input_channels': 1,
        'input_shape': 784, # Represents flattened length
        'num_classes': 10,
        'architecture': mnist_conv1d_architecture
    },
    'mnist_2d': {
        'input_channels': 1,
        'input_shape': (28, 28), # Represents (Height, Width)
        'num_classes': 10,
        'architecture': mnist_conv2d_architecture
    },
    'custom': {
        'input_channels': 1,
        'input_shape': 300,
        'num_classes': 10,
        'architecture': mnist_conv1d_architecture
    }
}