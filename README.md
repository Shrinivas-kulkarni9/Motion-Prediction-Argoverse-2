# Multi-Agent Trajectory Prediction with Transformer Architecture

A PyTorch implementation of a transformer-based model for predicting future vehicle trajectories in multi-agent traffic scenarios, designed for the Argoverse 2 dataset.

## Overview

This project implements an agent-aware trajectory prediction system that:
- Predicts 60 future timesteps (6 seconds) based on 50 historical timesteps (5 seconds)
- Handles multi-agent interactions using spatial attention mechanisms
- Employs temporal transformers for sequence modeling
- Uses comprehensive data augmentation for robust training
- Incorporates ego-centric coordinate transformations and rich feature engineering

## Features

- **Multi-Agent Modeling**: Handles up to 50 agents per scene with spatial attention
- **Temporal Encoding**: Transformer encoder with positional encoding for temporal dependencies
- **Rich Feature Engineering**: 19-dimensional feature vectors including velocities, accelerations, relative positions, and rotated coordinates
- **Data Augmentation**: Rotation, reflection, Gaussian noise, time warping, and velocity perturbation
- **Ego-Centric Processing**: Coordinate transformation relative to ego vehicle's heading
- **Combined Loss Function**: MSE + Final Displacement Error (FDE) weighting
- **Early Stopping & Scheduling**: Adaptive learning rate with patience-based early stopping

## Architecture

### Model Components

1. **Input Processing**
   - Agent type embeddings
   - Feature normalization and projection
   - Positional encoding for temporal sequences

2. **Temporal Encoder**
   - Multi-layer Transformer encoder
   - Attention pooling for temporal aggregation
   - Layer normalization and dropout

3. **Spatial Attention**
   - Cross-attention mechanism where ego vehicle attends to all agents
   - Multi-head attention for capturing spatial relationships

4. **Decoder**
   - Fully connected layers
   - Outputs 60 future trajectory points (x, y coordinates)

### Feature Engineering

The model uses 19 input features per agent per timestep:
- Position (x, y) and velocity (vx, vy)
- Agent type and validity flags  
- Acceleration components (dvx, dvy, magnitude)
- Heading change (dtheta)
- Relative positions to ego vehicle
- Ego-rotated coordinates and velocities
- Distance to ego vehicle

## Installation

```bash
# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install torch numpy pandas matplotlib tqdm scikit-learn
```

### Requirements

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- tqdm

## Data Format

The model expects data in the following format:

**Training Data**: `(N, 110, 50, 6)` 
- N: Number of scenes
- 110: Total timesteps (50 historical + 60 future)
- 50: Maximum number of agents
- 6: Features [x, y, vx, vy, agent_type, valid]

**Testing Data**: `(N, 50, 50, 6)`
- Same format but only historical timesteps

## Usage

### Training

```python
# Load your data
train_npz = np.load('train.npz')
train_data = train_npz['data']

# Initialize datasets
scale = 7.0  # Normalization scale
train_dataset = TrajectoryDatasetTrain(train_data, scale=scale, augment=True)

# Train the model
model = AgentAwarePredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training loop (see full code for details)
```

### Inference

```python
# Load test data
test_dataset = TrajectoryDatasetTest(test_data, scale=scale)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Generate predictions
pred_list = []
with torch.no_grad():
    for batch in test_loader:
        pred_norm = model(batch.to(device))
        pred = pred_norm * batch.scale + batch.origin
        pred_list.append(pred.cpu().numpy())
```

## Configuration

Key hyperparameters:

```python
# Model architecture
hidden_size = 128
nhead = 4  # Multi-head attention heads
num_layers = 2  # Transformer encoder layers
dropout_rate = 0.0

# Training
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 32
early_stopping_patience = 10
fde_weight = 0.2  # Final displacement error weight
```

## Performance

The model is evaluated using:
- **MSE Loss**: Mean squared error in normalized coordinates
- **World MSE**: Mean squared error in world coordinates  
- **MAE**: Mean absolute error in world coordinates
- **FDE**: Final displacement error at the 60th timestep

## Data Augmentation

Training includes several augmentation techniques:
- **Rotation**: Random rotation by angle θ ∈ [-π, π]
- **Reflection**: Horizontal flip with 50% probability
- **Gaussian Noise**: Position jitter with σ = 0.1
- **Time Warping**: Temporal scaling by factor ∈ [0.9, 1.1]
- **Velocity Perturbation**: Gaussian noise on velocity components

## File Structure

```
├── trajectory_prediction.py    # Main training script
├── best_model.pt              # Saved model weights
├── submission.csv             # Prediction output
└── README.md                  # This file
```

## Output

The model generates a CSV file with predicted trajectories:
```csv
index,x,y
0,prediction_x_t1,prediction_y_t1
1,prediction_x_t2,prediction_y_t2
...
```

Each scene contributes 60 rows (60 future timesteps) with (x, y) coordinates in world coordinates.

## Device Support

The model automatically detects and uses:
- Apple Silicon GPU (MPS) for M1/M2 Macs
- NVIDIA CUDA GPUs  
- CPU fallback

## Key Implementation Details

- **Dead Agent Handling**: Filters out inactive agents dynamically
- **Coordinate Transformation**: Ego-centric coordinate system with heading alignment
- **Gradient Clipping**: Prevents gradient explosion with max norm of 5.0
- **Learning Rate Scheduling**: StepLR with decay factor 0.5 every 10 epochs
- **Memory Efficiency**: Batch processing with attention mechanisms

## License

This project is available under standard academic/research usage terms.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.
