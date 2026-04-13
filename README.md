# Quantum-Classical Hybrid Image Classifier

A hybrid quantum-classical neural network for image classification using [PennyLane](https://pennylane.ai/) and [PyTorch](https://pytorch.org/). The model combines classical convolutional preprocessing with a variational quantum circuit to classify MNIST handwritten digits.

---

## Overview

This project implements two hybrid models:

1. **HybridQCNN** — A full hybrid model with classical preprocessing (adaptive pooling + linear layers) feeding into a variational quantum circuit, followed by a classical output head.
2. **SimpleQuantumClassifier** — A lightweight fallback model using a flat fully connected encoder before the quantum layer, useful for debugging or quick testing.

Both models use a shared variational quantum circuit built with PennyLane's `default.qubit` simulator.

---

## Architecture

```
Input Image (MNIST)
      │
      ▼
Classical Preprocessing
  AdaptiveAvgPool2d → Flatten → Linear → ReLU → Linear → Tanh
      │
      ▼
Quantum Circuit (PennyLane)
  Angle Encoding (RY) → Variational Layers (RY, RZ, CNOT, RX) → PauliZ Measurements
      │
      ▼
Classical Output Head
  Linear → ReLU → Linear → Logits (10 classes)
```

### Quantum Circuit Details

| Property | Value |
|---|---|
| Qubits | 4 |
| Variational layers | 2 |
| Gates per layer | RY, RZ (per qubit) + CNOT (neighbors) + RX (per qubit) |
| Measurement | PauliZ expectation on all qubits |
| Total quantum parameters | 24 |

---

## Requirements

```bash
pip install pennylane torch torchvision scikit-learn matplotlib numpy
```

Tested with:
- Python 3.8+
- PennyLane 0.35+
- PyTorch 2.0+

---

## Configuration

All hyperparameters are defined in the `Config` class at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `n_qubits` | 4 | Number of qubits in the quantum circuit |
| `n_quantum_layers` | 2 | Number of variational layers |
| `batch_size` | 8 | Training batch size |
| `lr` | 0.01 | Adam learning rate |
| `epochs` | 3 | Number of training epochs |
| `img_size` | 4 | Image resized to img_size × img_size |
| `n_classes` | 10 | Number of output classes (MNIST digits 0–9) |

---

## Dataset

Uses a small subset of **MNIST** for quick testing:
- **Training samples:** 500
- **Test samples:** 100
- Images are converted to grayscale and resized to 4×4 pixels.

The dataset is automatically downloaded to `./data/` on first run.

---

## Usage

Run the script directly:

```bash
python QImage1.py
```

This will:
1. Download MNIST data
2. Train `HybridQCNN` for 3 epochs
3. Run a quick forward/backward pass test on `SimpleQuantumClassifier`
4. Print quantum circuit structure and trade-off analysis
5. Display training loss and accuracy plots

---

## Outputs

- **Training logs** — Loss and accuracy printed per batch and per epoch
- **Loss curve** — Training loss over epochs
- **Accuracy curve** — Train vs. test accuracy over epochs
- **Trade-off table** — Expected accuracy and training time for different qubit/depth configurations

### Qubit Scaling Trade-offs

| Qubits | Depth | Parameters | Training Time | Expected Accuracy |
|---|---|---|---|---|
| 2 | 1 | 6 | Very Fast | ~55% |
| 4 | 2 | 24 | Fast | ~65% |
| 6 | 2 | 36 | Medium | ~70% |
| 8 | 3 | 72 | Slow | ~75% |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Import errors | `pip install --upgrade pennylane torch torchvision` |
| Out of memory / slow | Reduce `batch_size` to 4 or 2 |
| Quantum errors | Reduce `n_qubits` to 2 |
| GPU issues | Force CPU by setting `torch.device('cpu')` |
| Kernel crashes | Restart Python kernel and re-run |

---

## File Structure

```
.
├── QImage1.py       # Main script
├── README.md        # This file
└── data/            # MNIST dataset (auto-downloaded)
```

---

## Notes

- The quantum circuit runs on PennyLane's CPU simulator (`default.qubit`). For larger experiments, consider using `lightning.qubit` for faster simulation.
- Mid-circuit measurements are intentionally avoided for compatibility with PyTorch's autograd.
- Batch processing runs the quantum circuit sequentially per sample, which is the main bottleneck for training speed.
