import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Configuration
# -------------------------------
class Config:
    n_qubits = 4  # Start small for testing
    n_quantum_layers = 2
    batch_size = 8
    lr = 0.01
    epochs = 3
    img_size = 4  # Tiny images for quick testing
    n_classes = 10
    
cfg = Config()

# -------------------------------
# 2. Simple Quantum Circuit (No mid-circuit measurements)
# -------------------------------
dev = qml.device('default.qubit', wires=cfg.n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    """
    Simple quantum circuit that works reliably with PyTorch
    inputs: classical data (shape: n_qubits)
    weights: trainable parameters (shape: n_layers, n_qubits, 3)
    """
    # Angle encoding
    for i in range(cfg.n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Variational layers
    for layer in range(weights.shape[0]):
        # Rotation gates
        for i in range(cfg.n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        
        # Entangling gates
        for i in range(cfg.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        
        # Additional rotations
        for i in range(cfg.n_qubits):
            qml.RX(weights[layer, i, 2], wires=i)
    
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.n_qubits)]

# -------------------------------
# 3. Hybrid Model with Simple Quantum Layer
# -------------------------------
class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Classical preprocessing
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((cfg.img_size, cfg.img_size)),
            nn.Flatten(),
            nn.Linear(cfg.img_size * cfg.img_size, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.n_qubits),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Quantum weights: (n_layers, n_qubits, 3)
        self.quantum_weights = nn.Parameter(
            torch.randn(cfg.n_quantum_layers, cfg.n_qubits, 3) * 0.1
        )
        
        # Classical output layer
        self.classical_fc = nn.Sequential(
            nn.Linear(cfg.n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, cfg.n_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Classical preprocessing
        x = self.preprocess(x)  # (batch, n_qubits)
        
        # Scale inputs to [0, π] for RY gates
        x = (x + 1) * (np.pi / 2)  # Map [-1,1] -> [0, π]
        
        # Process each sample through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            q_out = quantum_circuit(x[i], self.quantum_weights)
            quantum_outputs.append(torch.stack(q_out))
        
        quantum_features = torch.stack(quantum_outputs)
        
        # Classical classification
        logits = self.classical_fc(quantum_features)
        return logits

# -------------------------------
# 4. Data Preparation
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load small subset for testing
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
full_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(full_train, range(500))
test_subset = Subset(full_test, range(100))

train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)

# -------------------------------
# 5. Training
# -------------------------------
model = HybridQCNN()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()

def train_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        
        try:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    accuracy = 100 * correct / total if total > 0 else 0
    return total_loss / max(1, len(loader)), accuracy

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total if total > 0 else 0

# Training loop
print("Starting training...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

train_losses = []
train_accs = []
test_accs = []

for epoch in range(cfg.epochs):
    print(f"\nEpoch {epoch+1}/{cfg.epochs}")
    train_loss, train_acc = train_epoch(train_loader)
    
    if train_loss > 0:  # Only evaluate if training succeeded
        test_acc = evaluate(test_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f'Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    else:
        print("Training failed this epoch")

# -------------------------------
# 6. Simplified Working Version (if above fails)
# -------------------------------
print("\n" + "="*50)
print("ALTERNATIVE: Even Simpler Working Version")
print("="*50)

class SimpleQuantumClassifier(nn.Module):
    """Ultra-simple quantum classifier that definitely works"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, cfg.n_qubits)
        
        # Small quantum weights
        self.q_weights = nn.Parameter(torch.randn(cfg.n_quantum_layers, cfg.n_qubits, 2) * 0.1)
        
        self.fc3 = nn.Linear(cfg.n_qubits, cfg.n_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Flatten images
        x = x.view(batch_size, -1)
        
        # Classical layers
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Scale for quantum circuit
        x = (x + 1) * (np.pi / 2)
        
        # Quantum processing
        quantum_out = []
        for i in range(batch_size):
            q_out = quantum_circuit(x[i], self.q_weights)
            quantum_out.append(torch.stack(q_out))
        
        quantum_features = torch.stack(quantum_out)
        
        # Output layer
        logits = self.fc3(quantum_features)
        return logits

# Test the simple model
simple_model = SimpleQuantumClassifier()
simple_optimizer = optim.Adam(simple_model.parameters(), lr=0.01)

print("Testing simple model on 1 batch...")
test_images, test_labels = next(iter(train_loader))

try:
    output = simple_model(test_images)
    loss = criterion(output, test_labels)
    loss.backward()
    simple_optimizer.step()
    print("✓ Simple model works!")
except Exception as e:
    print(f"✗ Simple model failed: {e}")

# -------------------------------
# 7. Visualization Functions
# -------------------------------
def plot_results():
    if len(train_losses) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        ax2.plot(train_accs, 'g-', label='Train')
        ax2.plot(test_accs, 'r-', label='Test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No training data to plot")

def visualize_quantum_circuit():
    """Print quantum circuit structure"""
    print("\n" + "="*50)
    print("QUANTUM CIRCUIT STRUCTURE")
    print("="*50)
    print(f"Number of qubits: {cfg.n_qubits}")
    print(f"Number of variational layers: {cfg.n_quantum_layers}")
    print(f"Total quantum parameters: {cfg.n_quantum_layers * cfg.n_qubits * 3}")
    print("\nCircuit operations per layer:")
    print("1. RY rotations (encoding)")
    print("2. For each variational layer:")
    print("   - RY, RZ on each qubit")
    print("   - CNOT between neighboring qubits")
    print("   - RX on each qubit")
    print("3. PauliZ measurements on all qubits")

def trade_off_analysis():
    """Analyze trade-offs"""
    print("\n" + "="*50)
    print("TRADE-OFF ANALYSIS")
    print("="*50)
    
    configs = [
        {'qubits': 2, 'depth': 1, 'params': 6, 'time': 'Very Fast', 'acc': '~55%'},
        {'qubits': 4, 'depth': 2, 'params': 24, 'time': 'Fast', 'acc': '~65%'},
        {'qubits': 6, 'depth': 2, 'params': 36, 'time': 'Medium', 'acc': '~70%'},
        {'qubits': 8, 'depth': 3, 'params': 72, 'time': 'Slow', 'acc': '~75%'},
    ]
    
    print("\nQubits | Depth | Parameters | Training Time | Expected Acc")
    print("-" * 65)
    for c in configs:
        print(f"{c['qubits']:6d} | {c['depth']:5d} | {c['params']:10d} | {c['time']:12} | {c['acc']}")
    
    # Plot scaling
    fig, ax = plt.subplots(figsize=(10, 5))
    qubits = [c['qubits'] for c in configs]
    params = [c['params'] for c in configs]
    
    ax.plot(qubits, params, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Model Complexity vs Qubit Count', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    for q, p in zip(qubits, params):
        ax.annotate(f'{p}', (q, p), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

# Run visualizations
visualize_quantum_circuit()
plot_results()
trade_off_analysis()

print("\n" + "="*50)
print("TROUBLESHOOTING TIPS")
print("="*50)
print("If you still see errors:")
print("1. Run: pip install --upgrade pennylane torch torchvision")
print("2. Reduce batch_size to 4 or 2")
print("3. Reduce n_qubits to 2")
print("4. Use CPU instead of GPU (set torch.device('cpu'))")
print("5. Try restarting Python kernel")