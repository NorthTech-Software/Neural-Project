"""
Basic learning example demonstrating The Neural Project
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from neural_core.network import NeuralNetwork
from adapters import DataHandler


def generate_synthetic_data(num_samples: int = 200) -> tuple:
    """Generate synthetic classification data"""
    # Create two class data
    np.random.seed(42)
    
    # Class 0: centered around (2, 2)
    class0_X = np.random.randn(num_samples // 2, 10) + 2
    class0_y = np.zeros((num_samples // 2, 2))
    class0_y[:, 0] = 1
    
    # Class 1: centered around (-2, -2)
    class1_X = np.random.randn(num_samples // 2, 10) - 2
    class1_y = np.zeros((num_samples // 2, 2))
    class1_y[:, 1] = 1
    
    # Combine
    X = np.vstack([class0_X, class1_X])
    y = np.vstack([class0_y, class1_y])
    
    # Normalize
    X = DataHandler.normalize(X)
    
    return X, y


def main():
    print("=" * 60)
    print("THE NEURAL PROJECT - Learning Demo")
    print("A system that learns and grows like a brain")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(200)
    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    print("\n2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = DataHandler.split_data(X, y, train_ratio=0.8)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create network
    print("\n3. Creating neural network...")
    network = NeuralNetwork(
        input_size=10,
        hidden_layers=[16, 8],
        output_size=2,
        activation='relu'
    )
    
    network_info = network.get_network_info()
    print(f"   Network architecture: {network_info['layer_sizes']}")
    print(f"   Total neurons: {network_info['total_neurons']}")
    
    # Train network
    print("\n4. Training network (learning and growing)...")
    print("   This may take a moment while the network learns...\n")
    network.train(X_train, y_train, epochs=100, batch_size=16)
    
    # Evaluate
    print("\n5. Evaluating network on test data...")
    test_loss, test_accuracy = network.evaluate(X_test, y_test)
    print(f"   Test Loss: {test_loss:.6f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    # Print final network info
    print("\n6. Final Network Status:")
    final_info = network.get_network_info()
    print(f"   Total layers: {final_info['num_layers']}")
    print(f"   Final layer sizes: {final_info['layer_sizes']}")
    print(f"   Total neurons: {final_info['total_neurons']}")
    print(f"   Growth events: {final_info['growth_events']}")
    
    # Print memory consolidation
    print(f"\n7. Memory System Status:")
    print(f"   Short-term memories: {len(network.memory.short_term.recall())}")
    print(f"   Long-term patterns: {len(network.memory.long_term.patterns)}")
    print(f"   Episodes recorded: {len(network.memory.episodic.episodes)}")
    
    # Make predictions
    print("\n8. Making predictions on new data...")
    sample_predictions = network.predict(X_test[:5])
    for i, pred in enumerate(sample_predictions):
        print(f"   Sample {i+1}: Class {np.argmax(pred)} (confidence: {np.max(pred):.4f})")
    
    print("\n" + "=" * 60)
    print("Demo completed! The Neural Project is learning and growing!")
    print("=" * 60)


if __name__ == "__main__":
    main()
