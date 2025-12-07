"""
Neural Network - Full adaptive neural network architecture
"""

import numpy as np
from typing import List, Tuple, Optional
from neural_core.neuron import Neuron
from neural_core.learning import AdaptiveLearning, MetaLearning
from neural_core.memory import MemorySystem


class Layer:
    """A layer of neurons"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
        self.activation = activation
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Process inputs through all neurons in layer"""
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, errors: np.ndarray) -> np.ndarray:
        """Backpropagate errors through layer"""
        input_errors = np.zeros(self.neurons[0].input_size)
        for neuron, error in zip(self.neurons, errors):
            input_errors += neuron.backward(error)
        return input_errors


class NeuralNetwork:
    """Adaptive neural network that learns and grows"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 activation: str = 'relu'):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output neurons
            activation: Activation function to use
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layers: List[Layer] = []
        
        for i in range(len(layer_sizes) - 1):
            # Last layer uses sigmoid for output
            layer_activation = 'sigmoid' if i == len(layer_sizes) - 2 else activation
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], layer_activation))
        
        # Learning systems
        self.adaptive_learning = AdaptiveLearning()
        self.meta_learning = MetaLearning()
        self.memory = MemorySystem()
        
        # Training history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []
        self.growth_events: List[dict] = []
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        x = inputs.copy()
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, output_error: np.ndarray) -> None:
        """Backward pass through network"""
        error = output_error.copy()
        for layer in reversed(self.layers):
            error = layer.backward(error)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error"""
        return np.mean((predictions - targets) ** 2)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train the network
        
        Args:
            X: Input data
            y: Target labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        num_samples = len(X)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                batch_loss = 0.0
                for sample_X, sample_y in zip(batch_X, batch_y):
                    # Forward pass
                    prediction = self.forward(sample_X)
                    
                    # Compute loss
                    loss = self.compute_loss(prediction, sample_y)
                    batch_loss += loss
                    
                    # Backward pass
                    output_error = prediction - sample_y
                    self.backward(output_error)
                
                epoch_loss += batch_loss / len(batch_X)
            
            # Average loss for epoch
            epoch_loss /= (num_samples // batch_size)
            self.loss_history.append(epoch_loss)
            
            # Adapt learning rate
            self.adaptive_learning.update_learning_rate(epoch_loss)
            
            # Store memory of training
            self.memory.short_term.store({
                'epoch': epoch,
                'loss': epoch_loss,
                'learning_rate': self.adaptive_learning.learning_rate
            })
            
            # Check for growth opportunities
            if epoch % 20 == 0:
                self._evaluate_growth()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}, "
                      f"LR: {self.adaptive_learning.learning_rate:.6f}")
        
        # Consolidate memories
        self.memory.consolidate_memories()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return np.array([self.forward(sample) for sample in X])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate network performance
        
        Returns:
            Tuple of (loss, accuracy)
        """
        predictions = self.predict(X)
        loss = self.compute_loss(predictions, y)
        
        # Calculate accuracy (for classification)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        
        self.accuracy_history.append(accuracy)
        return loss, accuracy
    
    def _evaluate_growth(self):
        """Evaluate if network should grow (add neurons/layers)"""
        if len(self.loss_history) < 2:
            return
        
        recent_improvement = (
            self.loss_history[-20] - self.loss_history[-1]
        ) / (self.loss_history[-20] + 1e-8)
        
        # If improvement is slowing, consider growing
        if recent_improvement < 0.01:
            self._grow_network()
    
    def _grow_network(self):
        """Add neurons to hidden layers to increase capacity"""
        # Add neurons to largest hidden layer
        if len(self.layers) > 2:
            # Find largest hidden layer
            hidden_layers = self.layers[:-1]
            largest_idx = 0
            largest_size = len(self.layers[largest_idx].neurons)
            
            for idx, layer in enumerate(hidden_layers):
                if len(layer.neurons) > largest_size:
                    largest_idx = idx
                    largest_size = len(layer.neurons)
            
            # Add neuron to this layer
            layer = self.layers[largest_idx]
            input_size = len(self.layers[largest_idx - 1].neurons) if largest_idx > 0 else self.input_size
            new_neuron = Neuron(input_size, layer.activation)
            layer.neurons.append(new_neuron)
            
            self.growth_events.append({
                'epoch': len(self.loss_history),
                'layer': largest_idx,
                'new_size': len(layer.neurons)
            })
            
            print(f"Network grew! Added neuron to layer {largest_idx}. "
                  f"New layer size: {len(layer.neurons)}")
    
    def get_network_info(self) -> dict:
        """Get current network information"""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'num_layers': len(self.layers),
            'layer_sizes': [len(layer.neurons) for layer in self.layers],
            'total_neurons': sum(len(layer.neurons) for layer in self.layers),
            'growth_events': len(self.growth_events),
            'current_loss': self.loss_history[-1] if self.loss_history else None,
            'best_accuracy': max(self.accuracy_history) if self.accuracy_history else None
        }
