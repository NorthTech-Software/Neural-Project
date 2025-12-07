"""
Neuron - Individual neural unit with learning capabilities
"""

import numpy as np
from typing import Callable, Optional


class Neuron:
    """A single neuron that processes inputs and learns from feedback"""
    
    def __init__(self, input_size: int, activation: str = 'relu'):
        """
        Initialize a neuron
        
        Args:
            input_size: Number of input connections
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.input_size = input_size
        self.activation_type = activation
        
        # Initialize weights with Xavier initialization
        limit = np.sqrt(6.0 / (input_size + 1))
        self.weights = np.random.uniform(-limit, limit, input_size)
        self.bias = 0.0
        
        # Learning rate (adaptive)
        self.learning_rate = 0.01
        self.learning_history = []
        
    def _activate(self, x: float) -> float:
        """Apply activation function"""
        if self.activation_type == 'relu':
            return max(0.0, x)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        return x
    
    def _activate_derivative(self, x: float) -> float:
        """Derivative of activation function for backprop"""
        if self.activation_type == 'relu':
            return 1.0 if x > 0 else 0.0
        elif self.activation_type == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        elif self.activation_type == 'tanh':
            return 1.0 - np.tanh(x) ** 2
        return 1.0
    
    def forward(self, inputs: np.ndarray) -> float:
        """Process inputs and return output"""
        z = np.dot(inputs, self.weights) + self.bias
        self.last_input = inputs
        self.last_z = z
        self.last_output = self._activate(z)
        return self.last_output
    
    def backward(self, error: float) -> np.ndarray:
        """
        Learn from error using backpropagation
        
        Args:
            error: Output error signal
            
        Returns:
            Input error gradient for upstream neurons
        """
        # Calculate gradient
        d_activation = self._activate_derivative(self.last_z)
        delta = error * d_activation
        
        # Update weights and bias
        self.weights -= self.learning_rate * delta * self.last_input
        self.bias -= self.learning_rate * delta
        
        # Adapt learning rate based on performance
        self.learning_history.append(abs(error))
        if len(self.learning_history) > 10:
            recent_avg = np.mean(self.learning_history[-10:])
            older_avg = np.mean(self.learning_history[-20:-10])
            if recent_avg < older_avg * 0.99:  # Improving
                self.learning_rate *= 1.01
            elif recent_avg > older_avg * 1.01:  # Getting worse
                self.learning_rate *= 0.99
        
        # Return gradient for previous layer
        return delta * self.weights
