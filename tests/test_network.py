"""
Tests for neural network module
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')

from neural_core.network import Layer, NeuralNetwork
from neural_core.neuron import Neuron


class TestLayer(unittest.TestCase):
    """Test layer functionality"""
    
    def test_layer_creation(self):
        """Test layer initialization"""
        layer = Layer(input_size=10, output_size=5, activation='relu')
        self.assertEqual(len(layer.neurons), 5)
    
    def test_layer_forward(self):
        """Test forward pass through layer"""
        layer = Layer(input_size=10, output_size=5, activation='relu')
        inputs = np.random.randn(10)
        outputs = layer.forward(inputs)
        self.assertEqual(len(outputs), 5)
    
    def test_layer_backward(self):
        """Test backward pass through layer"""
        layer = Layer(input_size=10, output_size=5, activation='relu')
        inputs = np.random.randn(10)
        layer.forward(inputs)
        
        errors = np.random.randn(5)
        input_errors = layer.backward(errors)
        self.assertEqual(len(input_errors), 10)


class TestNetworkGrowth(unittest.TestCase):
    """Test network growth mechanisms"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.network = NeuralNetwork(
            input_size=5,
            hidden_layers=[8],
            output_size=2
        )
    
    def test_network_growth_capacity(self):
        """Test network can grow neurons"""
        initial_neurons = sum(len(layer.neurons) for layer in self.network.layers)
        
        # Trigger growth by running evaluation
        self.network._evaluate_growth()
        
        # Network should remain stable or grow
        final_neurons = sum(len(layer.neurons) for layer in self.network.layers)
        self.assertGreaterEqual(final_neurons, initial_neurons)


if __name__ == '__main__':
    unittest.main()
