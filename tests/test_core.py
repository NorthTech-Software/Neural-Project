"""
Unit tests for The Neural Project
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')

from neural_core.neuron import Neuron
from neural_core.network import NeuralNetwork
from neural_core.memory import MemorySystem, ShortTermMemory, LongTermMemory


class TestNeuron(unittest.TestCase):
    """Test individual neuron functionality"""
    
    def test_neuron_creation(self):
        """Test neuron initialization"""
        neuron = Neuron(5, activation='relu')
        self.assertEqual(neuron.input_size, 5)
        self.assertEqual(neuron.activation_type, 'relu')
    
    def test_neuron_forward(self):
        """Test neuron forward pass"""
        neuron = Neuron(3, activation='relu')
        inputs = np.array([1.0, 2.0, 3.0])
        output = neuron.forward(inputs)
        self.assertIsInstance(output, (float, np.floating))
    
    def test_neuron_learning(self):
        """Test neuron learns from errors"""
        neuron = Neuron(3, activation='relu')
        inputs = np.array([1.0, 2.0, 3.0])
        initial_weights = neuron.weights.copy()
        
        # Forward and backward
        neuron.forward(inputs)
        neuron.backward(0.5)
        
        # Weights should have changed
        self.assertFalse(np.allclose(initial_weights, neuron.weights))


class TestNeuralNetwork(unittest.TestCase):
    """Test neural network functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.network = NeuralNetwork(
            input_size=4,
            hidden_layers=[8, 4],
            output_size=2,
            activation='relu'
        )
    
    def test_network_creation(self):
        """Test network initialization"""
        self.assertEqual(self.network.input_size, 4)
        self.assertEqual(self.network.output_size, 2)
        self.assertEqual(len(self.network.layers), 3)  # 2 hidden + 1 output
    
    def test_forward_pass(self):
        """Test forward pass through network"""
        inputs = np.random.randn(4)
        output = self.network.forward(inputs)
        self.assertEqual(output.shape, (2,))
    
    def test_training(self):
        """Test network training"""
        X = np.random.randn(50, 4)
        y = np.eye(2)[np.random.randint(0, 2, 50)]
        
        initial_loss = self.network.loss_history
        self.network.train(X, y, epochs=10, batch_size=10)
        
        # Should have loss history
        self.assertGreater(len(self.network.loss_history), 0)
        # Loss should change
        self.assertEqual(len(self.network.loss_history), 10)
    
    def test_prediction(self):
        """Test network predictions"""
        X = np.random.randn(10, 4)
        predictions = self.network.predict(X)
        self.assertEqual(predictions.shape, (10, 2))


class TestMemory(unittest.TestCase):
    """Test memory systems"""
    
    def test_short_term_memory(self):
        """Test short-term memory storage"""
        stm = ShortTermMemory(capacity=10)
        stm.store("data1")
        stm.store("data2")
        
        recalled = stm.recall()
        self.assertEqual(len(recalled), 2)
        self.assertIn("data1", recalled)
    
    def test_long_term_memory(self):
        """Test long-term memory pattern storage"""
        ltm = LongTermMemory()
        pattern = np.array([1, 2, 3, 4])
        ltm.store_pattern("pattern1", pattern)
        
        retrieved = ltm.retrieve_pattern("pattern1")
        np.testing.assert_array_equal(retrieved, pattern)
    
    def test_memory_system_integration(self):
        """Test integrated memory system"""
        memory = MemorySystem()
        
        # Store in short-term
        memory.short_term.store({"epoch": 1, "loss": 0.5})
        
        # Store pattern in long-term
        pattern = np.array([1, 2, 3])
        memory.long_term.store_pattern("test_pattern", pattern)
        
        # Record episode
        episode_id = memory.episodic.record_episode({"event": "test"})
        
        # Verify storage
        self.assertEqual(len(memory.short_term.recall()), 1)
        self.assertIsNotNone(memory.long_term.retrieve_pattern("test_pattern"))
        self.assertEqual(len(memory.episodic.episodes), 1)


if __name__ == '__main__':
    unittest.main()
