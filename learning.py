"""
Learning algorithms for neural network adaptation
"""

import numpy as np
from typing import Tuple, List


class BackpropagationLearning:
    """Standard backpropagation learning algorithm"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error loss"""
        return np.mean((predictions - targets) ** 2)
    
    def compute_gradients(self, error: np.ndarray) -> np.ndarray:
        """Compute gradients from error"""
        return error


class AdaptiveLearning:
    """Learning algorithm that adapts based on performance"""
    
    def __init__(self, initial_learning_rate: float = 0.01):
        self.learning_rate = initial_learning_rate
        self.loss_history: List[float] = []
        self.improvement_threshold = 0.999
    
    def update_learning_rate(self, current_loss: float):
        """Adapt learning rate based on loss trends"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) > 2:
            recent_loss = self.loss_history[-1]
            previous_loss = self.loss_history[-2]
            
            # If loss is improving (decreasing)
            if recent_loss < previous_loss * self.improvement_threshold:
                self.learning_rate *= 1.01  # Increase learning rate
            # If loss is getting worse
            elif recent_loss > previous_loss * 1.01:
                self.learning_rate *= 0.95  # Decrease learning rate
    
    def get_learning_rate(self) -> float:
        """Get current adaptive learning rate"""
        return self.learning_rate


class MetaLearning:
    """Learning to learn - improves learning strategy over time"""
    
    def __init__(self):
        self.strategy_performance: dict = {}
        self.best_strategy = None
    
    def evaluate_strategy(self, strategy_name: str, performance: float):
        """Evaluate how well a learning strategy performs"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(performance)
        
        # Update best strategy
        avg_performance = np.mean(self.strategy_performance[strategy_name])
        if self.best_strategy is None:
            self.best_strategy = (strategy_name, avg_performance)
        elif avg_performance > self.best_strategy[1]:
            self.best_strategy = (strategy_name, avg_performance)
    
    def get_best_strategy(self) -> Tuple[str, float]:
        """Get the best performing learning strategy"""
        return self.best_strategy if self.best_strategy else ("default", 0.0)


class ReinforcementLearning:
    """Learning through rewards and penalties"""
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.99):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  # Gamma - importance of future rewards
        self.q_values: dict = {}
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value based on reward"""
        state_action = (state, action)
        
        if state_action not in self.q_values:
            self.q_values[state_action] = 0.0
        
        # Q-learning update: Q(s,a) = Q(s,a) + lr * (r + gamma * max_Q(s') - Q(s,a))
        next_state_value = max(
            [self.q_values.get((next_state, a), 0.0) for a in ['improve', 'maintain', 'explore']],
            default=0.0
        )
        
        self.q_values[state_action] += self.learning_rate * (
            reward + self.discount_factor * next_state_value - self.q_values[state_action]
        )
    
    def get_best_action(self, state: str, actions: List[str]) -> str:
        """Get action with highest Q-value for a state"""
        q_values_for_state = {
            action: self.q_values.get((state, action), 0.0) 
            for action in actions
        }
        return max(q_values_for_state, key=q_values_for_state.get)
