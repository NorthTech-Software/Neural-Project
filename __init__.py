"""
Data handlers for various input formats
"""

import numpy as np
from typing import Tuple, List, Union


class DataHandler:
    """Base class for handling data"""
    
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """Standardize data to mean=0, std=1"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        num_samples = len(X)
        num_train = int(num_samples * train_ratio)
        
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class CSVHandler(DataHandler):
    """Handler for CSV data"""
    
    @staticmethod
    def load_csv(filepath: str, has_header: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Load data from CSV file"""
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1 if has_header else 0)
        
        # Get column names if available
        if has_header:
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
        else:
            header = [f"feature_{i}" for i in range(data.shape[1])]
        
        return data, header


class ImageHandler(DataHandler):
    """Handler for image data"""
    
    @staticmethod
    def flatten_images(images: np.ndarray) -> np.ndarray:
        """Flatten image data for neural network"""
        num_images = images.shape[0]
        flattened = images.reshape(num_images, -1)
        return flattened.astype(np.float32) / 255.0  # Normalize to [0, 1]


class SequenceHandler(DataHandler):
    """Handler for sequential/time-series data"""
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int, 
                        step_size: int = 1) -> np.ndarray:
        """Create sequences from time-series data"""
        sequences = []
        for i in range(0, len(data) - sequence_length, step_size):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    @staticmethod
    def create_sliding_window(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window for time-series prediction"""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)
