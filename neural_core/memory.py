"""
Memory systems for storing and retrieving learned patterns
"""

import numpy as np
from typing import List, Dict, Any
from collections import deque


class ShortTermMemory:
    """Temporary working memory for current processing"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def store(self, data: Any):
        """Add data to short-term memory"""
        self.memory.append(data)
    
    def recall(self) -> List[Any]:
        """Retrieve all data in short-term memory"""
        return list(self.memory)
    
    def clear(self):
        """Clear short-term memory"""
        self.memory.clear()


class LongTermMemory:
    """Persistent memory for learned patterns and weights"""
    
    def __init__(self):
        self.patterns: Dict[str, np.ndarray] = {}
        self.associations: Dict[str, List[str]] = {}
    
    def store_pattern(self, name: str, pattern: np.ndarray):
        """Store a learned pattern"""
        self.patterns[name] = pattern.copy()
    
    def retrieve_pattern(self, name: str) -> np.ndarray:
        """Retrieve a stored pattern"""
        return self.patterns.get(name, None)
    
    def create_association(self, pattern1: str, pattern2: str):
        """Create association between patterns"""
        if pattern1 not in self.associations:
            self.associations[pattern1] = []
        if pattern2 not in self.associations[pattern1]:
            self.associations[pattern1].append(pattern2)
    
    def get_associated_patterns(self, pattern: str) -> List[str]:
        """Get patterns associated with a given pattern"""
        return self.associations.get(pattern, [])


class EpisodicMemory:
    """Memory of experiences and events"""
    
    def __init__(self, max_episodes: int = 1000):
        self.episodes: deque = deque(maxlen=max_episodes)
        self.episode_counter = 0
    
    def record_episode(self, data: Dict[str, Any]):
        """Record an experience/episode"""
        episode = {
            'id': self.episode_counter,
            'data': data,
            'importance': 1.0
        }
        self.episodes.append(episode)
        self.episode_counter += 1
        return episode['id']
    
    def recall_episodes(self, n: int = 10) -> List[Dict]:
        """Recall recent episodes"""
        return list(self.episodes)[-n:]
    
    def recall_important_episodes(self, n: int = 5) -> List[Dict]:
        """Recall most important episodes"""
        sorted_episodes = sorted(
            self.episodes, 
            key=lambda x: x['importance'], 
            reverse=True
        )
        return sorted_episodes[:n]
    
    def update_episode_importance(self, episode_id: int, importance: float):
        """Update importance score of an episode"""
        for episode in self.episodes:
            if episode['id'] == episode_id:
                episode['importance'] = importance
                break


class MemorySystem:
    """Integrated memory system combining all memory types"""
    
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()
    
    def consolidate_memories(self):
        """Transfer important short-term memories to long-term storage"""
        recent = self.short_term.recall()
        for item in recent:
            if isinstance(item, np.ndarray):
                self.long_term.store_pattern(f"pattern_{id(item)}", item)
