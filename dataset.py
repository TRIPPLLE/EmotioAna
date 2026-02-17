import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from src.config import Config

class MovieOfflineDataset(Dataset):
    def __init__(self, csv_path, npy_path, filter_positive=True):
        """
        Args:
            csv_path (str): Path to the CSV file.
            npy_path (str): Path to the .npy file containing movie embeddings.
            filter_positive (bool): keep only positive interactions.
        """
        print(f"Loading CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        print(f"Loading Embeddings from {npy_path}...")
        self.movie_embeddings = np.load(npy_path) # Shape (num_movies, 768)
        
        # Preprocessing: Map user_response to reward signal if reward not fully reliable?
        # The CSV has 'reward'.
        # If filter_positive is True, keep only rewards > 0
        if filter_positive:
            original_len = len(self.df)
            self.df = self.df[self.df['reward'] > 0].reset_index(drop=True)
            print(f"Filtered dataset: {original_len} -> {len(self.df)} samples (Positive Rewards Only)")
            
        # Ensure mappings are correct
        # state_user_cluster: 0-9
        # state_last_movie: 0-NumMovies (index into embeddings)
        # action_recommended_movie: 0-NumMovies
        
        self.user_clusters = torch.tensor(self.df['state_user_cluster'].values, dtype=torch.long)
        self.last_movie_indices = torch.tensor(self.df['state_last_movie'].values, dtype=torch.long)
        self.actions = torch.tensor(self.df['action_recommended_movie'].values, dtype=torch.long)
        self.rewards = torch.tensor(self.df['reward'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # State Components
        user_cluster = self.user_clusters[idx]
        last_movie_idx = self.last_movie_indices[idx]
        
        # Get Movie Embedding (from numpy array, convert to tensor)
        last_movie_emb = torch.tensor(self.movie_embeddings[last_movie_idx], dtype=torch.float32)
        
        # Action
        action = self.actions[idx]
        
        # Reward (Weight for loss)
        reward = self.rewards[idx]
        
        return user_cluster, last_movie_emb, action, reward
