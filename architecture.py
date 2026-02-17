import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class GeminiEmotionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GeminiEmotionEncoder, self).__init__()
        # Optional: Project 768 -> 64 if we wanted to keep model small, 
        # but Config says State is 768+64, so we pass it through directly or process it.
        # Let's add a small dense layer to "adapt" the Gemini embedding to our task space.
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, emotion_embedding):
        """
        Args:
            emotion_embedding: Tensor of shape (batch, 768)
        Returns:
            Tensor of shape (batch, hidden_dim)
        """
        return self.adapter(emotion_embedding)

class UserPreferenceEncoder(nn.Module):
    def __init__(self, num_movies, embedding_dim):
        super(UserPreferenceEncoder, self).__init__()
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        
    def forward(self, history_indices, history_ratings):
        """
        Args:
            history_indices: Tensor of shape (batch_size, history_len)
            history_ratings: Tensor of shape (batch_size, history_len) - e.g. +1/-1
        Returns:
            Tensor of shape (batch_size, embedding_dim) representing P_u
        """
        # Element-wise multiplication of movie embeddings and ratings
        embedded_movies = self.movie_embeddings(history_indices) # (batch, len, dim)
        weighted_embeddings = embedded_movies * history_ratings.unsqueeze(-1) # (batch, len, dim)
        
        # Sum over history (simplified version)
        user_preference = torch.sum(weighted_embeddings, dim=1) # (batch, dim)
        return user_preference

class NeuralScoring(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(NeuralScoring, self).__init__()
        # Gating network g(S_t) -> alpha in (0,1)
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state_vector, s_int, s_taste):
        """
        Args:
            state_vector: (batch, state_dim)
            s_int: (batch, num_candidates) - e.g. cosine similarity with emotion
            s_taste: (batch, num_candidates) - e.g. cosine similarity with preference
        Returns:
            final_scores: (batch, num_candidates)
            alpha: (batch, 1) generic gate value
        """
        alpha = self.gate_net(state_vector) # (batch, 1)
        final_scores = alpha * s_int + (1 - alpha) * s_taste
        return final_scores, alpha

class ActorCriticNetwork(nn.Module):
    def __init__(self, user_cluster_dim, user_cluster_embedding_dim, movie_embedding_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # User Cluster Embedding Layer
        self.user_cluster_embedding = nn.Embedding(user_cluster_dim, user_cluster_embedding_dim)
        
        # Shared feature extractor
        # Input: [UserClusterEmb (32) + LastMovieEmb (768)]
        input_dim = user_cluster_embedding_dim + movie_embedding_dim
        
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # Output raw logits, softmax will be applied later
        )
        
        # Critic Head (Value) - Not strictly needed for pure Behavior Cloning but good to have
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, user_cluster_idx, last_movie_emb):
        """
        Args:
            user_cluster_idx: Tensor (batch,)
            last_movie_emb: Tensor (batch, 768)
        """
        # Embed User Cluster
        cluster_emb = self.user_cluster_embedding(user_cluster_idx) # (batch, 32)
        
        # Concatenate State
        state = torch.cat([cluster_emb, last_movie_emb], dim=1) # (batch, 32+768)
        
        shared_features = self.shared_layer(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_logits, state_value

    def act(self, state):
        """Sample action from policy"""
        action_logits, _ = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # Check if batch (dim > 0 and size > 1) or single item
        if action.dim() == 0 or (action.dim() == 1 and action.size(0) == 1):
             # Single environment case (legacy support if needed, but safer to return tensor)
             # But legacy code expects item.
             # If we are strictly in vectorized mode, we want tensor.
             # Let's just return tensor. The caller (train.py) handles it.
             # Wait, legacy demo.py expects item?
             # demo.py calls run_demo. model.act is NOT called in demo.py!
             # demo.py uses model(state) directly.
             # So safely return tensor.
             pass
             
        return action, dist.log_prob(action).detach()
    
    def evaluate(self, state, action):
        """Evaluate action for PPO update"""
        action_logits, state_value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_value, dist_entropy
