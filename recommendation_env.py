import torch
import numpy as np
import random
from src.config import Config
from src.utils import get_reward, cosine_similarity

class MockUser:
    def __init__(self, num_movies, embedding_dim, device):
        self.device = device
        self.preference_vector = torch.randn(embedding_dim).to(device)
        self.preference_vector = self.preference_vector / torch.norm(self.preference_vector)
        self.current_emotion_idx = 0
        self.embedding_dim = embedding_dim
        
    def set_emotion(self, emotion_idx):
        self.current_emotion_idx = emotion_idx
        
    def react(self, movie_embedding, emotion_embedding):
        """
        Simulate user reaction to a recommended movie.
        Returns:
            user_response: dict
        """
        # Calculate alignment with preference and emotion
        pref_score = torch.dot(self.preference_vector, movie_embedding).item()
        emotion_score = torch.dot(emotion_embedding, movie_embedding).item()
        
        # Combined score (weighted)
        total_score = 0.6 * pref_score + 0.4 * emotion_score
        
        # Probabilistic response
        watch_prob = 1 / (1 + np.exp(-5 * (total_score - 0.2))) # Sigmoid-like
        
        response = {
            'watch_ratio': 0.0,
            'liked': 0,
            'skipped': 0
        }
        
        if random.random() < watch_prob:
            response['watch_ratio'] = random.uniform(0.5, 1.0)
            if total_score > 0.4:
                response['liked'] = 1
        else:
            response['watch_ratio'] = random.uniform(0.0, 0.2)
            response['skipped'] = 1
            
        return response

class RecommendationEnv:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.num_movies = Config.ACTION_DIM
        self.embedding_dim = Config.MOVIE_EMBEDDING_DIM
        self.state_dim = Config.STATE_DIM
        
        # Initialize Movie Catalog (Real Embeddings)
        try:
            # We assume the file is in the project root or accessible path. 
            # Given current structure, it's in d:/PORJECT/ai_rec/mov_n_emb.npy
            self.movie_embeddings = torch.tensor(np.load('mov_n_emb.npy', allow_pickle=True)).float().to(self.device)
            # Normalize
            self.movie_embeddings = self.movie_embeddings / torch.norm(self.movie_embeddings, dim=1, keepdim=True)
            self.num_movies = self.movie_embeddings.shape[0]
            if self.num_movies != Config.ACTION_DIM:
                print(f"Warning: Config.ACTION_DIM ({Config.ACTION_DIM}) != Data ({self.num_movies})")
                
        except Exception as e:
            print(f"⚠️ Error loading mov_n_emb.npy: {e}")
            print("   Using RANDOM embeddings for simulation.")
            self.movie_embeddings = torch.randn(self.num_movies, self.embedding_dim).to(self.device)
            self.movie_embeddings = self.movie_embeddings / torch.norm(self.movie_embeddings, dim=1, keepdim=True)
            
        # Initialize Mock User
        # Preference dimension must match movie embedding dimension (768)
        self.user = MockUser(self.num_movies, self.embedding_dim, self.device)
        
        # Emotion Embeddings (Simulating Gemini Output)
        # In a real scenario, this would be the embedding of the text "Happy", "Sad", etc.
        # We generate random 768-dim vectors to represent these 'mean' embeddings.
        self.emotion_embeddings = torch.randn(Config.NUM_EMOTIONS, Config.EMOTION_DIM).to(self.device)
        self.emotion_embeddings = self.emotion_embeddings / torch.norm(self.emotion_embeddings, dim=1, keepdim=True)
        
        self.current_step = 0
        
    def reset(self):
        """
        Start a new episode (new user session).
        """
        # Randomize user preference for new session
        self.user = MockUser(self.num_movies, self.embedding_dim, self.device)
        
        # Random initial emotion
        initial_emotion = random.randint(0, Config.NUM_EMOTIONS - 1)
        self.user.set_emotion(initial_emotion)
        
        self.current_step = 0
        return self._get_observation()
        
    def step(self, action):
        """
        Execute action and return transition.
        Args:
            action: int (index of recommended movie)
        """
        self.current_step += 1
        
        # Get movie embedding (Already on GPU)
        movie_embedding = self.movie_embeddings[action]
        emotion_embedding = self.emotion_embeddings[self.user.current_emotion_idx]
        
        # Simulate User Response
        user_response = self.user.react(movie_embedding, emotion_embedding)
        
        # Calculate Reward
        # Note: get_reward uses cosine_similarity from utils.py which uses torch function
        # We need to make sure inputs are tensors on same device.
        reward = get_reward(action, None, user_response, emotion_embedding, movie_embedding, Config)
        
        # Transition Emotion (Simple Markov: 20% chance to change)
        if random.random() < 0.2:
            new_emotion = random.randint(0, Config.NUM_EMOTIONS - 1)
            self.user.set_emotion(new_emotion)
            
        done = self.current_step >= Config.MAX_STEPS_PER_EPISODE
        next_state = self._get_observation()
        
        return next_state, reward, done, {}
        
    def _get_observation(self):
        """
        Construct state vector S_t = [E_t || P_u]
        """
        emotion_vec = self.emotion_embeddings[self.user.current_emotion_idx]
        pref_vec = self.user.preference_vector
        
        # Concatenate
        state = torch.cat([emotion_vec, pref_vec])
        return state

class VectorizedRecommendationEnv:
    def __init__(self, num_envs=256, device=torch.device('cpu')):
        self.device = device
        self.num_envs = num_envs
        self.num_movies = Config.ACTION_DIM
        self.embedding_dim = Config.MOVIE_EMBEDDING_DIM
        self.state_dim = Config.STATE_DIM
        
        # Load Embeddings (Shared across all envs)
        try:
            self.movie_embeddings = torch.tensor(np.load('mov_n_emb.npy', allow_pickle=True)).float().to(self.device)
            self.movie_embeddings = self.movie_embeddings / torch.norm(self.movie_embeddings, dim=1, keepdim=True)
        except Exception:
            self.movie_embeddings = torch.randn(self.num_movies, self.embedding_dim).to(self.device)
            self.movie_embeddings = self.movie_embeddings / torch.norm(self.movie_embeddings, dim=1, keepdim=True)

        self.emotion_embeddings = torch.randn(Config.NUM_EMOTIONS, Config.EMOTION_DIM).to(self.device)
        self.emotion_embeddings = self.emotion_embeddings / torch.norm(self.emotion_embeddings, dim=1, keepdim=True)
        
        # Batch User State
        # CRITICAL FIX: Sample preferences from REAL movie embeddings + Noise
        # This ensures users actually "like" something in the dataset.
        indices = torch.randint(0, self.num_movies, (num_envs,))
        self.user_preferences = self.movie_embeddings[indices] + 0.1 * torch.randn(num_envs, self.embedding_dim).to(device)
        self.user_preferences = self.user_preferences / torch.norm(self.user_preferences, dim=1, keepdim=True)
        
        self.current_emotion_indices = torch.randint(0, Config.NUM_EMOTIONS, (num_envs,)).to(device)
        self.current_steps = torch.zeros(num_envs).to(device)
        
    def reset(self):
        # Sample NEW preferences from real data
        indices = torch.randint(0, self.num_movies, (self.num_envs,))
        self.user_preferences = self.movie_embeddings[indices] + 0.1 * torch.randn(self.num_envs, self.embedding_dim).to(self.device)
        self.user_preferences = self.user_preferences / torch.norm(self.user_preferences, dim=1, keepdim=True)
        
        self.current_emotion_indices = torch.randint(0, Config.NUM_EMOTIONS, (self.num_envs,)).to(self.device)
        self.current_steps = torch.zeros(self.num_envs).to(self.device)
        
        return self._get_observation()
    
    def step(self, actions):
        """
        actions: (num_envs,) indices
        """
        self.current_steps += 1
        
        # Gather Movie Embeddings
        # actions is (B,) -> movie_embeddings is (N, D) -> selected is (B, D)
        selected_movies = self.movie_embeddings[actions]
        
        # Gather Emotion Embeddings
        current_emotions = self.emotion_embeddings[self.current_emotion_indices]
        
        # Vectorized User Reaction
        # Preferences: (B, D), Movies: (B, D) -> Dot: (B,)
        pref_scores = (self.user_preferences * selected_movies).sum(dim=1)
        emotion_scores = (current_emotions * selected_movies).sum(dim=1)
        
        total_scores = 0.6 * pref_scores + 0.4 * emotion_scores
        watch_probs = 1 / (1 + torch.exp(-5 * (total_scores - 0.2)))
        
        # Random Outcomes
        rand_vals = torch.rand(self.num_envs).to(self.device)
        watched = rand_vals < watch_probs
        
        # Rewards
        rewards = torch.zeros(self.num_envs).to(self.device)
        
        # If watched
        watch_ratios = torch.zeros(self.num_envs).to(self.device)
        watch_ratios[watched] = torch.empty(watched.sum()).uniform_(0.5, 1.0).to(self.device)
        
        likes = torch.zeros(self.num_envs).to(self.device)
        likes[watched & (total_scores > 0.4)] = 1.0
        
        skips = torch.zeros(self.num_envs).to(self.device)
        skips[~watched] = 1.0
        
        # Emotion Match Reward
        emotion_matches = torch.nn.functional.cosine_similarity(current_emotions, selected_movies)
        
        rewards += Config.W_WATCH_RATIO * watch_ratios
        rewards += Config.W_LIKE * likes
        rewards -= Config.W_SKIP * skips
        rewards += Config.W_EMOTION_MATCH * emotion_matches
        
        # Update Emotions (Markov)
        change_emotion = torch.rand(self.num_envs).to(self.device) < 0.2
        new_emotions = torch.randint(0, Config.NUM_EMOTIONS, (change_emotion.sum(),)).to(self.device)
        self.current_emotion_indices[change_emotion] = new_emotions
        
        dones = self.current_steps >= Config.MAX_STEPS_PER_EPISODE
        # Auto-reset done envs? Usually in vec env we strictly return dones.
        # But for PPO loop we usually handle masking. 
        # For simplicity, we just return dones.
        
        # If any done, we should theoretically reset them, but simpler to just end batch.
        # OR: PPO implementation usually assumes masking.
        # Let's just return next state and dones. PPO script handles rollout.
        
        # CRITICAL: For training loop simplification, if all done, we reset?
        # No, standard is infinite horizon or manual reset.
        # We will reset individual envs? 
        # For now, let's just let the caller handle full reset if all done.
        
        return self._get_observation(), rewards, dones, {}
        
    def _get_observation(self):
        emotion_vecs = self.emotion_embeddings[self.current_emotion_indices] # (B, D)
        state = torch.cat([emotion_vecs, self.user_preferences], dim=1) # (B, 2D)
        return state
