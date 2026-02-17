import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import Config
from src.data.dataset import MovieOfflineDataset
from src.models.architecture import ActorCriticNetwork
import os
import time

def train_offline():
    print(f"🚀 Starting Offline Training on {Config.DEVICE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    
    # Files
    # Config.DATA_PATH is "code/data". We need to go up two levels if we are in code/data? 
    # No, we assume we run from project root "d:/PORJECT/ai_rec". 
    # The files are in "d:/PORJECT/ai_rec".
    # So we can just use the filenames or ".".
    csv_path = "rl_movie_recommendation_100k.csv"
    csv_path = "rl_movie_recommendation_100k.csv"
    
    if os.path.exists("mov_n_emb_v3.npy"):
        npy_path = "mov_n_emb_v3.npy"
        print(f"🚀 Using Local GPU Embeddings (v3): {npy_path}")
    elif os.path.exists("mov_n_emb_v2.npy"):
        npy_path = "mov_n_emb_v2.npy"
        print(f"✨ Using Regenerated Embeddings (v2): {npy_path}")
    else:
        npy_path = "mov_n_emb.npy"
        print(f"⚠️ Using Original Embeddings (Legacy): {npy_path}")
    
    # Dataset & DataLoader
    # Dataset & DataLoader
    dataset = MovieOfflineDataset(csv_path, npy_path, filter_positive=True)
    # Optimization: num_workers=4, pin_memory=True for faster CPU->GPU transfer
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = ActorCriticNetwork(
        user_cluster_dim=Config.USER_CLUSTER_DIM,
        user_cluster_embedding_dim=Config.USER_CLUSTER_EMBEDDING_DIM,
        movie_embedding_dim=Config.MOVIE_EMBEDDING_DIM,
        action_dim=Config.ACTION_DIM,
        hidden_dim=Config.HIDDEN_DIM
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none') # We want to weight it manually
    
    # Training Loop
    epochs = 2000
    print(f"🔥 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (user_cluster, last_movie_emb, action, reward) in enumerate(dataloader):
            # Move to device
            user_cluster = user_cluster.to(Config.DEVICE)
            last_movie_emb = last_movie_emb.to(Config.DEVICE)
            action = action.to(Config.DEVICE)
            reward = reward.to(Config.DEVICE)
            
            # Forward
            optimizer.zero_grad()
            action_logits, _ = model(user_cluster, last_movie_emb)
            
            # Loss: CrossEntropy between predicted logits and actual taken action
            # Weighted by Reward: We want to Clone behavior that got High Reward more strongly.
            # Since we filtered only positive rewards, we can just use reward as weight.
            # Only 'watch' (0.3) and 'like' (1.0) remain. 
            # 'like' events will be pushed 3.3x more than 'watch'.
            
            loss_unreduced = criterion(action_logits, action)
            weighted_loss = (loss_unreduced * reward).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"   Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] Loss: {weighted_loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}. Time: {duration:.1f}s")
        
    # Save Model
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        os.makedirs(Config.MODEL_SAVE_PATH)
    save_path = os.path.join(Config.MODEL_SAVE_PATH, "offline_actor_critic.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    train_offline()
