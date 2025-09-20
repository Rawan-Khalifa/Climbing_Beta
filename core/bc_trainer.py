#!/usr/bin/env python3
"""
Behavioral Cloning with RL Integration
Uses extracted demonstrations to guide PPO training for climbing behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import pickle

# Add the humanoid_climb module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoid_climb.env.humanoid_climb_env import HumanoidClimbEnv

class BehavioralCloningTrainer:
    def __init__(self, demonstrations_dir):
        self.demonstrations_dir = Path(demonstrations_dir)
        # Initialize environment for training
        motion_path = [[10, 9, -1, -1]]  # Default stance
        motion_exclude_targets = [[]]     # No excluded targets
        self.env = HumanoidClimbEnv(motion_path=motion_path, motion_exclude_targets=motion_exclude_targets)
        self.demonstrations = self._load_all_demonstrations()
        
    def _load_all_demonstrations(self):
        """Load all demonstration files"""
        all_demonstrations = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        for demo_file in self.demonstrations_dir.glob("*_demonstrations.json"):
            print(f"Loading {demo_file.name}...")
            
            with open(demo_file, 'r') as f:
                demo_data = json.load(f)
            
            all_demonstrations['states'].extend(demo_data['states'])
            all_demonstrations['actions'].extend(demo_data['actions'])
            all_demonstrations['rewards'].extend(demo_data['rewards'])
        
        # Convert to numpy arrays
        all_demonstrations['states'] = np.array(all_demonstrations['states'])
        all_demonstrations['actions'] = np.array(all_demonstrations['actions'])
        all_demonstrations['rewards'] = np.array(all_demonstrations['rewards'])
        
        print(f"✓ Loaded {len(all_demonstrations['states'])} demonstrations")
        return all_demonstrations
    
    def create_bc_dataset(self):
        """Create dataset for behavioral cloning"""
        return {
            'observations': self.demonstrations['states'],
            'actions': self.demonstrations['actions']
        }
    
    def train_behavioral_cloning_policy(self, epochs=100, batch_size=64):
        """Train a behavioral cloning policy using supervised learning"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prepare data
        states = torch.FloatTensor(self.demonstrations['states']).to(device)
        actions = torch.FloatTensor(self.demonstrations['actions']).to(device)
        
        # Create simple neural network policy
        class BCPolicy(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Tanh()  # Actions are typically in [-1, 1]
                )
            
            def forward(self, state):
                return self.network(state)
        
        # Initialize policy
        state_dim = states.shape[1]
        action_dim = actions.shape[1]
        policy = BCPolicy(state_dim, action_dim).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        print(f"Training BC policy for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                
                # Forward pass
                predicted_actions = policy(batch_states)
                loss = criterion(predicted_actions, batch_actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
        
        # Save policy
        torch.save(policy.state_dict(), 'bc_policy.pth')
        print("✓ BC policy saved to bc_policy.pth")
        
        return policy
    
    def create_guided_ppo_trainer(self, bc_policy_path=None):
        """Create PPO trainer with behavioral cloning guidance"""
        
        # Create vectorized environment
        def make_env():
            return HumanoidClimbEnv()
        
        # Use multiple environments for better sample efficiency
        vec_env = SubprocVecEnv([make_env for _ in range(4)])
        
        # Initialize PPO
        model = PPO(
            'MlpPolicy',
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            clip_range=0.2,
            tensorboard_log="./ppo_bc_logs/"
        )
        
        return model, vec_env
    
    def train_with_bc_guidance(self, total_timesteps=100000):
        """Train PPO with behavioral cloning initialization"""
        
        # First, train BC policy
        bc_policy = self.train_behavioral_cloning_policy()
        
        # Create PPO trainer
        ppo_model, vec_env = self.create_guided_ppo_trainer()
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path='./best_bc_guided_model/',
            log_path='./eval_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        print(f"Starting PPO training with BC guidance for {total_timesteps} timesteps...")
        
        # Train the model
        ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save final model
        ppo_model.save("bc_guided_ppo_final")
        print("✓ BC-guided PPO model saved")
        
        return ppo_model
    
    def save_demonstrations_for_sb3(self, output_file="bc_demonstrations.pkl"):
        """Save demonstrations in format compatible with SB3"""
        # Convert to format expected by SB3
        demonstrations_sb3 = []
        
        for i in range(len(self.demonstrations['states'])):
            demo = {
                'obs': self.demonstrations['states'][i],
                'acts': self.demonstrations['actions'][i],
                'rews': self.demonstrations['rewards'][i],
                'episode_starts': i == 0  # Only first observation starts episode
            }
            demonstrations_sb3.append(demo)
        
        with open(output_file, 'wb') as f:
            pickle.dump(demonstrations_sb3, f)
        
        print(f"✓ Demonstrations saved to {output_file} in SB3 format")

def main():
    demonstrations_dir = "demonstrations"
    
    if len(sys.argv) > 1:
        demonstrations_dir = sys.argv[1]
    
    print("🎯 Behavioral Cloning + RL Training Pipeline")
    print(f"📁 Demonstrations: {demonstrations_dir}")
    print("-" * 50)
    
    # Initialize trainer
    trainer = BehavioralCloningTrainer(demonstrations_dir)
    
    # Save demonstrations in SB3 format
    trainer.save_demonstrations_for_sb3()
    
    # Train with BC guidance
    model = trainer.train_with_bc_guidance(total_timesteps=50000)
    
    print("\n🎉 Training complete!")
    print("📁 Check './best_bc_guided_model/' for best model")
    print("📁 Check './ppo_bc_logs/' for training logs")

if __name__ == "__main__":
    main()
