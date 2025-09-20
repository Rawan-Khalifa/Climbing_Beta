#!/usr/bin/env python3
"""
Model Testing and Evaluation
Test the trained behavioral cloning model in the climbing environment.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the humanoid_climb module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoid_climb.env.humanoid_climb_env import HumanoidClimbEnv
from stable_baselines3 import PPO

def test_model(model_path, num_episodes=5, render=True):
    """Test the trained model"""
    
    # Load environment
    if render:
        env = HumanoidClimbEnv(render_mode='human')
    else:
        env = HumanoidClimbEnv()
    
    # Load model
    try:
        model = PPO.load(model_path, env=env)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n🎬 Episode {episode + 1}/{num_episodes}")
        
        while True:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
    
    # Summary
    print(f"\n📊 Test Results ({num_episodes} episodes):")
    print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Best episode: {np.max(episode_rewards):.2f}")
    print(f"  Worst episode: {np.min(episode_rewards):.2f}")
    
    env.close()

def main():
    model_path = "best_bc_guided_model/best_model.zip"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        
        # Look for models in common locations
        for path in ["best_bc_guided_model/", ".", "models/"]:
            path = Path(path)
            if path.exists():
                for model_file in path.glob("*.zip"):
                    print(f"  {model_file}")
        sys.exit(1)
    
    print("🎯 Testing Trained Climbing Model")
    print(f"📁 Model: {model_path}")
    print("-" * 50)
    
    test_model(model_path, num_episodes=3, render=True)

if __name__ == "__main__":
    main()
