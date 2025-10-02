#!/usr/bin/env python3
"""
Quick Model Evaluation (No Rendering)
Quickly test the model performance without visualization.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the humanoid_climb module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoid_climb.env.humanoid_climb_env import HumanoidClimbEnv
from stable_baselines3 import PPO

def quick_test(model_path, num_episodes=10):
    """Test the trained model without rendering"""
    
    # Define motion path for testing
    motion_path = [[10, 9, 2, 1]]  # Simple climbing stance
    motion_exclude_targets = [[[], [], [], []]]  # No excluded targets
    
    # Load environment (no rendering)
    env = HumanoidClimbEnv(
        motion_path=motion_path,
        motion_exclude_targets=motion_exclude_targets
    )
    
    # Load model
    try:
        model = PPO.load(model_path, env=env)
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test episodes
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    print(f"\n🎯 Running {num_episodes} test episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
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
        episode_lengths.append(steps)
        successes.append(1 if info.get('is_success', False) else 0)
        
        print(f"  Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}, Success={info.get('is_success', False)}")
    
    # Summary
    print(f"\n📊 Test Results ({num_episodes} episodes):")
    print(f"  Average reward:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average length:  {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Success rate:    {np.mean(successes)*100:.1f}%")
    print(f"  Best episode:    {np.max(episode_rewards):.2f}")
    print(f"  Worst episode:   {np.min(episode_rewards):.2f}")
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes)
    }

def main():
    model_path = "best_bc_guided_model/best_model.zip"
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("\nAvailable models:")
        
        # Look for models in common locations
        for path in ["best_bc_guided_model/", ".", "models/"]:
            path = Path(path)
            if path.exists():
                for model_file in path.glob("*.zip"):
                    print(f"  {model_file}")
        sys.exit(1)
    
    print("⚡ Quick Model Evaluation (No Rendering)")
    print(f"📁 Model: {model_path}")
    print("-" * 50)
    
    results = quick_test(model_path, num_episodes=10)

if __name__ == "__main__":
    main()