#!/usr/bin/env python3
"""
Script to check Wandb configuration and available organizations
"""

import wandb
import sys
import os

def main():
    print("🔍 Checking Wandb Configuration...")
    print("=" * 50)
    
    try:
        # Check if logged in
        api = wandb.Api()
        user_info = api.viewer
        
        if user_info:
            print(f"✅ Logged in as: {user_info}")
            
            # Try to get teams/organizations
            try:
                # Get current settings
                settings = wandb.Settings()
                env_entity = os.environ.get('WANDB_ENTITY')
                print(f"🏢 Default entity (settings): {settings.entity or 'None'}")
                print(f"🏢 Default entity (env var): {env_entity or 'None'}")
                print(f"🏢 Effective entity: {env_entity or settings.entity or 'None'}")
                
                # Try to list projects to see available entities
                print("\n📂 Recent projects and entities:")
                projects = list(api.projects(limit=10))
                entities = set()
                
                for project in projects:
                    entity = project.entity
                    entities.add(entity)
                    print(f"  - {entity}/{project.name}")
                
                print(f"\n🏢 Available entities/organizations:")
                for entity in sorted(entities):
                    print(f"  - {entity}")
                    
            except Exception as e:
                print(f"⚠️  Couldn't fetch organizations: {e}")
                
        else:
            print("❌ Not logged in to Wandb")
            print("Run 'wandb login' to authenticate")
            
    except Exception as e:
        print(f"❌ Error accessing Wandb API: {e}")
        print("Make sure you're logged in with 'wandb login'")
        
    print("\n" + "=" * 50)
    print("💡 Usage tips:")
    print("   # Use your Minerva University organization (default):")
    print("   python train.py HumanoidClimb-v0 PPO -w 4 -t")
    print("")
    print("   # Use different organization:")
    print("   python train.py HumanoidClimb-v0 PPO -w 4 -t -e biasdrive-neuromatch")
    print("")
    print("   # Disable wandb completely:")
    print("   python train.py HumanoidClimb-v0 PPO -w 4 -t --no-wandb")

if __name__ == "__main__":
    main()
