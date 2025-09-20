#!/usr/bin/env python3
"""
Complete Pipeline Runner
Orchestrates the entire video → pose → demonstrations → training pipeline.
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(command)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Complete climbing video analysis and training pipeline")
    parser.add_argument("--videos-dir", default="climbing_videos", help="Directory containing climbing videos")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip pose extraction (use existing)")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip pose to action conversion")
    parser.add_argument("--skip-training", action="store_true", help="Skip behavioral cloning training")
    parser.add_argument("--timesteps", type=int, default=50000, help="PPO training timesteps")
    
    args = parser.parse_args()
    
    print("🎬 Complete Climbing Video Analysis Pipeline")
    print("=" * 60)
    
    # Ensure core directory exists
    core_dir = Path("core")
    if not core_dir.exists():
        print("❌ Core directory not found. Please run from project root.")
        sys.exit(1)
    
    # Step 1: Extract poses from videos
    if not args.skip_extraction:
        success = run_command([
            sys.executable, "core/video_processor.py", 
            args.videos_dir, "extracted_poses"
        ], "Extracting poses from climbing videos")
        
        if not success:
            print("❌ Pipeline failed at pose extraction step")
            sys.exit(1)
    else:
        print("⏭️  Skipping pose extraction (using existing data)")
    
    # Step 2: Convert poses to state-action pairs
    if not args.skip_conversion:
        success = run_command([
            sys.executable, "core/pose_to_actions.py",
            "extracted_poses", "demonstrations"
        ], "Converting poses to state-action demonstrations")
        
        if not success:
            print("❌ Pipeline failed at pose conversion step")
            sys.exit(1)
    else:
        print("⏭️  Skipping pose conversion (using existing data)")
    
    # Step 3: Train with behavioral cloning
    if not args.skip_training:
        success = run_command([
            sys.executable, "core/bc_trainer.py",
            "demonstrations"
        ], f"Training BC-guided PPO for {args.timesteps} timesteps")
        
        if not success:
            print("❌ Pipeline failed at training step")
            sys.exit(1)
    else:
        print("⏭️  Skipping training")
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📁 Generated files:")
    print("  • extracted_poses/ - Pose data from videos")
    print("  • demonstrations/ - State-action pairs")
    print("  • bc_demonstrations.pkl - SB3 format demonstrations")
    print("  • best_bc_guided_model/ - Best trained model")
    print("  • ppo_bc_logs/ - Training logs")
    
    print("\n🚀 Next steps:")
    print("  • Test the trained model: python core/test_model.py")
    print("  • View training logs: tensorboard --logdir ppo_bc_logs")

if __name__ == "__main__":
    main()
