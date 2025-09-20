# Humanoid Climbing - Clean Pipeline

A streamlined pipeline for training humanoid robots to climb using behavioral cloning from human climbing videos.

## 🎯 Core Pipeline

1. **Video Processing** → Extract poses from climbing videos
2. **Pose Conversion** → Convert poses to state-action pairs  
3. **Behavioral Cloning** → Train PPO with demonstration guidance

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure your climbing videos are in the videos directory
mkdir -p climbing_videos
# Add your .mp4 files to climbing_videos/
```

### 2. Run Complete Pipeline
```bash
# Process videos, extract poses, and train model
python run_pipeline.py --videos-dir climbing_videos
```

### 3. Test Trained Model
```bash
# Test the trained model
python core/test_model.py
```

## 📁 Project Structure

```
├── core/                           # Core pipeline components
│   ├── video_processor.py          # Extract poses from videos
│   ├── pose_to_actions.py          # Convert poses to state-action pairs
│   ├── bc_trainer.py               # Behavioral cloning + RL training
│   └── test_model.py               # Test trained models
├── humanoid_climb/                 # Humanoid environment
│   ├── env/                        # Gymnasium environment
│   ├── assets/                     # Robot and environment models
│   └── stances/                    # Climbing stance definitions
├── climbing_videos/                # Input videos (.mp4)
├── extracted_poses/                # Generated pose data
├── demonstrations/                 # State-action demonstrations
└── run_pipeline.py                 # Complete pipeline runner
```

## 🔧 Individual Components

### Extract Poses
```bash
python core/video_processor.py climbing_videos extracted_poses
```

### Convert to Demonstrations
```bash
python core/pose_to_actions.py extracted_poses demonstrations
```

### Train Model
```bash
python core/bc_trainer.py demonstrations
```

## 📊 Outputs

- `extracted_poses/` - Pose keypoints and annotated videos
- `demonstrations/` - State-action pairs in JSON format
- `bc_demonstrations.pkl` - SB3-compatible demonstration format
- `best_bc_guided_model/` - Best trained model
- `ppo_bc_logs/` - Training logs for TensorBoard

## 🎮 Testing

```bash
# Test with visualization
python core/test_model.py best_bc_guided_model/best_model.zip

# View training progress
tensorboard --logdir ppo_bc_logs
```

## 🧠 Key Features

- **Pose Extraction**: MediaPipe-based human pose detection
- **Smart Conversion**: Maps human poses to humanoid joint angles
- **Behavioral Cloning**: Supervised learning from demonstrations
- **PPO Integration**: Reinforcement learning with BC initialization
- **Multi-processing**: Parallel environment training

## 📝 Pipeline Options

```bash
# Skip steps if you have existing data
python run_pipeline.py --skip-extraction    # Use existing poses
python run_pipeline.py --skip-conversion    # Use existing demonstrations
python run_pipeline.py --skip-training      # Don't train

# Adjust training
python run_pipeline.py --timesteps 100000   # More training steps
```

## 🎯 Next Steps

1. Add your climbing videos to `climbing_videos/`
2. Run `python run_pipeline.py`
3. Test with `python core/test_model.py`
4. Iterate and improve!
