# Humanoid Climbing RL Environment

A reinforcement learning environment for training humanoid robots to climb walls using PyBullet physics simulation and Stable-Baselines3.

## Features

- **Custom Gymnasium Environment**: `HumanoidClimb-v0` with 306-dimensional observation space and 21-dimensional action space
- **Stance-Based Climbing**: Progressive climbing through predefined hand/foot positions
- **Multi-Component Rewards**: Distance, velocity, posture, wall impact, and floor contact rewards
- **PyBullet Physics**: Realistic physics simulation with constraint-based grasping
- **Experiment Tracking**: Weights & Biases integration for monitoring training progress

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd humanoid_climb
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train a new model from scratch:
```bash
python train.py HumanoidClimb-v0 PPO -w 4 -t
```

Continue training from an existing model:
```bash
python train.py HumanoidClimb-v0 PPO -w 4 -t -f path/to/model.zip
```

### Testing

Test a trained model:
```bash
python train.py HumanoidClimb-v0 PPO -s path/to/model.zip
```

Run the sequential stance demo (uses your trained model):
```bash
python climb.py
```

Interactive manual control:
```bash
python test.py
```

## Environment Details

- **Observation Space**: 306-dimensional vector containing joint states, target positions, and stance information
- **Action Space**: 21-dimensional continuous actions (17 joint torques + 4 grasp actions)
- **Reward Components**:
  - Distance reward: Progress toward target holds
  - Velocity reward: Upward movement encouragement
  - Posture reward: Proper climbing stance maintenance
  - Impact penalty: Excessive wall contact forces
  - Floor penalty: Touching the ground

## Climbing Stances

The environment uses predefined stances that specify target positions for each effector:
- `STANCE_1`: Initial grip positions
- `STANCE_2`: First foot placement
- `STANCE_3`: Full initial stance
- `STANCE_4`: Progressive climbing stance

## Requirements

- Python 3.8+
- PyBullet for physics simulation
- Gymnasium for RL environment interface
- Stable-Baselines3 for RL algorithms
- Weights & Biases for experiment tracking (optional)
---

## 🏗️ Core Architecture Components

### 1. **Physics Simulation Layer (PyBullet)**

**Purpose**: Provides the foundational physics engine for realistic humanoid-environment interaction.

**Key Components**:
- **BulletClient**: Core physics simulation engine
- **URDF Models**: XML-based robot descriptions defining joint hierarchies, constraints, and physical properties
- **Collision Detection**: Handles humanoid-wall contact physics
- **Gravity & Dynamics**: Realistic 9.8 m/s² gravity with configurable time steps

**Implementation Details**:
```python
# Physics configuration with deterministic behavior
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
bullet_client.setGravity(0, 0, -9.8)
```

### 2. **Humanoid Robot Model (`humanoid.py`)**

**Architecture**: 21-degree-of-freedom articulated robot with:
- **17 joint actuators**: Primary locomotion control
- **4 grasp actuators**: Hand/grip control for wall contact
- **State management**: Position, velocity, orientation tracking
- **Action space**: Continuous control [-1, +1] for all actuators

**Key Features**:
- **State persistence**: Can save/load robot configurations using NumPy serialization
- **Motor abstraction**: Clean interface between RL actions and joint commands
- **Physical constraints**: Realistic joint limits and dynamics

### 3. **Environment Framework (`HumanoidClimbEnv`)**

**Gymnasium Integration**: Follows OpenAI Gym standards for RL compatibility.

**Observation Space**: 306-dimensional vector containing:
- Joint positions and velocities (17 × 2 = 34 dimensions)
- Body orientation and angular velocity (7 dimensions)
- Target positions and distances (multiple targets × 3D coordinates)
- Contact forces and wall proximity sensors
- Stance progression indicators

**Action Space**: 21-dimensional continuous control:
- Actions ∈ [-1, +1] mapped to joint torques
- Real-time motor command application

**Reward Function Architecture**:
```python
# Multi-component reward system
total_reward = (
    distance_reward +      # Progress toward climbing targets
    velocity_reward +      # Efficient movement incentive
    slouch_reward +        # Posture maintenance
    floor_reward +         # Staying off ground
    stance_reward +        # Stance completion bonuses
    wall_penalty          # Contact force penalties
)
```

## 🎯 Multi-Stage Learning System

### **Stance-Based Progression**

The climbing task is decomposed into **4 progressive stances**, each building upon previous skills:

1. **STANCE_1** (`[10, 9, -1, -1]`): Initial wall contact
   - Targets: Upper hand holds (positions 10, 9)
   - Focus: Basic wall approach and upper body coordination

2. **STANCE_2** (`[10, 9, 2, -1]`): Adding foot support
   - Previous + foot target (position 2)
   - Action override: `[1, 1, -1, -1]` (forced hand grips)

3. **STANCE_3** (`[10, 9, 2, 1]`): Full limb coordination
   - Complete 4-point contact
   - State file: Builds from previous stance configuration

4. **STANCE_4** (`[14, 13, -1, -1]`): Advanced progression
   - New target positions with exclusion lists
   - Complex coordination requirements

**State File System**:
```python
# Stance progression with saved states
state_file="/states/state_10_9_2_1.npz"  # Previous successful configuration
action_override=[1, 1, 1, -1]           # Forced actions for stability
exclude_targets=[[10], [9], [2, 6], [1, 5]]  # Forbidden positions
```

## 🧠 Reinforcement Learning Implementation

### **Algorithm Choice: PPO (Proximal Policy Optimization)**

**Why PPO**:
- **Stability**: Prevents destructive policy updates
- **Sample efficiency**: Good for continuous control tasks
- **Robustness**: Handles high-dimensional action spaces well

**Key Hyperparameters**:
```python
model = sb.PPO(
    'MlpPolicy',           # Multi-layer perceptron policy
    vec_env,               # Vectorized environments
    batch_size=2048,       # Large batches for stability
    device=DEVICE,         # CPU/GPU acceleration
    tensorboard_log=log_dir # Logging integration
)
```

### **Multi-Worker Parallel Training**

**SubprocVecEnv Architecture**:
- **4 parallel environments** for 4x data collection speed
- **Independent physics simulations** in separate processes
- **Synchronized experience collection** for batch training

```python
vec_env = SubprocVecEnv(
    [make_env(env_name, i, max_steps=max_ep_steps, stance=stance) 
     for i in range(workers)], 
    start_method="spawn"
)
```

**Benefits**:
- **Sample efficiency**: 4x faster experience collection
- **Diverse experience**: Multiple simultaneous explorations
- **Stability**: Reduces correlation in training data

## 📊 Training Infrastructure & Monitoring

### **Weights & Biases Integration**

**Comprehensive Experiment Tracking**:
```python
wandb_kwargs = {
    "project": "HumanoidClimb-RL",
    "entity": "rawan_khalifa-minerva-university",
    "config": config,
    "sync_tensorboard": True,
    "monitor_gym": False
}
```

**Tracked Metrics**:
- Training loss and policy gradients
- Episode rewards and success rates
- Custom reward component breakdowns
- Model checkpoints and evaluation scores

### **Custom Callback System**

**Reward Component Logging**:
```python
class CustomCallback(BaseCallback):
    def _on_rollout_end(self):
        # Aggregate multi-environment reward data
        reward_infos = self.training_env.get_attr('last_reward_components')
        
        # Calculate component averages
        avg_rewards = {k: sum(v)/len(v) for k, v in components.items()}
        
        # Log to wandb/tensorboard
        for key, value in avg_rewards.items():
            self.logger.record(f"climb/{key}", value)
```

### **Evaluation Pipeline**

**EvalCallback Configuration**:
- **Frequency**: Every 500 training steps
- **Deterministic evaluation**: Consistent performance measurement
- **Best model saving**: Automatic checkpoint management
- **Success rate tracking**: Task completion metrics

## 🔧 Development & Testing Tools

### **Interactive Testing (`test.py`)**

**Manual Control Interface**:
- **PyBullet GUI**: Real-time physics visualization
- **Debug sliders**: Individual joint control for analysis
- **Interactive manipulation**: Direct humanoid motor testing

### **Trained Model Evaluation (`train.py -s`)**

**Visual Performance Assessment**:
```python
# Test trained model with visual rendering
env = gym.make(args.gymenv, render_mode='human', **stance.get_args())
model = sb.PPO.load(path_to_model, env=env)
```

### **Multi-Model Climbing Demo (`climb.py`)**

**Production Climbing System**:
- **Sequential stance models**: Different trained models for each stance
- **Action override integration**: Forced actions for stability
- **Success rate tracking**: Performance monitoring
- **Interactive controls**: Pause/reset functionality

## 📦 Library Ecosystem & Dependencies

### **Core RL Framework**
- **Stable-Baselines3 (2.0.0+)**: Modern RL algorithms with PyTorch backend
- **Gymnasium (0.29.0+)**: Standardized RL environment interface
- **PyTorch (1.13.0+)**: Deep learning framework for policy networks

### **Physics & Simulation**
- **PyBullet (3.2.5+)**: Real-time physics simulation engine
- **NumPy (1.21.0+)**: Numerical computing for state management

### **Monitoring & Visualization**
- **Weights & Biases (0.15.0+)**: Experiment tracking and collaboration
- **TensorBoard (2.13.0+)**: Training metrics visualization
- **Rich (13.0.0+)**: Enhanced terminal output formatting
- **tqdm (4.60.0+)**: Progress bars for training feedback

### **Development Tools**
- **Matplotlib (3.5.0+)**: Data visualization and analysis
- **Seaborn (0.11.0+)**: Statistical plotting enhancements

## 🚀 Training Process & Optimization

### **Training Configuration**

**Scale**: 25 million timesteps over ~6 hours
**Hardware**: CPU-optimized (1,800+ iterations/second)
**Memory**: Efficient multi-process architecture

### **Learning Progression**

**Observed Performance**:
- **Initial**: -169.91 average reward (poor coordination)
- **Mid-training**: -83.52 reward (basic wall contact)
- **Advanced**: -57.6 reward (coordinated climbing motions)
- **Improvement**: ~65% performance gain demonstrating successful learning

### **Curriculum Learning Elements**

**Progressive Difficulty**:
1. **Basic physics**: Standing and balance
2. **Wall approach**: Moving toward targets
3. **Contact management**: Grip force control
4. **Multi-limb coordination**: Full climbing sequences

