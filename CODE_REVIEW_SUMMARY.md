# Code Review Summary - Humanoid Climbing Pipeline

## 📋 Overview

This project implements a **video → pose extraction → behavioral cloning → reinforcement learning** pipeline for training a humanoid robot to climb walls. The system extracts human climbing movements from videos and uses them to guide robot training.

---

## 🎯 Pipeline Flow

```
Climbing Videos (.mp4)
    ↓
[1. video_processor.py] → Extract human poses using MediaPipe
    ↓
Pose Data (JSON + annotated videos)
    ↓
[2. pose_to_actions.py] → Convert poses to humanoid state-action pairs
    ↓
Demonstrations (state-action pairs)
    ↓
[3. bc_trainer.py] → Train BC policy → Fine-tune with PPO
    ↓
Trained Climbing Model
```

---

## 📂 Code Structure Analysis

### 1. **video_processor.py** ✅ WORKING
**Purpose**: Extract pose keypoints from climbing videos using MediaPipe

**What it does**:
- Reads video files frame by frame
- Detects human pose using MediaPipe (33 body landmarks)
- Extracts 3D keypoints (x, y, z) and visibility scores
- Saves pose data as JSON files
- Creates annotated videos with skeleton overlay

**Status**: ✅ **Working correctly** - Successfully extracted poses from all 5 videos

**Output**:
- `extracted_poses/Vid1_poses.json` (and similar for other videos)
- `extracted_poses/Vid1_with_poses.mp4` (annotated videos)

**Potential Issues**: ⚠️
- Uses **fixed 30 FPS** for output videos (should match source video FPS)
- No handling for frames where pose detection fails (they're skipped)
- Memory could be an issue with very long videos (stores all frames)

---

### 2. **pose_to_actions.py** ⚠️ PARTIALLY WORKING
**Purpose**: Convert human poses to humanoid robot state-action pairs

**What it does**:
- Loads extracted pose data
- Calculates joint angles from pose keypoints
- Maps human joint angles to humanoid state space
- Generates action sequences from state transitions
- Saves demonstrations for behavioral cloning

**Status**: ⚠️ **Has bugs - see issues below**

**Critical Issues Found**:

#### 🐛 **Bug #1: Environment Initialization**
```python
# Line 22-28 in __init__
self.env = HumanoidClimbEnv(
    motion_path=motion_path,
    motion_exclude_targets=motion_exclude_targets
)
self.robot = self.env.robot  # ❌ WRONG!
```

**Problem**: `HumanoidClimbEnv` doesn't have a `robot` attribute - it has `humanoid`
**Fix**: Change to `self.robot = self.env.humanoid` or rename the variable

#### 🐛 **Bug #2: Oversimplified Joint Mapping**
```python
# Lines 161-173
angle_mapping = {
    'left_shoulder': 0,
    'right_shoulder': 1,
    # ...only 8 joints mapped
}
```

**Problems**:
- Only maps 8 joints, but humanoid has **17 actuated joints + 4 grasp actions = 21 DoF**
- Direct index mapping without considering actual joint names
- No mapping for spine, neck, or detailed hand/foot joints
- Normalized angle range might not match robot's actual joint limits

#### 🐛 **Bug #3: Naive Action Generation**
```python
# Lines 184-199
def generate_action_from_state_transition(self, current_state, next_state):
    state_diff = next_state - current_state
    action = np.clip(state_diff * 10.0, -1.0, 1.0)  # ❌ Too simplistic!
```

**Problems**:
- Assumes actions = scaled state differences (not physically accurate)
- No inverse dynamics or physics consideration
- Arbitrary scaling factor (10.0) with no justification
- Doesn't account for robot dynamics, torque limits, or actuator constraints

#### 🐛 **Bug #4: Coordinate Frame Mismatch**
```python
# Line 143-144
p1 = np.array(point1[:2])  # Uses only x,y
```

**Problem**: MediaPipe outputs normalized coordinates (0-1 range), but the code treats them as physical distances. The z-coordinate (depth) is ignored which could lose important climbing information.

---

### 3. **bc_trainer.py** ⚠️ NEEDS ATTENTION
**Purpose**: Train behavioral cloning policy and fine-tune with PPO

**What it does**:
- Loads demonstration data
- Trains a neural network to clone demonstrated actions
- Creates PPO agent with BC initialization
- Trains with reinforcement learning

**Status**: ⚠️ **Untested - potential issues**

**Issues Found**:

#### ⚠️ **Issue #1: Environment Creation Mismatch**
```python
# Line 146
def make_env():
    return HumanoidClimbEnv()  # ❌ Missing required arguments!
```

**Problem**: `HumanoidClimbEnv` requires `motion_path` and `motion_exclude_targets` but the make_env function doesn't provide them. This will crash when creating vectorized environments.

#### ⚠️ **Issue #2: No BC Policy Integration**
```python
# Lines 171-173
bc_policy = self.train_behavioral_cloning_policy()
# ... but then creates PPO from scratch without using bc_policy!
ppo_model, vec_env = self.create_guided_ppo_trainer()
```

**Problem**: The BC policy is trained but never actually used to initialize the PPO policy. The "guidance" is lost.

#### ⚠️ **Issue #3: Demonstration Data Quality**
The demonstrations depend on the buggy `pose_to_actions.py`, so the data quality is questionable.

---

### 4. **humanoid_climb_env.py** ✅ MOSTLY GOOD
**Purpose**: Gymnasium environment for humanoid climbing simulation

**What it does**:
- Manages PyBullet physics simulation
- Defines observation/action spaces
- Implements reward function
- Tracks climbing progress through stances

**Status**: ✅ **Well-implemented** - This is your existing, working code

**Observations**:
- **Observation space**: 306 dimensions (joint states + targets + stance info)
- **Action space**: 21 dimensions (17 joints + 4 grasps)
- **Reward function**: Multi-component (distance, velocity, posture, wall impact, floor contact)
- **Stance-based progression**: Smart climbing through predefined hold sequences

**Minor Issues**:
- Hardcoded target positions (not flexible for different walls)
- No way to dynamically adjust stances based on demonstrations

---

## 🚨 Critical Bugs Summary

### High Priority (Breaks Pipeline)
1. ✅ **Fixed in code**: `pose_to_actions.py` - `self.env.robot` should be `self.env.humanoid`
2. ❌ **Not fixed**: `bc_trainer.py` - `make_env()` missing required arguments
3. ❌ **Not fixed**: Joint mapping only covers 8/21 joints

### Medium Priority (Reduces Quality)
4. ❌ **Not fixed**: Action generation is too simplistic (state diff × 10)
5. ❌ **Not fixed**: BC policy not integrated into PPO training
6. ❌ **Not fixed**: Coordinate frame mismatch (normalized vs physical)

### Low Priority (Nice to Have)
7. ⚠️ **Improvement needed**: Video FPS should match source
8. ⚠️ **Improvement needed**: Handle missing pose detections better
9. ⚠️ **Improvement needed**: Add validation for demonstration quality

---

## 🔍 Data Flow Validation

### Current Status:
1. ✅ **Videos → Poses**: Working (5 videos processed, poses extracted)
2. ⚠️ **Poses → Demonstrations**: Runs but produces questionable data
3. ❌ **Demonstrations → Training**: Will crash due to env initialization bug

### What Actually Happens:
```
Vid1.mp4 (1640 frames) → 1640 poses → ~1639 state-action pairs
Vid2.mp4 (1412 frames) → 1412 poses → ~1411 state-action pairs
... (total ~8000+ demonstrations)
```

**But**: The state-action pairs are likely not physically accurate due to the bugs above.

---

## 💡 Recommendations

### Immediate Fixes (Must Do):
1. **Fix `pose_to_actions.py` line 30**: Change `self.robot = self.env.robot` to `self.robot = self.env.humanoid`
2. **Fix `bc_trainer.py` line 146**: Add motion_path and motion_exclude_targets to make_env()
3. **Expand joint mapping**: Map all 21 joints properly based on humanoid model

### Important Improvements (Should Do):
4. **Rewrite action generation**: Use inverse kinematics or learned inverse model instead of state diff
5. **Integrate BC policy**: Transfer BC weights to PPO policy before training
6. **Add data validation**: Check if generated demonstrations are physically plausible
7. **Fix coordinate frames**: Properly scale MediaPipe coordinates to physical robot space

### Nice to Have:
8. Add visualization of demonstrations in simulation before training
9. Add metrics to evaluate demonstration quality
10. Support for multiple camera angles/perspectives
11. Dynamic stance generation from video analysis

---

## 🎓 Conceptual Issues

### The Fundamental Challenge:
**You're trying to map human climbing to robot climbing, but they're very different:**

- **Human**: 33 pose keypoints in normalized image space (0-1)
- **Robot**: 21 joint angles/torques in physical space with specific constraints

### What's Missing:
1. **Retargeting algorithm**: Proper mapping between human and robot morphologies
2. **Inverse kinematics**: Convert desired end-effector positions to joint angles
3. **Dynamics consideration**: Account for robot's mass, inertia, torque limits
4. **Contact planning**: Where and how to grasp (critical for climbing)

### Current Approach:
The code tries to directly convert pose angles to robot states, which assumes:
- Human and robot have similar kinematic structures (they don't)
- Joint angles alone determine climbing success (they don't - contact forces matter)
- Simple angle interpolation produces valid trajectories (it doesn't - physics violations)

---

## ✅ What's Working Well

1. **Video processing**: MediaPipe pose extraction is solid
2. **Environment**: Your `HumanoidClimbEnv` is well-designed
3. **Pipeline structure**: The modular design is clean and logical
4. **Error handling**: Good try-except blocks and status reporting

---

## 🎯 Next Steps

### Option A: Quick Fixes (Get it running)
1. Fix the 3 critical bugs mentioned above
2. Run pipeline end-to-end
3. See what trained model produces (likely not great, but instructive)

### Option B: Proper Implementation (Better results)
1. Implement proper human-to-robot retargeting
2. Use inverse kinematics for joint angle calculation
3. Add contact/grasp planning from video
4. Validate demonstrations in simulation before training

### Option C: Alternative Approach
Instead of direct pose mapping, consider:
1. Train robot in simulation with RL (no human demos)
2. Use human videos only for **high-level planning** (which holds to use)
3. Let RL figure out **how** to reach those holds

---

## 📊 Expected Behavior vs Reality

| Component | Expected | Reality | Status |
|-----------|----------|---------|--------|
| Video Processing | Extract poses | ✅ Works perfectly | ✅ |
| Pose → States | Map to robot states | ⚠️ Crude mapping | ⚠️ |
| Pose → Actions | Generate actions | ❌ Oversimplified | ❌ |
| BC Training | Learn from demos | ❌ Can't initialize env | ❌ |
| PPO Training | Refine with RL | ❌ Not reached yet | ❌ |

---

## 🏁 Conclusion

**Good News**: Your project structure and video processing work well!

**Bad News**: The pose-to-action conversion has fundamental issues that will produce low-quality demonstrations.

**Bottom Line**: The pipeline will run (after fixing bugs) but won't produce a climbing robot without addressing the conceptual issues in human-to-robot mapping.

**Recommendation**: Start with Option A to get it running, observe the results, then decide if Option B or C makes more sense for your goals.