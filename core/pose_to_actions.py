#!/usr/bin/env python3
"""
Pose to State-Action Converter
Converts extracted pose data to humanoid state-action pairs for behavioral cloning.
"""

import numpy as np
import json
from pathlib import Path
import sys
import os

# Add the humanoid_climb module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from humanoid_climb.env.humanoid_climb_env import HumanoidClimbEnv
from humanoid_climb.assets.humanoid import Humanoid

class PoseToStateActionConverter:
    def __init__(self):
        # Initialize environment for state-action conversion
        # Use a simple climbing path for demonstration conversion
        motion_path = [[10, 9, 2, 1]]  # List of stances, each stance has 4 positions (2 hands, 2 feet)
        motion_exclude_targets = [[[], [], [], []]]  # List of excluded targets per effector per stance
        
        self.env = HumanoidClimbEnv(
            motion_path=motion_path,
            motion_exclude_targets=motion_exclude_targets
        )
        self.robot = self.env.robot  # This is actually the humanoid
        
        # Mediapipe pose landmark indices
        self.landmark_indices = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
    
    def load_poses(self, poses_file):
        """Load extracted poses from JSON file"""
        with open(poses_file, 'r') as f:
            poses_data = json.load(f)
        
        poses = []
        for pose_data in poses_data:
            poses.append({
                'frame': pose_data['frame'],
                'keypoints': np.array(pose_data['keypoints']),
                'visibility': np.array(pose_data['visibility'])
            })
        
        return poses
    
    def convert_pose_to_joint_angles(self, keypoints):
        """Convert human pose keypoints to approximate joint angles"""
        joint_angles = {}
        
        try:
            # Extract key landmarks
            landmarks = {}
            for name, idx in self.landmark_indices.items():
                landmark = keypoints[idx]
                if not isinstance(landmark, (list, np.ndarray)) or len(landmark) < 2:
                    print(f"Warning: Invalid landmark {name} at index {idx}: {landmark}")
                    continue
                landmarks[name] = landmark
            
            # Calculate joint angles using vector math
            # Shoulder angles
            if 'left_hip' in landmarks and 'left_shoulder' in landmarks and 'left_elbow' in landmarks:
                joint_angles['left_shoulder'] = self._calculate_angle(
                    landmarks['left_hip'], landmarks['left_shoulder'], landmarks['left_elbow']
                )
            if 'right_hip' in landmarks and 'right_shoulder' in landmarks and 'right_elbow' in landmarks:
                joint_angles['right_shoulder'] = self._calculate_angle(
                    landmarks['right_hip'], landmarks['right_shoulder'], landmarks['right_elbow']
                )
            
            # Elbow angles
            if 'left_shoulder' in landmarks and 'left_elbow' in landmarks and 'left_wrist' in landmarks:
                joint_angles['left_elbow'] = self._calculate_angle(
                    landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']
                )
            if 'right_shoulder' in landmarks and 'right_elbow' in landmarks and 'right_wrist' in landmarks:
                joint_angles['right_elbow'] = self._calculate_angle(
                    landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']
                )
            
            # Hip angles
            if 'left_shoulder' in landmarks and 'left_hip' in landmarks and 'left_knee' in landmarks:
                joint_angles['left_hip'] = self._calculate_angle(
                    landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee']
                )
            if 'right_shoulder' in landmarks and 'right_hip' in landmarks and 'right_knee' in landmarks:
                joint_angles['right_hip'] = self._calculate_angle(
                    landmarks['right_shoulder'], landmarks['right_hip'], landmarks['right_knee']
                )
            
            # Knee angles
            if 'left_hip' in landmarks and 'left_knee' in landmarks and 'left_ankle' in landmarks:
                joint_angles['left_knee'] = self._calculate_angle(
                    landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle']
                )
            if 'right_hip' in landmarks and 'right_knee' in landmarks and 'right_ankle' in landmarks:
                joint_angles['right_knee'] = self._calculate_angle(
                    landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle']
                )
                
        except Exception as e:
            print(f"Error in convert_pose_to_joint_angles: {e}")
            print(f"Keypoints shape: {np.array(keypoints).shape}")
            return {}
        
        return joint_angles
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        # Convert to numpy arrays
        p1 = np.array(point1[:2])  # Use only x,y coordinates
        p2 = np.array(point2[:2])
        p3 = np.array(point3[:2])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return angle
    
    def map_to_humanoid_state(self, joint_angles, frame_idx):
        """Map joint angles to humanoid state representation"""
        # Get current humanoid state
        obs = self.env.reset()[0]
        
        # Create state vector based on joint angles
        # This is a simplified mapping - you may need to adjust based on your humanoid's joint configuration
        state = obs.copy()
        
        # Map calculated angles to humanoid joint positions
        # The humanoid has 17 actuated joints + 4 grasp actions = 21 total
        # We map the major body joints from the pose estimation
        angle_mapping = {
            'left_shoulder': 0,    # Left shoulder
            'right_shoulder': 1,   # Right shoulder
            'left_elbow': 2,       # Left elbow
            'right_elbow': 3,      # Right elbow
            'left_hip': 4,         # Left hip
            'right_hip': 5,        # Right hip
            'left_knee': 6,        # Left knee
            'right_knee': 7,       # Right knee
            # Additional joints filled with neutral values (index 8-20)
            # These could be: spine, neck, ankles, wrists, etc.
        }
        
        for joint_name, angle in joint_angles.items():
            if joint_name in angle_mapping:
                joint_idx = angle_mapping[joint_name]
                if joint_idx < len(state):
                    # Normalize angle to [-1, 1] range
                    # MediaPipe angles are in radians [0, π]
                    normalized_angle = (angle - np.pi/2) / (np.pi/2)
                    normalized_angle = np.clip(normalized_angle, -1.0, 1.0)
                    state[joint_idx] = normalized_angle
        
        return state
    
    def generate_action_from_state_transition(self, current_state, next_state):
        """Generate action that would transition from current to next state"""
        # Calculate state difference
        state_diff = next_state - current_state
        
        # Apply scaling with smoother transition
        # Use hyperbolic tangent for smooth scaling instead of hard clipping
        action = np.tanh(state_diff * 5.0)  # Scale factor reduced from 10 to 5 for smoother transitions
        
        # Ensure action matches environment's action space (21 dimensions)
        action_space_size = self.env.action_space.shape[0]
        
        if len(action) > action_space_size:
            # Truncate if action is too long
            action = action[:action_space_size]
        elif len(action) < action_space_size:
            # Pad with zeros if action is too short
            padded_action = np.zeros(action_space_size)
            padded_action[:len(action)] = action
            action = padded_action
        
        # Final clipping to ensure actions are in valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def convert_poses_to_demonstrations(self, poses_file, output_file):
        """Convert pose sequence to state-action demonstrations"""
        poses = self.load_poses(poses_file)
        
        demonstrations = {
            'states': [],
            'actions': [],
            'rewards': [],
            'metadata': {
                'source_video': str(poses_file),
                'total_frames': len(poses),
                'conversion_method': 'pose_to_joint_angles'
            }
        }
        
        states = []
        
        print(f"Converting {len(poses)} poses to state-action pairs...")
        
        for i, pose in enumerate(poses):
            # Convert pose to joint angles
            joint_angles = self.convert_pose_to_joint_angles(pose['keypoints'])
            
            # Map to humanoid state
            state = self.map_to_humanoid_state(joint_angles, pose['frame'])
            states.append(state)
            
            if i > 0:
                # Generate action from previous state to current state
                action = self.generate_action_from_state_transition(states[i-1], state)
                demonstrations['actions'].append(action.tolist())
                demonstrations['rewards'].append(1.0)  # Positive reward for demonstrated behavior
        
        # Add states (except the last one, which has no corresponding action)
        demonstrations['states'] = [state.tolist() for state in states[:-1]]
        
        # Save demonstrations
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(demonstrations, f, indent=2)
        
        print(f"✓ Saved {len(demonstrations['states'])} state-action pairs to {output_file}")
        
        return demonstrations

def convert_all_poses(poses_dir, output_dir):
    """Convert all pose files to demonstrations"""
    converter = PoseToStateActionConverter()
    poses_dir = Path(poses_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for poses_file in poses_dir.glob("*_poses.json"):
        print(f"\nProcessing {poses_file.name}...")
        
        try:
            output_file = output_dir / f"{poses_file.stem}_demonstrations.json"
            demonstrations = converter.convert_poses_to_demonstrations(poses_file, output_file)
            
            results.append({
                'poses_file': poses_file.name,
                'demonstrations_file': output_file.name,
                'state_action_pairs': len(demonstrations['states']),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'poses_file': poses_file.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save conversion summary
    summary_file = output_dir / "conversion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    poses_dir = "extracted_poses"
    output_dir = "demonstrations"
    
    if len(sys.argv) > 1:
        poses_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("🤖 Pose to State-Action Conversion Pipeline")
    print(f"📁 Input: {poses_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 50)
    
    results = convert_all_poses(poses_dir, output_dir)
    
    print("\n📊 Summary:")
    successful = sum(1 for r in results if r['status'] == 'success')
    total_pairs = sum(r.get('state_action_pairs', 0) for r in results)
    print(f"✓ {successful}/{len(results)} pose files converted successfully")
    print(f"✓ {total_pairs} total state-action pairs generated")
