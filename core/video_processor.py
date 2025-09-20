#!/usr/bin/env python3
"""
Core Video Processing Pipeline
Extracts poses from climbing videos and converts them to state-action pairs for behavioral cloning.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import sys
import os

class ClimbingVideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def extract_poses_from_video(self, video_path, output_dir=None):
        """Extract pose keypoints from climbing video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        poses = []
        frames = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame for pose detection
            results = self.pose_detector.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract keypoints
                keypoints = self._extract_keypoints(results.pose_landmarks)
                poses.append({
                    'frame': frame_count,
                    'keypoints': keypoints,
                    'visibility': self._extract_visibility(results.pose_landmarks)
                })
                
                # Draw pose on frame for visualization
                annotated_frame = frame.copy()
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                frames.append(annotated_frame)
            
            frame_count += 1
            
        cap.release()
        
        # Save results
        if output_dir:
            self._save_poses(poses, video_path, output_dir)
            self._save_video_with_poses(frames, video_path, output_dir)
            
        return poses
    
    def _extract_keypoints(self, landmarks):
        """Extract 3D keypoints from pose landmarks"""
        keypoints = []
        for landmark in landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints)
    
    def _extract_visibility(self, landmarks):
        """Extract visibility scores for each landmark"""
        visibility = []
        for landmark in landmarks.landmark:
            visibility.append(landmark.visibility)
        return np.array(visibility)
    
    def _save_poses(self, poses, video_path, output_dir):
        """Save extracted poses to file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        poses_file = output_dir / f"{video_name}_poses.json"
        
        # Convert numpy arrays to lists for JSON serialization
        poses_serializable = []
        for pose in poses:
            pose_data = {
                'frame': pose['frame'],
                'keypoints': pose['keypoints'].tolist(),
                'visibility': pose['visibility'].tolist()
            }
            poses_serializable.append(pose_data)
        
        with open(poses_file, 'w') as f:
            json.dump(poses_serializable, f, indent=2)
    
    def _save_video_with_poses(self, frames, video_path, output_dir):
        """Save video with pose annotations"""
        if not frames:
            return
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        output_video = output_dir / f"{video_name}_with_poses.mp4"
        
        # Video properties
        height, width = frames[0].shape[:2]
        fps = 30  # Default FPS
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()

def process_all_videos(videos_dir, output_dir):
    """Process all videos in directory"""
    processor = ClimbingVideoProcessor()
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    
    results = []
    
    for video_file in videos_dir.glob("*.mp4"):
        print(f"Processing {video_file.name}...")
        
        try:
            poses = processor.extract_poses_from_video(str(video_file), output_dir)
            results.append({
                'video': video_file.name,
                'poses_extracted': len(poses),
                'status': 'success'
            })
            print(f"  ✓ Extracted {len(poses)} poses")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'video': video_file.name,
                'poses_extracted': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Save processing summary
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    videos_dir = "videos"
    output_dir = "extracted_poses"
    
    if len(sys.argv) > 1:
        videos_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("🎬 Climbing Video Pose Extraction Pipeline")
    print(f"📁 Input: {videos_dir}")
    print(f"📁 Output: {output_dir}")
    print("-" * 50)
    
    results = process_all_videos(videos_dir, output_dir)
    
    print("\n📊 Summary:")
    successful = sum(1 for r in results if r['status'] == 'success')
    total_poses = sum(r['poses_extracted'] for r in results)
    print(f"✓ {successful}/{len(results)} videos processed successfully")
    print(f"✓ {total_poses} total poses extracted")
