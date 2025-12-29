import torch
import numpy as np
import sys

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (degrees)"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def main():
    try:
        data = torch.load('mydata/processed_final/1_poses.pt', weights_only=False)
        poses = data['poses'] # Shape: (T, 15, 3, 3)
        
        print(f"Loaded poses: {poses.shape}")
        
        # Check Frame 0 (should be T-pose)
        print("\nFrame 0 Joint Rotations (Euler Degrees):")
        print(f"{'Joint':<6} {'Roll':<10} {'Pitch':<10} {'Yaw':<10}")
        print("-" * 40)
        
        frame0 = poses[0].numpy()
        for i in range(15):
            r, p, y = rotation_matrix_to_euler(frame0[i])
            print(f"{i:<6} {r:>9.2f} {p:>9.2f} {y:>9.2f}")
            
        # Check Frame 10 (Stability)
        print("\nFrame 10 Joint Rotations:")
        frame10 = poses[10].numpy()
        for i in range(15):
            r, p, y = rotation_matrix_to_euler(frame10[i])
            print(f"{i:<6} {r:>9.2f} {p:>9.2f} {y:>9.2f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
