"""
Correct IMU data processing with proper coordinate transformations
Key insight: Fill missing data in RAW format first, then transform
"""
import os
import numpy as np
import pandas as pd

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (degrees) to rotation matrix"""
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x

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

def get_coordinate_transform(imu_id):
    """
    Get coordinate transformation matrix for each IMU sensor.
    
    IMU orientations (body-centric):
    - IMU 0,3,4,5: Z-forward, X-up, Y-right
    - IMU 2 (left hand): X-forward, Y-left, Z-up
    - IMU 1 (right hand): X-forward, Y-right, Z-up (mirror of ID 2)
    
    Target (project): X-right, Y-up, Z-forward
    """
    if imu_id in [0, 3, 4, 5]:
        transform = np.array([
            [0, 1, 0],   # X_new = Y_old (right)
            [1, 0, 0],   # Y_new = X_old (up)
            [0, 0, 1]    # Z_new = Z_old (forward)
        ])
    elif imu_id == 2:
        # ID 2 (left hand): X-forward, Y-left, Z-up
        transform = np.array([
            [0, -1, 0],  # X_new = -Y_old (right = -left)
            [0, 0, 1],   # Y_new = Z_old (up)
            [1, 0, 0]    # Z_new = X_old (forward)
        ])
    elif imu_id == 1:
        # ID 1 (right hand): X-forward, Y-right, Z-up
        transform = np.array([
            [0, 1, 0],   # X_new = Y_old (right)
            [0, 0, 1],   # Y_new = Z_old (up)
            [1, 0, 0]    # Z_new = X_old (forward)
        ])
    else:
        transform = np.eye(3)
    
    return transform

def apply_coordinate_transform(accel, rotation_matrix, imu_id):
    """Apply coordinate transformation to acceleration and rotation data"""
    transform = get_coordinate_transform(imu_id)
    
    # Transform acceleration vector
    accel_transformed = transform @ accel
    
    # Transform rotation matrix: R_new = T @ R_old @ T^T
    rotation_transformed = transform @ rotation_matrix @ transform.T
    
    return accel_transformed, rotation_transformed

def fill_and_transform_data(csv_file, output_file, id2_avg=None):
    """
    Fill missing IMU data in RAW format, then apply coordinate transformations
    """
    print(f"\nProcessing {csv_file}...")
    
    # Read raw CSV
    df = pd.read_csv(csv_file)
    
    # Step 1: Fill missing data in RAW format
    filled_data = []
    
    for timestamp in df['timestamp'].unique():
        frame_data = df[df['timestamp'] == timestamp]
        
        # Get existing IMU data
        imu_data = {}
        for _, row in frame_data.iterrows():
            imu_data[int(row['imu_id'])] = row
        
        # Fill ID 1 if missing
        if 1 not in imu_data:
            if 2 in imu_data:
                # For 1.txt: copy from ID 2 in same file
                id1_raw = imu_data[2].copy()
                id1_raw['imu_id'] = 1
                filled_data.append(id1_raw)
            elif id2_avg is not None:
                # For 2.txt/3.txt: use average from 1.txt
                id1_raw = id2_avg.copy()
                id1_raw['timestamp'] = timestamp
                id1_raw['imu_id'] = 1
                filled_data.append(id1_raw)
        
        # Fill ID 3, 4 from ID 5
        if 5 in imu_data:
            for new_id in [3, 4]:
                if new_id not in imu_data:
                    id_raw = imu_data[5].copy()
                    id_raw['imu_id'] = new_id
                    filled_data.append(id_raw)
        
        # Keep existing data
        for imu_id in [0, 1, 2, 5]:
            if imu_id in imu_data:
                filled_data.append(imu_data[imu_id])
    
    df_filled = pd.DataFrame(filled_data).sort_values(['timestamp', 'imu_id']).reset_index(drop=True)
    
    # Step 2: Apply coordinate transformations
    transformed_data = []
    
    for idx, row in df_filled.iterrows():
        imu_id = int(row['imu_id'])
        accel = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        rotation_matrix = euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])
        
        # Apply coordinate transformation
        accel_transformed, rotation_transformed = apply_coordinate_transform(
            accel, rotation_matrix, imu_id
        )
        
        # Convert back to Euler angles
        roll_new, pitch_new, yaw_new = rotation_matrix_to_euler(rotation_transformed)
        
        transformed_data.append({
            'timestamp': row['timestamp'],
            'imu_id': imu_id,
            'accel_x': accel_transformed[0],
            'accel_y': accel_transformed[1],
            'accel_z': accel_transformed[2],
            'roll': roll_new,
            'pitch': pitch_new,
            'yaw': yaw_new
        })
    
    df_out = pd.DataFrame(transformed_data)
    df_out.to_csv(output_file, index=False)
    print(f"  Saved {len(df_out)} records to {output_file}")
    
    return df_out

def main():
    csv_dir = './mydata/csv'
    output_dir = './mydata/processed_final'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("IMU Data Processing - Correct Version")
    print("="*70)
    print("\nCoordinate Transformations:")
    print("  IMU 0,3,4,5: Z-forward, X-up, Y-right → X-right, Y-up, Z-forward")
    print("  IMU 2 (left hand): X-forward, Y-left, Z-up → X-right, Y-up, Z-forward")
    print("  IMU 1 (right hand): X-forward, Y-right, Z-up → X-right, Y-up, Z-forward")
    print("  Note: ID 1 and ID 2 have different IMU orientations (mirror symmetric)")
    print("  Target: X-right, Y-up, Z-forward")
    print("="*70)
    
    # Process 1.csv first (T-pose)
    print("\n[1/3] Processing 1.csv (T-pose calibration)")
    print("  - Fill ID 1 from ID 2 (raw data)")
    print("  - Fill ID 3,4 from ID 5 (raw data)")
    print("  - Then apply coordinate transformations")
    
    df1 = fill_and_transform_data(
        os.path.join(csv_dir, '1.csv'),
        os.path.join(output_dir, '1_final.csv'),
        id2_avg=None
    )
    
    # Calculate average of RAW ID 2 data from 1.csv for use in 2.csv and 3.csv
    df1_raw = pd.read_csv(os.path.join(csv_dir, '1.csv'))
    id2_avg_raw = df1_raw[df1_raw['imu_id'] == 2][['accel_x', 'accel_y', 'accel_z', 'roll', 'pitch', 'yaw']].mean()
    
    print(f"\nAverage RAW ID 2 data from 1.csv (for filling ID 1 in 2.csv/3.csv):")
    print(f"  Accel: ({id2_avg_raw['accel_x']:.3f}, {id2_avg_raw['accel_y']:.3f}, {id2_avg_raw['accel_z']:.3f})")
    print(f"  Orientation: Roll={id2_avg_raw['roll']:.1f}°, Pitch={id2_avg_raw['pitch']:.1f}°, Yaw={id2_avg_raw['yaw']:.1f}°")
    
    # Process 2.csv
    print("\n[2/3] Processing 2.csv")
    print("  - Fill ID 1 with average RAW ID 2 from 1.csv")
    print("  - Fill ID 3,4 from ID 5")
    print("  - Then apply coordinate transformations")
    
    df2 = fill_and_transform_data(
        os.path.join(csv_dir, '2.csv'),
        os.path.join(output_dir, '2_final.csv'),
        id2_avg=id2_avg_raw
    )
    
    # Process 3.csv
    print("\n[3/3] Processing 3.csv")
    print("  - Fill ID 1 with average RAW ID 2 from 1.csv")
    print("  - Fill ID 3,4 from ID 5")
    print("  - Then apply coordinate transformations")
    
    df3 = fill_and_transform_data(
        os.path.join(csv_dir, '3.csv'),
        os.path.join(output_dir, '3_final.csv'),
        id2_avg=id2_avg_raw
    )
    
    print("\n" + "="*70)
    print("Processing completed!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}:")
    print("  - 1_final.csv, 2_final.csv, 3_final.csv")
    print("\nVerify gravity direction (should be ~+1.0g in Y-axis after transformation)")
    
    for name, df in [('1', df1), ('2', df2), ('3', df3)]:
        avg_accel = df[['accel_x', 'accel_y', 'accel_z']].mean()
        print(f"\n{name}.csv average acceleration:")
        print(f"  X={avg_accel['accel_x']:.3f}g, Y={avg_accel['accel_y']:.3f}g, Z={avg_accel['accel_z']:.3f}g")
        if 0.8 < avg_accel['accel_y'] < 1.1:
            print(f"  ✓ Y-axis gravity correct!")
        else:
            print(f"  ✗ Warning: Y-axis gravity = {avg_accel['accel_y']:.3f}g")

if __name__ == "__main__":
    main()
