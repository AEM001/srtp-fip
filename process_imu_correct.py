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
        # Root/Others: X-up, Y-right, Z-back (Standard RH)
        # Target: X-right, Y-up, Z-forward
        # Map: X->-Y (Gravity -1g), Y->X, Z->Z
        transform = np.array([
            [0, 1, 0],   # X_new = Y_old
            [-1, 0, 0],  # Y_new = -X_old
            [0, 0, 1]    # Z_new = Z_old
        ])
    elif imu_id == 2:
        # ID 2 (left hand): X-left, Y-forward, Z-up
        # Target: X-right, Y-up, Z-forward
        # Map: X->-X (Left), Z->-Y (Gravity -1g), Y->-Z (Back)
        transform = np.array([
            [-1, 0, 0],  # X_new = -X_old
            [0, 0, -1],  # Y_new = -Z_old
            [0, -1, 0]   # Z_new = -Y_old
        ])
    elif imu_id == 1:
        # ID 1 (right hand): X-right, Y-forward, Z-up
        # Target: X-right, Y-up, Z-forward
        # Map: X->X (Right), Z->-Y (Gravity -1g), Y->Z (Forward)
        transform = np.array([
            [1, 0, 0],   # X_new = X_old
            [0, 0, -1],  # Y_new = -Z_old
            [0, 1, 0]    # Z_new = Y_old
        ])
    else:
        transform = np.eye(3)
    
    return transform

def compute_calibration_offsets(df_tpose, root_id=5):
    """
    Compute calibration rotation matrices from T-pose data.
    Aligns all sensors to the Root sensor's orientation in T-pose.
    Assumption: In T-pose, all body parts are 'aligned' in the model's orientation space
    (i.e., relative rotations are Identity).
    """
    print("\nComputing T-pose calibration...")
    offsets = {}
    
    # Get average rotation matrix for Root
    root_data = df_tpose[df_tpose['imu_id'] == root_id]
    if len(root_data) == 0:
        print("Warning: Root IMU not found in calibration data!")
        return {}
        
    # Average rotation for Root (simple average of Euler angles for stability)
    # Actually, we should calibrate to IDENTITY (Target Global Frame).
    # In T-pose, we assume the person is standing upright (Root = Identity)
    # and Limbs are aligned with World axes as per SMPL T-pose (Identity).
    
    # Compute offsets for all IDs to make them Identity in T-pose
    for imu_id in df_tpose['imu_id'].unique():
        imu_data = df_tpose[df_tpose['imu_id'] == imu_id]
        
        avg_roll = imu_data['roll'].mean()
        avg_pitch = imu_data['pitch'].mean()
        avg_yaw = imu_data['yaw'].mean()
        R_meas = euler_to_rotation_matrix(avg_roll, avg_pitch, avg_yaw)
        
        # We want: R_meas @ R_offset = I (Identity)
        # So: R_offset = R_meas^T
        R_offset = R_meas.T
        offsets[int(imu_id)] = R_offset
        
    print("Calibration offsets computed (Target: Identity).")
    return offsets

def apply_coordinate_transform(accel, rotation_matrix, imu_id, calibration_matrix=None):
    """Apply coordinate transformation and optional calibration"""
    transform = get_coordinate_transform(imu_id)
    
    # Transform acceleration vector: v_body = T @ v_sensor
    accel_transformed = transform @ accel
    
    # Transform rotation matrix: R_body = R_sensor @ T^T
    # R maps Local -> Global. 
    # v_global = R_sensor @ v_sensor
    # v_body = T @ v_sensor => v_sensor = T^T @ v_body
    # v_global = R_sensor @ T^T @ v_body
    # So R_body = R_sensor @ T^T
    rotation_transformed = rotation_matrix @ transform.T
    
    # Apply Calibration: R_final = R_transformed @ R_calib
    if calibration_matrix is not None:
        rotation_transformed = rotation_transformed @ calibration_matrix
        
        # Note: We do NOT rotate acceleration by calibration matrix.
        # Calibration aligns the *Orientation Output* to be Identity.
        # Acceleration is a physical vector in the body frame.
        # If we just 'reset' the orientation to Identity, we are effectively
        # saying "The sensor was mounted with this rotation R_offset relative to the bone".
        # So we should also rotate the Acceleration vector into the Bone Frame?
        # v_bone = R_offset.T @ v_sensor_body?
        # NO. R_final maps Bone -> Global.
        # R_transformed maps SensorBody -> Global.
        # R_transformed @ R_calib = R_final.
        # This implies R_calib maps Bone -> SensorBody?
        # No, it's a post-multiply.
        # Let's trace:
        # v_global = R_transformed @ v_sensor_body.
        # We want v_global = R_final @ v_bone.
        # If we define v_bone = v_sensor_body (Sensor is aligned with bone in position, just rotated?),
        # No, if sensor is rotated, v_bone != v_sensor_body.
        
        # If we assume the "Calibration" is correcting for Sensor Mounting Rotation.
        # R_meas (at T-pose) should be Identity. It is R_err.
        # We apply R_offset = R_err.T.
        # So R_final = R_meas @ R_err.T = I.
        # This Rotation Correction implies the Sensor Frame is rotated relative to Bone Frame.
        # R_sensor = R_bone @ R_mount.
        # We want R_bone = R_sensor @ R_mount.T.
        # This matches our operation if R_mount = R_err.
        
        # Does Acceleration need to be rotated?
        # v_sensor = R_mount.T @ v_bone.
        # v_bone = R_mount @ v_sensor.
        # v_bone = R_err @ v_sensor.
        # So yes, we SHOULD rotate the acceleration vector by the inverse of the calibration matrix?
        # Wait. R_final = R_sensor @ R_offset.
        # R_offset = R_mount.T.
        # So R_bone = R_sensor @ R_mount.T.
        # And v_bone = R_mount @ v_sensor.
        # So v_bone = R_offset.T @ v_sensor.
        
        # However, in many IMU pipelines, we only calibrate Orientation.
        # Because 'accel' is used for position/velocity integration in Global Frame.
        # v_global_acc = R_final @ v_bone_acc.
        # v_global_acc = (R_sensor @ R_offset) @ (R_offset.T @ v_sensor).
        # v_global_acc = R_sensor @ v_sensor.
        # This remains unchanged!
        # Which is CORRECT. The physical acceleration in global space (Gravity) is invariant.
        # We are just changing the definition of the "Body Frame".
        # So we do NOT need to transform the acceleration vector values.
        # We just compute a new Orientation Matrix for the body.
        # And the Model uses (Acc, Ori) pairs.
        # If the Model assumes Acc is in the Body Frame (Local)?
        # infer_from_csv.py:
        # acc_root = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
        # It projects (Acc_Global - Acc_Root_Global) into the Root Frame.
        # Wait. `acc` in the CSV is treated as Global or Local?
        # The CSV has `accel_x, y, z`.
        # My code `accel_transformed = transform @ accel`.
        # This maps Sensor Local -> Body Local.
        # Is this Body Local treated as Global by the inference script?
        # "load_csv_to_tensors":
        # It reads accel.
        # "acc_root = ori... @ acc".
        # This implies `acc` is GLOBAL.
        # Because it projects it *into* Root frame using `ori.T`.
        # If `acc` was already Local, we wouldn't project it.
        
        # So `acc` must be in Global Frame.
        # BUT `transform @ accel` maps Sensor Local -> Body Local (Aligned with Global).
        # Wait.
        # If `transform` maps Sensor Axes to Global Axes (e.g. X->-Y).
        # Then `transform @ accel_sensor` = `accel_global`.
        # Correct.
        # So `accel_transformed` IS Global Acceleration.
        # And R_transformed IS Body Orientation (Local -> Global).
        
        # So we are good. We don't touch accel.
        pass
    
    return accel_transformed, rotation_transformed

def fill_and_transform_data(csv_file, output_file, id2_avg=None, calibration_offsets=None):
    """
    Fill missing IMU data in RAW format, then apply coordinate transformations and calibration
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
    
    # Step 2: Apply coordinate transformations and Calibration
    transformed_data = []
    
    for idx, row in df_filled.iterrows():
        imu_id = int(row['imu_id'])
        accel = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        rotation_matrix = euler_to_rotation_matrix(row['roll'], row['pitch'], row['yaw'])
        
        # Get calibration matrix for this ID
        calib_mat = None
        if calibration_offsets and imu_id in calibration_offsets:
            calib_mat = calibration_offsets[imu_id]
            
        # Apply coordinate transformation
        accel_transformed, rotation_transformed = apply_coordinate_transform(
            accel, rotation_matrix, imu_id, calib_mat
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
    print("IMU Data Processing - Correct Version with Calibration")
    print("="*70)
    print("\nCoordinate Transformations:")
    print("  IMU 0,3,4,5: Z-forward, X-up, Y-right → X-right, Y-up, Z-forward")
    print("  IMU 2 (left hand): X-left, Y-back, Z-up → X-right, Y-up, Z-forward")
    print("  IMU 1 (right hand): X-right, Y-forward, Z-up → X-right, Y-up, Z-forward")
    print("  Calibration: All sensors aligned to Root orientation in T-pose (1.csv)")
    print("="*70)
    
    # First Pass: Process 1.csv without calibration to get base T-pose data
    print("\n[Phase 1] Analyzing 1.csv for Calibration...")
    df1_base = fill_and_transform_data(
        os.path.join(csv_dir, '1.csv'),
        os.path.join(output_dir, '1_temp.csv'),
        id2_avg=None,
        calibration_offsets=None
    )
    
    # Compute Calibration Offsets
    calibration_offsets = compute_calibration_offsets(df1_base)
    
    # Prepare ID 2 average for filling
    df1_raw = pd.read_csv(os.path.join(csv_dir, '1.csv'))
    id2_avg_raw = df1_raw[df1_raw['imu_id'] == 2][['accel_x', 'accel_y', 'accel_z', 'roll', 'pitch', 'yaw']].mean()
    
    # Phase 2: Process all files WITH calibration
    print("\n[Phase 2] Processing all files with Calibration...")
    
    # Process 1.csv
    print("\n[1/3] Processing 1.csv (Final)")
    df1 = fill_and_transform_data(
        os.path.join(csv_dir, '1.csv'),
        os.path.join(output_dir, '1_final.csv'),
        id2_avg=None,
        calibration_offsets=calibration_offsets
    )
    
    # Process 2.csv
    print("\n[2/3] Processing 2.csv")
    df2 = fill_and_transform_data(
        os.path.join(csv_dir, '2.csv'),
        os.path.join(output_dir, '2_final.csv'),
        id2_avg=id2_avg_raw,
        calibration_offsets=calibration_offsets
    )
    
    # Process 3.csv
    print("\n[3/3] Processing 3.csv")
    df3 = fill_and_transform_data(
        os.path.join(csv_dir, '3.csv'),
        os.path.join(output_dir, '3_final.csv'),
        id2_avg=id2_avg_raw,
        calibration_offsets=calibration_offsets
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
