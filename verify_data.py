import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('mydata/processed_final/1_final.csv')
        print("Data loaded. Shape:", df.shape)
        
        # Check first timestamp
        t0 = df['timestamp'].unique()[0]
        df_t0 = df[df['timestamp'] == t0]
        
        print(f"\nFirst Frame (t={t0}):")
        print(df_t0[['imu_id', 'roll', 'pitch', 'yaw', 'accel_x', 'accel_y', 'accel_z']].to_string())
        
        # Check means
        print("\nMeans of first 10 frames:")
        print(df.head(10*6).groupby('imu_id')[['roll', 'pitch', 'yaw']].mean())
        
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
