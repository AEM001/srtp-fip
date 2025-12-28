# IMU坐标转换说明

## 坐标系统定义

### 项目坐标系（目标坐标系）
- **X轴**：向右
- **Y轴**：向上
- **Z轴**：向前

### IMU原始坐标系（以人体为本体坐标系）

#### IMU 0, 3, 4, 5（躯干和腿部）
- **Z轴**：向前
- **X轴**：向上
- **Y轴**：向右

#### IMU 2（左手）
- **X轴**：向前
- **Y轴**：向左（外侧）
- **Z轴**：向上

#### IMU 1（右手）
- **X轴**：向前
- **Y轴**：向右（外侧）
- **Z轴**：向上

**关键点**：ID 1和ID 2虽然都是Z轴朝上，但Y轴方向相反（左右镜像对称）

## 坐标转换矩阵

### IMU 0, 3, 4, 5
```
原始: Z-前, X-上, Y-右
目标: X-右, Y-上, Z-前

转换矩阵:
┌         ┐
│ 0  1  0 │  X_new = Y_old (右)
│ 1  0  0 │  Y_new = X_old (上)
│ 0  0  1 │  Z_new = Z_old (前)
└         ┘
```

### IMU 2（左手）
```
原始: X-前, Y-左, Z-上
目标: X-右, Y-上, Z-前

转换矩阵:
┌          ┐
│ 0  -1  0 │  X_new = -Y_old (右 = -左)
│ 0   0  1 │  Y_new = Z_old (上)
│ 1   0  0 │  Z_new = X_old (前)
└          ┘
```

### IMU 1（右手）
```
原始: X-前, Y-右, Z-上
目标: X-右, Y-上, Z-前

转换矩阵:
┌         ┐
│ 0  1  0 │  X_new = Y_old (右)
│ 0  0  1 │  Y_new = Z_old (上)
│ 1  0  0 │  Z_new = X_old (前)
└         ┘
```

## 数据处理流程

### 1. 填充缺失数据（在原始坐标系下）
- **1.txt（T-pose）**：
  - ID 1 ← 复制 ID 2的原始数据
  - ID 3, 4 ← 复制 ID 5的原始数据
  
- **2.txt, 3.txt**：
  - ID 1 ← 使用1.txt中ID 2的平均原始数据
  - ID 3, 4 ← 复制 ID 5的原始数据

### 2. 应用坐标转换
对每个IMU应用各自的转换矩阵：
- **加速度向量**：`a_new = T @ a_old`
- **旋转矩阵**：`R_new = T @ R_old @ T^T`

### 3. 验证
转换后重力应在Y轴正方向，约+1.0g：
- 1.txt（T-pose）：Y = +0.952g ✓
- 2.txt：Y = +0.783g ✓（运动中）
- 3.txt：Y = +0.682g ✓（运动中）

## Python实现

```python
import numpy as np

def get_coordinate_transform(imu_id):
    """获取IMU坐标转换矩阵"""
    if imu_id in [0, 3, 4, 5]:
        # 躯干和腿部
        return np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    elif imu_id == 2:
        # 左手
        return np.array([
            [0, -1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
    elif imu_id == 1:
        # 右手
        return np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
    else:
        return np.eye(3)

def apply_coordinate_transform(accel, rotation_matrix, imu_id):
    """应用坐标转换"""
    T = get_coordinate_transform(imu_id)
    
    # 转换加速度
    accel_new = T @ accel
    
    # 转换旋转矩阵
    rotation_new = T @ rotation_matrix @ T.T
    
    return accel_new, rotation_new
```

## 使用脚本

**正确的处理脚本**：`process_imu_correct.py`

运行命令：
```bash
python3 process_imu_correct.py
```

输出文件位置：`mydata/processed_final/`
- `1_final.csv`, `2_final.csv`, `3_final.csv`

## 注意事项

1. **必须先填充原始数据，再应用坐标转换**
2. **ID 1和ID 2使用不同的转换矩阵**（关键！）
3. **T-pose中左右手应该对称**
4. **转换后Y轴重力应为正值**（约+1.0g）

---
*最后更新：2025-12-28*
