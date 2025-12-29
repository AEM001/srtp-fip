import torch
import numpy as np
import trimesh
import pyrender
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'human_body_prior'))
from src import kinematic_model
from src.eval_tools import glb2local

def main():
    device = 'cpu'
    smpl_file = 'data/SMPL_male.pkl'
    
    print("Loading SMPL model...")
    m = kinematic_model.ParametricModel(smpl_file, device=device)
    
    # Create an Identity Pose (Global) for 15 joints
    # Shape: (1, 15, 3, 3)
    poses_identity = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 15, 1, 1).to(device)
    
    print("Testing Identity Pose (15 joints)...")
    
    # Convert to SMPL local
    poses_local = glb2local(poses_identity)
    
    # Override same joints as visualize_final.py
    poses_local[:, [0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=device)
    
    # Forward Kinematics
    _, _, mesh = m.forward_kinematics(poses_local, calc_mesh=True)
    verts = mesh[0].cpu().numpy()
    faces = m.face
    
    # Render
    mesh_trimesh = trimesh.Trimesh(vertices=verts, faces=faces)
    scene = pyrender.Scene()
    py_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene.add(py_mesh)
    
    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
    color, _ = r.render(scene)
    
    import imageio
    imageio.imwrite('test_identity.png', color)
    print("Saved test_identity.png")

if __name__ == "__main__":
    main()
