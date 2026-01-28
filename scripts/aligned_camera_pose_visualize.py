import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def read_colmap_images(images_file):
    poses = []
    if not os.path.exists(images_file):
        print("Error: File {} not found.".format(images_file))
        return []

    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        
        # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        
        q = np.array([qw, qx, qy, qz])
        t = np.array([tx, ty, tz])
        
        R = quaternion_to_rotation_matrix(q)
        
        # Camera center in world coordinates: C = -R^T * t
        center = -R.T @ t
        
        poses.append({
            'id': image_id,
            'name': name,
            'R': R,
            't': t,
            'center': center,
            'q': q
        })
        
        # Skip the next line (points2D info)
        i += 2
        
    return poses

def save_poses_to_file(poses, output_path):
    if not poses:
        return
    with open(output_path, 'w') as f:
        f.write("# image_name x y z qx qy qz qw\n")
        for p in poses:
            name = p['name']
            x, y, z = p['center']
            qw, qx, qy, qz = p['q']
            # Inverse quaternion for Camera -> World orientation
            # COLMAP q is World->Camera. We want Camera->World.
            # Conjugate: (qw, -qx, -qy, -qz)
            f.write("{} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                name, x, y, z, -qx, -qy, -qz, qw))
    print("Poses saved to {}".format(output_path))

def visualize_poses(poses, output_file=None):
    if not poses:
        print("No poses to visualize.")
        return

    fig = plt.figure(figsize=(12, 10))
    # Use 2D plot for orthographic projection (Top-Down view)
    ax = fig.add_subplot(111)
    
    centers = np.array([p['center'] for p in poses])
    
    # Plot camera centers
    colors = np.arange(len(centers))
    sc = ax.scatter(centers[:, 0], centers[:, 1], c=colors, cmap='viridis', s=20, marker='o', label='Camera Centers')
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Image Sequence/ID Order')

    # Draw trajectory line
    ax.plot(centers[:, 0], centers[:, 1], color='gray', alpha=0.5, linewidth=0.5)

    # Removed orientation (pose) visualization as requested

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Camera Trajectory (2D Projection) - {} images'.format(len(poses)))
    
    ax.axis('equal')
    ax.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print("Visualization saved to {}".format(output_file))
    else:
        plt.show()

def get_sort_key(p):
    """Sort key function to extract number from filename."""
    # Extract number from filename like 'cameraImage_100.jpg' or 'camera_Image_100'
    match = re.search(r'(\d+)', p['name'])
    if match:
        return int(match.group(1))
    return p['name'] # Fallback to string sort if no number found

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP camera poses from images.txt.")
    parser.add_argument("images_txt", help="Path to COLMAP images.txt file", default="/home/mark50/dataset/colmap3.13_images/2026-01-15/aligned/images.txt")
    parser.add_argument("--output", help="Path to save the output plot", default="colmap_poses.png")
    parser.add_argument("--save_txt", help="Path to save sorted poses as txt file (format: name x y z qx qy qz qw)", default=None)
    
    args = parser.parse_args()
    
    poses = read_colmap_images(args.images_txt)
    
    # Sort poses by name to approximate temporal sequence if names are numbered
    if poses:
        poses.sort(key=get_sort_key) 

    print("Loaded {} camera poses.".format(len(poses)))
    
    if args.save_txt:
        save_poses_to_file(poses, args.save_txt)

    visualize_poses(poses, args.output)
