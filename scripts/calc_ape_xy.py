import os
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
import argparse
import re

def get_sort_key(filename):
    """Sort key function to extract number from filename."""
    # Extract number from filename like 'cameraImage_100.jpg' or 'camera_Image_100'
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return filename # Fallback to string sort if no number found

def read_colmap_poses(filepath):
    """
    Read COLMAP poses from txt file.
    Format: image_name x y z qx qy qz qw
    """
    poses = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            image_name = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6]) # Notice the order in file vs variable
            # Although the file says qx qy qz qw, let's keep it simple and just read position for APE
            poses[image_name] = np.array([x, y]) # We only care about XY plan
    return poses

def read_gps_data(filepath):
    """
    Read GPS data from txt file.
    Format: image_name lat lon alt
    """
    gps_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            image_name = parts[0]
            lat, lon, alt = float(parts[1]), float(parts[2]), float(parts[3])
            gps_data[image_name] = (lat, lon, alt)
    return gps_data

def align_and_calc_ape(colmap_poses, gps_data):
    """
    Compute APE between COLMAP trajectory and GPS trajectory.
    1. Convert GPS (LatLon) to local Flat (ENU/NED) coordinates.
    2. Finding corresponding points.
    3. Align trajectories (Sim3 or Rigid Body Transform). 
       Since absolute GPS vs arbitrary COLMAP frame, we definitely need alignment (Scale, Rotation, Translation).
    """

    common_images = sorted(list(set(colmap_poses.keys()) & set(gps_data.keys())), key=get_sort_key)
    if not common_images:
        print("No common images found between COLMAP poses and GPS data.")
        return

    print(f"Found {len(common_images)} common frames.")

    # 1. Prepare data arrays
    colmap_points = []
    gps_raw_points = []

    for img in common_images:
        colmap_points.append(colmap_poses[img])
        gps_raw_points.append(gps_data[img])

    colmap_points = np.array(colmap_points) # Nx2
    gps_raw_points = np.array(gps_raw_points) # Nx3 (lat, lon, alt)

    # 2. Convert GPS to local cartesian (ENU)
    # Use the first point as origin
    lat0, lon0, h0 = gps_raw_points[0]
    gps_enu = []
    for i in range(len(gps_raw_points)):
        lat, lon, h = gps_raw_points[i]
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        gps_enu.append([e, n]) # Only keep EN (XY)
    
    gps_enu = np.array(gps_enu)

    # 3. Umeyama alignment (Compute S, R, t to align Model to GT)
    # GT: gps_enu (2D), Model: colmap_points (2D)
    # T_model = s * R * model + t
    # Minimal implementation for 2D alignment
    
    gt_mean = np.mean(gps_enu, axis=0)
    model_mean = np.mean(colmap_points, axis=0)
    
    gt_centered = gps_enu - gt_mean
    model_centered = colmap_points - model_mean
    
    # Calculate scale, rotation, translation
    # Covariance matrix H = Model_centered^T * GT_centered
    H = model_centered.T @ gt_centered
    
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    # Scale
    # User requested NO scale calculation (assume scale=1.0, Rigid Body Transform instead of Sim3)
    scale = 1.0
    
    # Previous scale estimation logic (commented out):
    # var_model = np.sum(np.square(model_centered)) / len(model_centered)
    # scale = (1/var_model) * np.trace(np.diag(S)) 
    # dist_gt = np.mean(np.linalg.norm(gt_centered, axis=1))
    # dist_model = np.mean(np.linalg.norm(model_centered, axis=1))
    # scale = dist_gt / dist_model

    t = gt_mean - scale * (model_mean @ R.T)

    # Apply transformation
    colmap_aligned = scale * (colmap_points @ R.T) + t

    # 4. Compute APE (Absolute Pose Error) - Euclidean distance in XY
    errors = np.linalg.norm(gps_enu - colmap_aligned, axis=1)
    
    rmse = np.sqrt(np.mean(errors**2))
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    min_error = np.min(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    print("\n=== APE (Absolute Pose Error) Text Report (XY Plane) ===")
    print(f"Alignment: Computed Scale = {scale:.4f}")
    print(f"RMSE:   {rmse:.4f} m")
    print(f"Mean:   {mean_error:.4f} m")
    print(f"Median: {median_error:.4f} m")
    print(f"Min:    {min_error:.4f} m")
    print(f"Max:    {max_error:.4f} m")
    print(f"StdDev: {std_error:.4f} m")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Trajectory comparison
    plt.subplot(1, 2, 1)
    plt.plot(gps_enu[:, 0], gps_enu[:, 1], 'g-', label='GPS (Ground Truth)', alpha=0.7)
    plt.plot(colmap_aligned[:, 0], colmap_aligned[:, 1], 'b--', label='COLMAP (Aligned)', alpha=0.7)
    plt.plot(gps_enu[0, 0], gps_enu[0, 1], 'go', label='Start')
    plt.title("Aligned 2D Trajectories")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

    # Error distribution
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=30, color='orange', edgecolor='black', alpha=0.7)
    plt.axvline(mean_error, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}m')
    plt.axvline(median_error, color='b', linestyle='dashed', linewidth=1, label=f'Median: {median_error:.2f}m')
    plt.title("Absolute Pose Error (XY) Histogram")
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    output_img = "/home/mark50/MapMatching_ws/colmapTools/output/evaluate/ape_analysis.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\nAnalysis plot saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate APE between COLMAP poses and GPS.")
    parser.add_argument("--colmap", default="../output/colmap_poses.txt", help="Path to colmap poses txt")
    parser.add_argument("--gps", default="../output/gps.txt", help="Path to gps txt")
    
    args = parser.parse_args()
    
    colmap_poses = read_colmap_poses(args.colmap)
    gps_data = read_gps_data(args.gps)
    
    align_and_calc_ape(colmap_poses, gps_data)
