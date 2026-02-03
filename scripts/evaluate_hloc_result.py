import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re

def get_sort_key(filename):
    """Extract number from filename for sorting."""
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return filename

def qvec2rotmat(qvec):
    """Convert quaternion (qw, qx, qy, qz) to rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_colmap_poses(path):
    """
    Read reference poses (already centers).
    Format: image_name x y z qx qy qz qw
    """
    poses = {}
    if not os.path.exists(path):
        print("Warning: Reference file {} not found.".format(path))
        return poses
        
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 4: continue
            name = parts[0]
            # Read position directly
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            poses[name] = np.array([x, y, z])
    return poses

def read_hloc_result(path):
    """
    Read HLoc result (usually World-to-Camera poses).
    Format inferred: image_name qw qx qy qz tx ty tz
    """
    poses = {}
    if not os.path.exists(path):
        print("Error: HLoc file {} not found.".format(path))
        return poses

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            name = parts[0]
            values = [float(v) for v in parts[1:]]
            
            if len(values) >= 7:
                # Assuming qw, qx, qy, qz, tx, ty, tz
                qvec = np.array(values[0:4])
                tvec = np.array(values[4:7])
                
                # Convert World-to-Camera (T_w2c) to Camera Center (C)
                # C = -R^T * t
                R = qvec2rotmat(qvec)
                center = -R.T @ tvec
                poses[name] = center
                
    return poses

def main():
    parser = argparse.ArgumentParser(description="Visualize HLoc trajectory vs COLMAP reference.")
    parser.add_argument("--colmap", default="../output/colmap_poses.txt", help="Path to reference colmap poses")
    parser.add_argument("--hloc", required=True, help="Path to hloc result txt")
    parser.add_argument("--output", default="../output/hloc_eval_traj.png", help="Output plot path")
    parser.add_argument("--threshold", type=float, default=50.0, help="Outlier threshold in meters (adjacent frame distance)")
    
    args = parser.parse_args()
    
    ref_poses = read_colmap_poses(args.colmap)
    hloc_poses = read_hloc_result(args.hloc)
    
    if not hloc_poses:
        print("No HLoc poses loaded.")
        return

    # Sort keys to ensure correct line connectivity
    ref_keys = sorted(ref_poses.keys(), key=get_sort_key)
    hloc_keys = sorted(hloc_poses.keys(), key=get_sort_key)
    
    # Filter outliers based on "Spike" logic:
    # A point is an outlier if it is far (> 20m) from BOTH its previous and next neighbor.
    # This detects single points that "fly away" from the smooth trajectory.
    
    threshold = 50.0
    print("Filtering outliers (Spike detection, threshold: {}m)...".format(threshold))
    
    # We might need multiple passes to handle consecutive outliers
    max_passes = 3
    for pass_idx in range(max_passes):
        keys_to_remove = []
        n_keys = len(hloc_keys)
        
        if n_keys < 3:
            break
            
        for i in range(n_keys):
            curr_pos = hloc_poses[hloc_keys[i]]
            
            # Check deviation from previous
            d_prev = float('inf')
            if i > 0:
                prev_pos = hloc_poses[hloc_keys[i-1]]
                d_prev = np.linalg.norm(curr_pos - prev_pos)
            
            # Check deviation from next
            d_next = float('inf')
            if i < n_keys - 1:
                next_pos = hloc_poses[hloc_keys[i+1]]
                d_next = np.linalg.norm(curr_pos - next_pos)
            
            # Logic: If disconnected from both sides (or from only existing side)
            # Boundary cases:
            # - Start: if d_next > 20 -> outlier? (Maybe just the start of a diff segment, but let's be strict for "flying")
            # - End: if d_prev > 20 -> outlier?
            # - Middle: if d_prev > 20 AND d_next > 20 -> outlier.
            
            is_outlier = False
            if i == 0:
                if d_next > threshold: is_outlier = True
            elif i == n_keys - 1:
                if d_prev > threshold: is_outlier = True
            else:
                if d_prev > threshold and d_next > threshold:
                    is_outlier = True
            
            # Note: Checking boundaries strictly might kill valid segments if they are far apart. 
            # But "flying points" usually implies single bad points. 
            # If d_prev > 20 and d_next < 20, the point is attached to the right, so it's valid (start of a segment).
            # If d_prev < 20 and d_next > 20, the point is attached to the left, so it's valid (end of a segment).
            # ONLY if d_prev > 20 AND d_next > 20 is it truly isolated.
            
            # Refined Logic for boundaries to avoid killing segment ends:
            if i == 0:
                # Can't judge purely on next, look at next next? 
                # Simplest: Don't kill start/end unless we are sure.
                # Let's start with strict middle checking.
                pass 
            elif i == n_keys - 1:
                 pass
            else: # Middle
                 if d_prev > threshold and d_next > threshold:
                     is_outlier = True
                     print("  Pass {}: Outlier removed: {} (dist_prev={:.1f}, dist_next={:.1f})".format(pass_idx, hloc_keys[i], d_prev, d_next))
                     keys_to_remove.append(hloc_keys[i])
        
        if not keys_to_remove:
            break
            
        # Update hloc_keys
        hloc_keys = [k for k in hloc_keys if k not in keys_to_remove]
        print("  Pass {} removed {} points.".format(pass_idx, len(keys_to_remove)))

    valid_hloc_keys = hloc_keys

    ref_points = np.array([ref_poses[k] for k in ref_keys])
    hloc_points = np.array([hloc_poses[k] for k in hloc_keys])
    
    print("Loaded {} reference poses and {} valid hloc poses.".format(len(ref_points), len(hloc_points)))

    plt.figure(figsize=(10, 8))
    
    # Plot Reference
    if len(ref_points) > 0:
        plt.plot(ref_points[:, 0], ref_points[:, 1], 'g-', label='COLMAP Reference', alpha=0.5, linewidth=2)
        plt.plot(ref_points[0, 0], ref_points[0, 1], 'go', label='Ref Start')
        
    # Plot HLoc
    # User requested to visualize trajectory points AND connect them with lines
    plt.plot(hloc_points[:, 0], hloc_points[:, 1], 'b.-', label='HLoc Result', alpha=0.8, linewidth=1.0, markersize=4)
    plt.plot(hloc_points[0, 0], hloc_points[0, 1], 'bx', label='HLoc Start', markersize=10, markeredgewidth=2) 
    
    # Mark specific points to check synchronization if needed (optional)
    # plt.scatter(hloc_points[:, 0], hloc_points[:, 1], s=5, c='b')

    plt.title("2D Trajectory Comparison (XY Plane)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    print("Saving plot to {}".format(args.output))
    plt.savefig(args.output)
    # plt.show()

if __name__ == "__main__":
    main()
