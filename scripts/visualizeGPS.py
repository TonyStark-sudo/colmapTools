import matplotlib.pyplot as plt
import os

def visualize_gps_track(gps_file_path, output_image_path):
    """
    Reads a GPS data file and plots the trajectory.

    Args:
        gps_file_path (str): The path to the GPS data file (e.g., 'gps.txt').
        output_image_path (str): The path to save the output plot image.
    """
    lats = []
    lons = []

    # Check if the GPS file exists
    if not os.path.exists(gps_file_path):
        print(f"Error: GPS file not found at {gps_file_path}")
        return

    # Read the GPS data from the file
    with open(gps_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    lats.append(float(parts[1]))
                    lons.append(float(parts[2]))
                except ValueError:
                    print(f"Warning: Could not parse line: {line.strip()}")

    # Check if any data was loaded
    if not lats or not lons:
        print("No valid GPS data found to plot.")
        return

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Create a color array based on the index of the points
    colors = range(len(lats))

    # Plot the connecting line in a subtle color
    plt.plot(lons, lats, color='lightgray', linestyle='-', zorder=1)

    # Plot the points with a colormap
    scatter = plt.scatter(lons, lats, c=colors, cmap='viridis', marker='o', zorder=2, label='GPS Points')
    
    # Add a colorbar to show the sequence
    cbar = plt.colorbar(scatter)
    cbar.set_label('Point Order')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Trajectory Heatmap')
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # Ensure aspect ratio is equal

    # Save the plot to a file
    plt.savefig(output_image_path)
    print(f"GPS trajectory plot saved to {output_image_path}")

if __name__ == '__main__':
    # Assuming the script is in the 'scripts' directory, the gps.txt is in the parent's 'output' directory
    gps_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'gps.txt')
    output_image = os.path.join(os.path.dirname(__file__), '..', 'output', 'gps_trajectory.png')
    
    visualize_gps_track(gps_file, output_image)
