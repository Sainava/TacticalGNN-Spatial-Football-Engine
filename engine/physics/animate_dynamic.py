import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import pandas as pd

# Import our enterprise architecture
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from physics.dynamic_control import DynamicSpatialEngine

print("Loading data and calculating momentum (This takes a moment)...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
momentum_df = KinematicsEngine().add_kinematics(clean_df)

# Initialize the dynamic engine (Projecting 0.5 seconds into the future)
engine = DynamicSpatialEngine(projection_time=0.5)

# --- THE DIRECTOR'S CUT ---
# A 10-second clip at 25 fps = 250 frames. 
# Let's animate the Fast Break we analyzed earlier (Frames 1200 to 1450)
START_FRAME = 1200
END_FRAME = 1450

fig, ax = plt.subplots(figsize=(10.5, 6.8))

def update(frame_idx):
    """
    This function runs 25 times per second. It wipes the canvas and draws the next frame.
    """
    ax.clear() # Wipe the previous frame
    test_frame = momentum_df.iloc[frame_idx]
    
    # Run the physics engine
    h_polys, a_polys, h_coords, a_coords, h_vx, h_vy, a_vx, a_vy = engine.calculate_dynamic_voronoi(test_frame)
    
    # DIAGNOSTIC: Print player count to the terminal as a live progress bar
    print(f"Rendering Frame {frame_idx}/{END_FRAME} | Home: {len(h_coords)}, Away: {len(a_coords)}", end='\r')

    # Draw the Pitch
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='green', linestyle='--', linewidth=2)

    # Draw Home Polygons
    for poly in h_polys:
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', alpha=0.5, linewidth=1.5)
            ax.fill(x, y, color='red', alpha=0.1)

    # Draw Away Polygons
    for poly in a_polys:
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', alpha=0.5, linewidth=1.5)
            ax.fill(x, y, color='blue', alpha=0.1)

    # Draw Players & Ball
    ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=8)
    ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=8)
    ax.plot(test_frame['Home_Ball_X'], test_frame['Home_Ball_Y'], 'ko', markersize=6)

    # Draw Momentum Arrows
    if h_vx:
        ax.quiver(h_coords[:, 0], h_coords[:, 1], h_vx, h_vy, color='darkred', scale=30, width=0.005, headwidth=4)
    if a_vx:
        ax.quiver(a_coords[:, 0], a_coords[:, 1], a_vx, a_vy, color='darkblue', scale=30, width=0.005, headwidth=4)

    # UI: Lock the camera and update the title with live player counts
    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_title(f"Dynamic Pitch Control | Frame: {frame_idx} | Active: Home ({len(h_coords)}) v Away ({len(a_coords)})")

print("\nStarting Animation Engine...")

# interval=40 means 40 milliseconds between frames (which equals 25 frames per second)
ani = animation.FuncAnimation(fig, update, frames=range(START_FRAME, END_FRAME), interval=40, repeat=True)

plt.show()