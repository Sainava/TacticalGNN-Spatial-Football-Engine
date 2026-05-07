import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import numpy as np
import matplotlib.patches as patches

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from physics.dynamic_control import DynamicSpatialEngine
from analytics.passing_lanes import PassingViabilityEngine

print("Loading Data & Physics Engines (This takes a moment)...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
momentum_df = KinematicsEngine().add_kinematics(clean_df)

# Initialize engines
spatial_engine = DynamicSpatialEngine(projection_time=0.5)
passing_engine = PassingViabilityEngine()

# Let's animate a 3-second sequence around the Fast Break (Frames 1210 to 1285)
START_FRAME = 1210
END_FRAME = 1450#1285

fig, ax = plt.subplots(figsize=(10.5, 6.8))

def draw_pitch(ax):
    """Draws professional football pitch markings on a green surface."""
    # Set the grass color
    ax.set_facecolor('#2E8B57') # SeaGreen

    # Pitch Outline & Halfway Line (White)
    ax.plot([-52.5, -52.5, 52.5, 52.5, -52.5], [-34, 34, 34, -34, -34], color='white', linewidth=1.5)
    ax.plot([0, 0], [-34, 34], color='white', linewidth=1.5)

    # Center Circle and Center Spot
    centre_circle = patches.Circle((0, 0), 9.15, color='white', fill=False, linewidth=1.5)
    ax.add_patch(centre_circle)
    ax.plot(0, 0, 'wo', markersize=4) # White dot in the middle

    # Penalty Areas
    ax.plot([-52.5, -36, -36, -52.5], [20.16, 20.16, -20.16, -20.16], color='white', linewidth=1.5)
    ax.plot([52.5, 36, 36, 52.5], [20.16, 20.16, -20.16, -20.16], color='white', linewidth=1.5)

    # Six-yard Boxes
    ax.plot([-52.5, -47, -47, -52.5], [9.16, 9.16, -9.16, -9.16], color='white', linewidth=1.5)
    ax.plot([52.5, 47, 47, 52.5], [9.16, 9.16, -9.16, -9.16], color='white', linewidth=1.5)

def update(frame_idx):
    ax.clear()
    test_frame = momentum_df.iloc[frame_idx]
    
    # 1. Calculate the spatial physics
    h_polys, a_polys, h_coords, a_coords, h_vx, h_vy, a_vx, a_vy = spatial_engine.calculate_dynamic_voronoi(test_frame)
    
    # 2. Find the ball and evaluate vision
    ball_coord = np.array([test_frame['Home_Ball_X'], test_frame['Home_Ball_Y']])
    possession_team, carrier_coord = passing_engine.find_ball_carrier(ball_coord, h_coords, a_coords)
    
    if possession_team == 'Home':
        teammates, opp_polys = h_coords, a_polys
        carrier_color = 'red'
    else:
        teammates, opp_polys = a_coords, h_polys
        carrier_color = 'blue'
        
    open_passes, _ = passing_engine.evaluate_passing_lanes(carrier_coord, teammates, opp_polys)
    
    print(f"Rendering Frame {frame_idx}/{END_FRAME} | Possession: {possession_team} | Open Options: {len(open_passes)}", end='\r')

    # --- VISUALS ---
    # Pitch
    draw_pitch(ax)
    
    # Polygons (Extremely faint so they don't distract from the passing lines)
    for poly in h_polys:
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', alpha=0.05, linewidth=1)
            ax.fill(x, y, color='red', alpha=0.02)
    for poly in a_polys:
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='blue', alpha=0.05, linewidth=1)
            ax.fill(x, y, color='blue', alpha=0.02)

   # Players (with white halos)
    ax.plot(h_coords[:, 0], h_coords[:, 1], marker='o', color='red', markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=6, linestyle='None')
    ax.plot(a_coords[:, 0], a_coords[:, 1], marker='o', color='blue', markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=6, linestyle='None')
    
    # Ball (Black dot with white halo)
    ax.plot(ball_coord[0], ball_coord[1], marker='o', color='black', markersize=5, markeredgecolor='white', markeredgewidth=1, zorder=7, linestyle='None')

    # Highlight Ball Carrier
    ax.plot(carrier_coord[0], carrier_coord[1], marker='o', color='yellow', markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=5)

    # Glowing Passing Lanes (Only drawing the OPEN ones, as you suggested)
    for tm in open_passes:
        ax.plot([carrier_coord[0], tm[0]], [carrier_coord[1], tm[1]], color='lime', linewidth=3, zorder=4)

    # UI setup
    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_title(f"Broadcast Vision Loop | Frame {frame_idx} | Open Options: {len(open_passes)}")

print("\nStarting Animation Engine...")
ani = animation.FuncAnimation(fig, update, frames=range(START_FRAME, END_FRAME), interval=40, repeat=True)
plt.show()