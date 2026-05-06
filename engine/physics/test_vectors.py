import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Import our parsers and engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from physics.spatial_control import SpatialEngine

print("Loading and processing data...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)

k_engine = KinematicsEngine()
momentum_df = k_engine.add_kinematics(clean_df)

# CHANGE: Let's test exactly 1 minute into the game (25 fps * 60 seconds = Frame 1500)
FRAME_TO_TEST = 1232
test_frame = momentum_df.iloc[FRAME_TO_TEST]

s_engine = SpatialEngine()
h_coords, a_coords = s_engine.extract_active_players(test_frame)

# CHANGE: Explicitly print the player counts to the terminal to verify our 11v11 bouncer
print("\n--- SANITY CHECK: ACTIVE PLAYERS ---")
print(f"Home Players Found: {len(h_coords)}")
print(f"Away Players Found: {len(a_coords)}")

# Prepare to extract the velocities matching those exact coordinates
h_vx, h_vy = [], []
a_vx, a_vy = [], []

for x, y in h_coords:
    for i in range(1, 15):
        col_x = f'Home_Player_{i}_X'
        if col_x in test_frame and pd.notna(test_frame[col_x]) and abs(test_frame[col_x] - x) < 0.01:
            h_vx.append(test_frame[f'Home_Player_{i}_VX'])
            h_vy.append(test_frame[f'Home_Player_{i}_VY'])
            break

for x, y in a_coords:
    for i in range(15, 30): 
        col_x = f'Away_Player_{i}_X'
        if col_x in test_frame and pd.notna(test_frame[col_x]) and abs(test_frame[col_x] - x) < 0.01:
            a_vx.append(test_frame[f'Away_Player_{i}_VX'])
            a_vy.append(test_frame[f'Away_Player_{i}_VY'])
            break

# Draw the Pitch and Vectors
fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='green', linestyle='--', linewidth=2)

# Draw players
ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=8, label=f'Home ({len(h_coords)})')
ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=8, label=f'Away ({len(a_coords)})')
ax.plot(test_frame['Home_Ball_X'], test_frame['Home_Ball_Y'], 'ko', markersize=6, label='Ball')

# Draw Momentum Vectors
if h_vx:
    ax.quiver(h_coords[:, 0], h_coords[:, 1], h_vx, h_vy, color='red', scale=30, width=0.005, headwidth=4)
if a_vx:
    ax.quiver(a_coords[:, 0], a_coords[:, 1], a_vx, a_vy, color='blue', scale=30, width=0.005, headwidth=4)

plt.title(f"Kinematics Vector Test: Momentum at Frame {FRAME_TO_TEST}")
plt.legend(loc="upper right")
plt.xlim(-60, 60)
plt.ylim(-40, 40)
plt.show()