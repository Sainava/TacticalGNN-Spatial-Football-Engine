import matplotlib.pyplot as plt
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from physics.dynamic_control import DynamicSpatialEngine
from analytics.passing_lanes import PassingViabilityEngine

print("Loading Data & Physics Engines...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
momentum_df = KinematicsEngine().add_kinematics(clean_df)

# Test our Fast Break frame again
FRAME = 1238
test_frame = momentum_df.iloc[FRAME]

print("Calculating Dynamic Pitch Control...")
spatial_engine = DynamicSpatialEngine(projection_time=0.5)
h_polys, a_polys, h_coords, a_coords, h_vx, h_vy, a_vx, a_vy = spatial_engine.calculate_dynamic_voronoi(test_frame)

print("Evaluating Passing Lanes...")
passing_engine = PassingViabilityEngine()

# 1. Find the ball and who has it
ball_coord = np.array([test_frame['Home_Ball_X'], test_frame['Home_Ball_Y']])
possession_team, carrier_coord = passing_engine.find_ball_carrier(ball_coord, h_coords, a_coords)

# 2. Set up the geometry based on possession
if possession_team == 'Home':
    teammates = h_coords
    opp_polys = a_polys
else:
    teammates = a_coords
    opp_polys = h_polys

# 3. Run the Analytics Engine
open_passes, blocked_passes = passing_engine.evaluate_passing_lanes(carrier_coord, teammates, opp_polys)

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='green', linestyle='--', linewidth=2)

# Draw Voronoi Polygons (Faded to 10% opacity so they don't distract)
for poly in h_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', alpha=0.1, linewidth=1)
        ax.fill(x, y, color='red', alpha=0.05)
for poly in a_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', alpha=0.1, linewidth=1)
        ax.fill(x, y, color='blue', alpha=0.05)

# Draw Players
ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=6, alpha=0.6)
ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=6, alpha=0.6)

# Highlight Ball Carrier
ax.plot(carrier_coord[0], carrier_coord[1], marker='o', color='yellow', markersize=10, markeredgecolor='black', markeredgewidth=2)
ax.plot(ball_coord[0], ball_coord[1], 'ko', markersize=4)

# Draw Passing Lanes
for tm in open_passes:
    ax.plot([carrier_coord[0], tm[0]], [carrier_coord[1], tm[1]], color='lime', linewidth=2.5, zorder=5)

for tm in blocked_passes:
    ax.plot([carrier_coord[0], tm[0]], [carrier_coord[1], tm[1]], color='darkred', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)

# Legend entries
ax.plot([], [], color='lime', linewidth=2.5, label=f'Open Passes ({len(open_passes)})')
ax.plot([], [], color='darkred', linestyle='--', linewidth=1.5, label=f'Blocked Passes ({len(blocked_passes)})')

plt.title(f"Tactical Vision: Passing Viability | Frame {FRAME} | Possession: {possession_team}")
plt.legend(loc="upper right")
plt.xlim(-60, 60)
plt.ylim(-40, 40)
plt.show()