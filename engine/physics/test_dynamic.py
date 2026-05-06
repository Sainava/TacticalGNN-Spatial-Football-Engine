import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from physics.dynamic_control import DynamicSpatialEngine

print("Loading data and calculating momentum...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
momentum_df = KinematicsEngine().add_kinematics(clean_df)

FRAME_TO_TEST = 1232
test_frame = momentum_df.iloc[FRAME_TO_TEST]

engine = DynamicSpatialEngine(projection_time=0.5)
h_polys, a_polys, h_coords, a_coords, h_vx, h_vy, a_vx, a_vy = engine.calculate_dynamic_voronoi(test_frame)

# --- ADDED: TERMINAL DIAGNOSTIC ---
print("\n--- SANITY CHECK: ACTIVE PLAYERS ---")
print(f"Home Players Found: {len(h_coords)}")
print(f"Away Players Found: {len(a_coords)}")

fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='green', linestyle='--', linewidth=2)

for poly in h_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', alpha=0.5, linewidth=2)
        ax.fill(x, y, color='red', alpha=0.1)

for poly in a_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', alpha=0.5, linewidth=2)
        ax.fill(x, y, color='blue', alpha=0.1)

# --- ADDED: COUNTS TO LEGEND ---
ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=8, label=f'Home ({len(h_coords)})')
ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=8, label=f'Away ({len(a_coords)})')
ax.plot(test_frame['Home_Ball_X'], test_frame['Home_Ball_Y'], 'ko', markersize=6, label='Ball')

ax.quiver(h_coords[:, 0], h_coords[:, 1], h_vx, h_vy, color='darkred', scale=30, width=0.005, headwidth=4)
ax.quiver(a_coords[:, 0], a_coords[:, 1], a_vx, a_vy, color='darkblue', scale=30, width=0.005, headwidth=4)

# --- ADDED: COUNTS TO TITLE ---
plt.title(f"Dynamic Pitch Control: Regions warped by 0.5s of Momentum | Home ({len(h_coords)}) v Away ({len(a_coords)})")
plt.legend(loc="upper right")
plt.xlim(-60, 60)
plt.ylim(-40, 40)
plt.show()