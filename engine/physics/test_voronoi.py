import matplotlib.pyplot as plt
from spatial_control import SpatialEngine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser

print("Loading data for Enterprise Voronoi...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
engine = SpatialEngine()

# Grab frame 300
test_frame = clean_df.iloc[1232]
home_polys, away_polys, h_coords, a_coords = engine.calculate_bounded_voronoi(test_frame)

# Draw the pitch
fig, ax = plt.subplots(figsize=(10.5, 6.8))
ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='green', linestyle='--', linewidth=2, label='Pitch Boundary')

# Draw Home Regions
for poly in home_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', alpha=0.5, linewidth=2)
        ax.fill(x, y, color='red', alpha=0.1)

# Draw Away Regions
for poly in away_polys:
    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', alpha=0.5, linewidth=2)
        ax.fill(x, y, color='blue', alpha=0.1)

# Draw the actual players
ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=8, label=f'Home ({len(h_coords)})')
ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=8, label=f'Away ({len(a_coords)})')
ax.plot(test_frame['Home_Ball_X'], test_frame['Home_Ball_Y'], 'ko', markersize=6, label='Ball')

plt.title("Enterprise Voronoi: 11v11 Geometrically Bounded Pitch Control")
plt.legend(loc="upper right")
plt.xlim(-60, 60)
plt.ylim(-40, 40)
plt.show()