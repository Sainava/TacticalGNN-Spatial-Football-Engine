import sys
import os
import pandas as pd
import numpy as np

# Import parser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser

parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
frame_data = clean_df.iloc[300]

print("--- DIAGNOSTIC: AWAY TEAM FRAME 300 ---")
away_x_cols = [c for c in frame_data.index if c.startswith('Away_Player') and c.endswith('_X')]
away_y_cols = [c for c in frame_data.index if c.startswith('Away_Player') and c.endswith('_Y')]

away_coords = np.column_stack((frame_data[away_x_cols].values, frame_data[away_y_cols].values))

# Print every player's location
for i, coord in enumerate(away_coords):
    if not np.isnan(coord[0]):
        x, y = coord[0], coord[1]
        # Check if they fell into our Bermuda Triangle
        in_trap = (abs(x) < 20.0) and (y < -32.0)
        status = "DELETED BY FILTER" if in_trap else "SAFE"
        print(f"Away Player Index {i}: X={x:>7.2f}, Y={y:>7.2f} -> {status}")