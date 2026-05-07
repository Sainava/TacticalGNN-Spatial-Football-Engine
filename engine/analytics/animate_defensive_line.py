import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from analytics.defensive_line import DefensiveLineEngine

print("Initializing Enterprise Tactical Tracker...")
parser = MetricaParser()
home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"

clean_df = parser.merge_and_clean(home_file, away_file)
momentum_df = KinematicsEngine().add_kinematics(clean_df)
line_engine = DefensiveLineEngine()

START_FRAME, END_FRAME = 1200, 1350

# --- STATIC COMPASS ANCHOR ---
# Teams do not switch sides mid-clip. We calculate direction once based on the starting frame
# to prevent the engine from breaking if the Goalkeeper goes off-camera.
frame_0 = momentum_df.iloc[START_FRAME]
h_cols = [c for c in frame_0.index if c.startswith('Home_') and c.endswith('_X') and not np.isnan(frame_0[c])]
a_cols = [c for c in frame_0.index if c.startswith('Away_') and c.endswith('_X') and not np.isnan(frame_0[c])]
h_mean_x = np.mean([frame_0[x] for x in h_cols])
a_mean_x = np.mean([frame_0[x] for x in a_cols])

# Whichever team has the lower average X in frame 1 is defending the Left goal.
if h_mean_x < a_mean_x:
    HOME_DEFENDS = 'left'
    AWAY_DEFENDS = 'right'
else:
    HOME_DEFENDS = 'right'
    AWAY_DEFENDS = 'left'

# --- VISUALIZATION SETUP ---
fig, ax = plt.subplots(figsize=(10.5, 6.8))

def draw_pitch(ax):
    """Draws professional football pitch markings on a green surface."""
    ax.set_facecolor('#2E8B57')
    ax.plot([-52.5, -52.5, 52.5, 52.5, -52.5], [-34, 34, 34, -34, -34], color='white', linewidth=1.5)
    ax.plot([0, 0], [-34, 34], color='white', linewidth=1.5)
    ax.add_patch(patches.Circle((0, 0), 9.15, color='white', fill=False, linewidth=1.5))
    ax.plot([-52.5, -36, -36, -52.5], [20.16, 20.16, -20.16, -20.16], color='white', linewidth=1.5)
    ax.plot([52.5, 36, 36, 52.5], [20.16, 20.16, -20.16, -20.16], color='white', linewidth=1.5)

def update(frame_idx):
    ax.clear()
    frame = momentum_df.iloc[frame_idx]
    draw_pitch(ax)
    
    # 1. DYNAMIC EXTRACTION: Safely grab players on camera
    h_x_cols = [c for c in frame.index if c.startswith('Home_') and c.endswith('_X') and not np.isnan(frame[c])]
    a_x_cols = [c for c in frame.index if c.startswith('Away_') and c.endswith('_X') and not np.isnan(frame[c])]
    
    raw_h = np.array([[frame[x], frame[x.replace('_X', '_Y')]] for x in h_x_cols])
    raw_a = np.array([[frame[x], frame[x.replace('_X', '_Y')]] for x in a_x_cols])
    ball_coord = np.array([frame['Home_Ball_X'], frame['Home_Ball_Y']])

# --- THE INLINE BOUNCER: Purge the Bench Players ---
    # Shrink the bottom boundary to Y = -31.5 to exclude players warming up on the touchline
    h_coords = np.array([p for p in raw_h if -52.5 <= p[0] <= 52.5 and -31.5 <= p[1] <= 34])
    a_coords = np.array([p for p in raw_a if -52.5 <= p[0] <= 52.5 and -31.5 <= p[1] <= 34])

    # 2. POSSESSION TOGGLE: Who is closest to the ball?
    # (These are the lines that accidentally went missing!)
    h_dists = np.linalg.norm(h_coords - ball_coord, axis=1)
    a_dists = np.linalg.norm(a_coords - ball_coord, axis=1)
    
    if np.min(h_dists) < np.min(a_dists):
        # Home has the ball -> Away is defending
        att_coords, def_coords = h_coords, a_coords
        att_color, def_color = 'ro', 'bo'
        possession_status = "Home (Red)"
        defending_side = AWAY_DEFENDS
        attacking_side = AWAY_DEFENDS # Home attacks the goal Away is defending
    else:
        # Away has the ball -> Home is defending
        att_coords, def_coords = a_coords, h_coords
        att_color, def_color = 'bo', 'ro'
        possession_status = "Away (Blue)"
        defending_side = HOME_DEFENDS
        attacking_side = HOME_DEFENDS # Away attacks the goal Home is defending

    # 3. TACTICAL CALCULATIONS
    back_four = line_engine.get_back_four_chain(def_coords, defending_side)
    attacker_coord = line_engine.get_target_attacker(att_coords, attacking_side)
    nearest_defender, gap_distance = line_engine.get_nearest_marker(attacker_coord, back_four)

    # --- RENDER GRAPHICS ---
    # Draw Defensive Chain
    ax.plot(back_four[:, 0], back_four[:, 1], color='white', linestyle='--', linewidth=2.5, alpha=0.9, zorder=3)

    # Draw Threat Radius
    ax.add_patch(patches.Circle((attacker_coord[0], attacker_coord[1]), gap_distance, color='yellow', alpha=0.2, zorder=2))
    
    # Draw Isolation Line
    ax.plot([attacker_coord[0], nearest_defender[0]], [attacker_coord[1], nearest_defender[1]], color='yellow', linewidth=2, zorder=4)

    # Draw Standard Players (Slightly transparent)
    ax.plot(h_coords[:, 0], h_coords[:, 1], 'ro', markersize=6, markeredgecolor='white', alpha=0.4)
    ax.plot(a_coords[:, 0], a_coords[:, 1], 'bo', markersize=6, markeredgecolor='white', alpha=0.4)
    ax.plot(ball_coord[0], ball_coord[1], 'ko', markersize=4, markeredgecolor='white')

    # Draw Key Tactical Actors (Bright and large)
    ax.plot(back_four[:, 0], back_four[:, 1], def_color, markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=6) # The Defense
    ax.plot(attacker_coord[0], attacker_coord[1], att_color, markersize=10, markeredgecolor='yellow', markeredgewidth=2, zorder=6) # The Attacker
    ax.plot(nearest_defender[0], nearest_defender[1], def_color, markersize=12, markeredgecolor='yellow', markeredgewidth=2, zorder=7) # The Marker

    # Telemetry Widget
    status = "ISOLATED" if gap_distance > 8 else "MARKED"
    telemetry = f"ATTACK: {possession_status}\nTarget Gap: {gap_distance:.1f}m ({status})"
    ax.text(0.02, 0.95, telemetry, transform=ax.transAxes, fontsize=11, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    # Zoom the camera closer to the actual pitch boundaries
    ax.set_xlim(-55, 55)
    ax.set_ylim(-35, 35)
    ax.set_title(f"Dynamic Shape & Isolation Analysis | Frame: {frame_idx}")

print("Rendering Engine Started...")
ani = animation.FuncAnimation(fig, update, frames=range(START_FRAME, END_FRAME), interval=40)
plt.show()