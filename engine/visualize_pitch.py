import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from parsers.metrica_parser import MetricaParser

def draw_pitch():
    """Draws a standard 105x68m football pitch."""
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor('#2e8b57') # SeaGreen pitch
    ax.set_xlim(-55, 55)
    ax.set_ylim(-37, 37)

    # Pitch Outline & Center Line
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color='white', zorder=1)
    ax.plot([0, 0], [-34, 34], color='white', zorder=1)

    # Center Circle
    center_circle = plt.Circle((0, 0), 9.15, color='white', fill=False, zorder=1)
    ax.add_patch(center_circle)

    # Penalty Boxes
    ax.plot([-52.5, -36], [-20.16, -20.16], color='white', zorder=1)
    ax.plot([-52.5, -36], [20.16, 20.16], color='white', zorder=1)
    ax.plot([-36, -36], [-20.16, 20.16], color='white', zorder=1)

    ax.plot([52.5, 36], [-20.16, -20.16], color='white', zorder=1)
    ax.plot([52.5, 36], [20.16, 20.16], color='white', zorder=1)
    ax.plot([36, 36], [-20.16, 20.16], color='white', zorder=1)

    return fig, ax

def animate_match(df: pd.DataFrame, frames: int = 250):
    """Animates the player and ball movements for a set number of frames."""
    fig, ax = draw_pitch()
    plt.title("Metrica Tracking Data: First 10 Seconds", color='black')

    # Identify columns dynamically
    home_x = [c for c in df.columns if c.startswith('Home_Player') and c.endswith('_X')]
    home_y = [c for c in df.columns if c.startswith('Home_Player') and c.endswith('_Y')]
    away_x = [c for c in df.columns if c.startswith('Away_Player') and c.endswith('_X')]
    away_y = [c for c in df.columns if c.startswith('Away_Player') and c.endswith('_Y')]

    # Setup scatter plots (zorder ensures players are drawn on top of the pitch lines)
    home_scatter = ax.scatter([], [], c='red', edgecolors='white', s=80, label='Home', zorder=2)
    away_scatter = ax.scatter([], [], c='blue', edgecolors='white', s=80, label='Away', zorder=2)
    ball_scatter = ax.scatter([], [], c='black', s=40, zorder=3)
    
    # Add a clock
    time_text = ax.text(-50, 35, '', color='white', fontsize=12, fontweight='bold', zorder=4)

    def update(frame):
        row = df.iloc[frame]
        
        # Update Home Team
        hx, hy = row[home_x].values, row[home_y].values
        home_coords = np.column_stack((hx, hy))
        home_scatter.set_offsets(home_coords[~np.isnan(home_coords).any(axis=1)]) # Filter NaNs
        
        # Update Away Team
        ax_, ay_ = row[away_x].values, row[away_y].values
        away_coords = np.column_stack((ax_, ay_))
        away_scatter.set_offsets(away_coords[~np.isnan(away_coords).any(axis=1)])
        
        # Update Ball
        ball_scatter.set_offsets(np.column_stack(([row['Home_Ball_X']], [row['Home_Ball_Y']])))
        
        # Update Clock
        time_text.set_text(f"Time: {row['Time']:.2f}s")
        
        return home_scatter, away_scatter, ball_scatter, time_text

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    plt.show()

if __name__ == "__main__":
    print("Loading parser and ingesting data...")
    parser = MetricaParser()
    
    home_file = "../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_file = "../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    
    clean_df = parser.merge_and_clean(home_file, away_file)
    print("Data ingested successfully! Launching visualization...")
    
    # Animate the first 250 frames (10 seconds at 25fps)
    animate_match(clean_df, frames=250)