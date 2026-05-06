import numpy as np
import pandas as pd

class KinematicsEngine:
    """
    Enterprise Kinematics Engine for Optical Tracking Data.
    Applies calculus and rolling-window smoothing to model true player momentum.
    """
    
    def __init__(self, max_speed_mps: float = 12.0, smoothing_window: int = 7):
        # 12 m/s is roughly 43 km/h. Anything faster is rejected as optical noise.
        self.max_speed = max_speed_mps
        
        # 7 frames at 25fps = 0.28 seconds of spatial smoothing.
        self.window = smoothing_window

    def add_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates X-Velocity, Y-Velocity, and Absolute Speed for every player.
        Optimized to prevent pandas memory fragmentation.
        """
        dt = df['Time'].diff()
        dt.iloc[0] = 0.04 

        # A temporary holding list for all our new calculated columns
        new_dataframes = []

        for team in ['Home', 'Away']:
            x_cols = [c for c in df.columns if c.startswith(team) and c.endswith('_X')]
            y_cols = [c for c in df.columns if c.startswith(team) and c.endswith('_Y')]

            vx = df[x_cols].diff().divide(dt, axis=0)
            vy = df[y_cols].diff().divide(dt, axis=0)

            vx = vx.rolling(window=self.window, min_periods=1, center=True).mean()
            vy = vy.rolling(window=self.window, min_periods=1, center=True).mean()

            speed = np.sqrt(vx**2 + vy.values**2)

            speed_excess = speed > self.max_speed
            scale_factor = np.where(speed_excess, self.max_speed / speed.replace(0, 1), 1.0)
            
            vx = vx.multiply(scale_factor)
            vy = vy.multiply(scale_factor)
            speed = speed.mask(speed_excess, self.max_speed)

            # Rename the columns so they match our naming convention
            vx.columns = [c.replace('_X', '_VX') for c in x_cols]
            vy.columns = [c.replace('_Y', '_VY') for c in y_cols]
            speed.columns = [c.replace('_X', '_Speed') for c in x_cols]

            # Append the new math blocks to our holding list
            new_dataframes.extend([vx, vy, speed])

        # Smash the original dataframe and all 66 new columns together in ONE operation
        return pd.concat([df] + new_dataframes, axis=1)

if __name__ == "__main__":
    import sys
    import os
    
    # Import the parser to load data for testing
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from parsers.metrica_parser import MetricaParser

    print("Loading raw tracking data (This will take a moment)...")
    parser = MetricaParser()
    home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    
    clean_df = parser.merge_and_clean(home_file, away_file)
    
    print("\nApplying Kinematics Engine...")
    engine = KinematicsEngine()
    momentum_df = engine.add_kinematics(clean_df)
    
    # --- DIAGNOSTIC TEST ---
    # Let's inspect Home Player 11's movement at exactly 3.2 seconds into the match
    test_frames = momentum_df.iloc[80:85]
    
    print("\n--- SPRINT DIAGNOSTIC: Home Player 11 ---")
    for index, row in test_frames.iterrows():
        t = row['Time']
        x = row['Home_Player_11_X']
        vx = row['Home_Player_11_VX']
        vy = row['Home_Player_11_VY']
        s = row['Home_Player_11_Speed']
        
        print(f"Time: {t:>5.2f}s | Pos_X: {x:>6.2f} | Vel_X: {vx:>6.2f} m/s | Vel_Y: {vy:>6.2f} m/s | Total Speed: {s:>5.2f} m/s")