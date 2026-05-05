import pandas as pd
import numpy as np

class MetricaParser:
    """
    Data ingestion engine for Metrica Sports tracking data.
    Handles data merging, coordinate system transformation, and missing value interpolation.
    """
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def extract_tracking_data(self, file_path: str) -> pd.DataFrame:
        """
        Extracts tracking data by dynamically locating the header row 
        and enforcing alternating X/Y coordinate pairs.
        """
        # 1. Locate the actual header row dynamically
        raw_head = pd.read_csv(file_path, header=None, nrows=5)
        header_idx = None
        for i in range(5):
            row_vals = [str(x).lower() for x in raw_head.iloc[i].values]
            if 'frame' in row_vals or 'time [s]' in row_vals:
                header_idx = i
                break
                
        if header_idx is None:
            raise ValueError("Invalid CSV: Could not locate temporal headers.")
            
        entities = raw_head.iloc[header_idx].values
        
        # 2. Build deterministic column names
        columns = []
        current_entity = ""
        axis = ""
        
        for i, entity in enumerate(entities):
            str_ent = str(entity).strip().lower()
            
            # Handle Temporal Data
            if 'period' in str_ent:
                columns.append("Period")
            elif 'frame' in str_ent:
                columns.append("Frame")
            elif 'time' in str_ent:
                columns.append("Time")
            else:
                # Handle Spatial Data
                if str_ent != 'nan' and str_ent != '':
                    if 'ball' in str_ent:
                        current_entity = "Ball"
                    else:
                        # Strip all text, keep only the jersey number integer
                        current_entity = "".join(filter(str.isdigit, str_ent)) 
                    
                    axis = "X"
                    col_prefix = "Ball" if current_entity == "Ball" else f"Player_{current_entity}"
                    columns.append(f"{col_prefix}_{axis}")
                    
                else:
                    # Blank cell logic: The first blank is Y, all subsequent blanks are garbage
                    if axis == "X":
                        axis = "Y"
                        col_prefix = "Ball" if current_entity == "Ball" else f"Player_{current_entity}"
                        columns.append(f"{col_prefix}_{axis}")
                    else:
                        columns.append(f"Drop_{i}")

        # 3. Read data using the enforced columns and purge garbage
        df = pd.read_csv(file_path, header=None, skiprows=header_idx + 1, names=columns)
        df.drop(columns=[c for c in df.columns if c.startswith("Drop_")], inplace=True)
        
        return df

    def transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms normalized [0,1] coordinates to pitch dimensions in meters.
        Shifts the origin (0,0) to the center of the pitch.
        """
        # Find all X and Y columns safely
        x_cols = [c for c in df.columns if c.endswith('_X')]
        y_cols = [c for c in df.columns if c.endswith('_Y')]

        # Metrica's origin is top-left [0,0]. Standard ML origin is center pitch.
        for x in x_cols:
            df[x] = (df[x] - 0.5) * self.pitch_length
            
        for y in y_cols:
            df[y] = -1 * (df[y] - 0.5) * self.pitch_width 

        return df

    def merge_and_clean(self, home_path: str, away_path: str) -> pd.DataFrame:
        """
        Merges home and away datasets and interpolates missing spatial data.
        """
        home_df = self.extract_tracking_data(home_path)
        away_df = self.extract_tracking_data(away_path)

        # Prefix columns to distinguish teams before merging
        home_df = home_df.add_prefix('Home_')
        away_df = away_df.add_prefix('Away_')

        # Rename core temporal columns back to standard so we can merge on them
        # FIXED: We now correctly target 'Home_Time' and 'Away_Time' without the '[s]'
        home_df.rename(columns={'Home_Frame': 'Frame', 'Home_Period': 'Period', 'Home_Time': 'Time'}, inplace=True)
        away_df.rename(columns={'Away_Frame': 'Frame', 'Away_Period': 'Period', 'Away_Time': 'Time'}, inplace=True)

        # Both files contain ball tracking data; drop it from the away dataframe to prevent duplicates
        away_df.drop(columns=[c for c in away_df.columns if 'Ball' in c], inplace=True)

        # Merge the two dataframes based on the exact frame and time
        merged_df = pd.merge(home_df, away_df, on=['Frame', 'Period', 'Time'])

        # Apply geometric scaling
        merged_df = self.transform_coordinates(merged_df)

        # Interpolate NaNs (e.g., when a player temporarily steps off camera/pitch)
        merged_df.interpolate(method='linear', limit_direction='both', inplace=True)

        return merged_df

if __name__ == "__main__":
    parser = MetricaParser()
    
    # Update paths based on execution directory
    home_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_file = "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
    
    try:
        clean_data = parser.merge_and_clean(home_file, away_file)
        print(f"Data ingestion successful. Shape: {clean_data.shape}")
        print("\nSample of first 5 frames (Time, Ball X, Ball Y):")
        print(clean_data[['Time', 'Home_Ball_X', 'Home_Ball_Y']].head())
    except Exception as e:
        print(f"Data ingestion failed: {e}")