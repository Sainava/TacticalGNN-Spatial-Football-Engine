import numpy as np
import pandas as pd
import sys
import os
import pickle
from tqdm import tqdm # For a progress bar

# Ensure we can import our previous modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine

class CCSNBuilder:
    """
    Carrier-Centric Spatial Network (CCSN) Builder.
    Converts raw spatiotemporal frames into Ego-Graphs centered on the ball carrier,
    using Kinematic Time-To-Intercept (TTI) to define graph edges.
    """
    def __init__(self, max_tti_threshold=3.5, max_distance_fallback=15.0):
        self.max_tti = max_tti_threshold
        self.max_dist = max_distance_fallback

    def extract_nodes_from_frame(self, frame):
        """Extracts all visible players (ignoring the ball and bench) into a clean dictionary."""
        players = []
        
        for team in ['Home', 'Away']:
            cols = [c for c in frame.index if c.startswith(f'{team}_') and c.endswith('_X') and not np.isnan(frame[c])]
            for x_col in cols:
                if 'Ball' in x_col:
                    continue
                    
                pid = x_col.replace('_X', '')
                px, py = frame[f'{pid}_X'], frame[f'{pid}_Y']
                
                # THE INLINE BOUNCER: Ignore bench players outside the touchlines
                if not (-52.5 <= px <= 52.5 and -31.5 <= py <= 34):
                    continue
                
                vx = frame[f'{pid}_VX'] if f'{pid}_VX' in frame else 0.0
                vy = frame[f'{pid}_VY'] if f'{pid}_VY' in frame else 0.0
                
                players.append({
                    'id': pid,
                    'team': team,
                    'pos': np.array([px, py]),
                    'vel': np.array([vx, vy])
                })
                
        ball_pos = np.array([frame['Home_Ball_X'], frame['Home_Ball_Y']])
        return players, ball_pos

    def identify_carrier(self, players, ball_pos):
        """Finds the specific node (player) closest to the ball."""
        min_dist = float('inf')
        carrier = None
        
        for p in players:
            dist = np.linalg.norm(p['pos'] - ball_pos)
            if dist < min_dist:
                min_dist = dist
                carrier = p
                
        return carrier

    def build_ego_graph(self, players, carrier, play_direction="Right"):
        """
        Filters the global pitch down to the Carrier-Centric Graph using TTI physics.
        Normalizes all coordinates so the carrier is always attacking Left-to-Right (+X).
        """
        nodes = []
        edges = []
        
        # ==========================================
        # SPATIAL NORMALIZATION
        # If attacking Left (-X), multiply by -1 so the GNN always sees an attack towards +X
        # ==========================================
        flip_multiplier = -1 if play_direction == "Left" else 1
        
        norm_carrier_pos = carrier['pos'] * flip_multiplier
        norm_carrier_vel = carrier['vel'] * flip_multiplier
        
        carrier_node = {
            **carrier, 
            'pos': norm_carrier_pos, 
            'vel': norm_carrier_vel, 
            'is_teammate': 1
        }
        nodes.append(carrier_node)
        
        for p in players:
            if p['id'] == carrier['id']:
                continue
                
            is_teammate = 1 if p['team'] == carrier['team'] else 0
            
            # Normalize surrounding players
            norm_p_pos = p['pos'] * flip_multiplier
            norm_p_vel = p['vel'] * flip_multiplier
                
            vec_to_carrier = norm_carrier_pos - norm_p_pos
            distance = np.linalg.norm(vec_to_carrier)
            
            # SAFEGUARD: Prevent division by zero
            if distance == 0:
                distance = 1e-9 
            
            unit_vec_to_carrier = vec_to_carrier / distance
            closing_speed = np.dot(norm_p_vel, unit_vec_to_carrier)
            
            if closing_speed > 0:
                tti = distance / closing_speed
            else:
                tti = float('inf') 
                
            if tti <= self.max_tti or distance <= self.max_dist:
                nodes.append({
                    **p, 
                    'pos': norm_p_pos, 
                    'vel': norm_p_vel, 
                    'is_teammate': is_teammate
                })
                edges.append({
                    'source': p['id'],
                    'target': carrier['id'],
                    'distance': distance,
                    'tti': tti,
                    'closing_speed': closing_speed,
                    'is_teammate': is_teammate 
                })
                
        return nodes, edges

    def get_play_direction(self, frame, carrier_team):
        """
        Dynamically infers which way the team is attacking.
        Metrica Rule of Thumb: Home attacks Right in Period 1, Left in Period 2.
        """
        try:
            period = frame['Period']
            if carrier_team == 'Home':
                return "Right" if period == 1 else "Left"
            else:
                return "Left" if period == 1 else "Right"
        except KeyError:
            # Fallback if 'Period' column is missing: default to Right
            return "Right"

    def process_match(self, momentum_df, output_filename, start_frame=None, end_frame=None):
        """
        Loops through tracking data, generates normalized Ego-Graphs, 
        and saves the entire sequence.
        """
        dataset = []
        
        start = start_frame if start_frame else 0
        end = end_frame if end_frame else len(momentum_df)
        
        print(f"Generating CCSN Dataset from frame {start} to {end}...")
        
        for frame_idx in tqdm(range(start, end)):
            frame = momentum_df.iloc[frame_idx]
            
            # Skip dead-ball frames
            if np.isnan(frame['Home_Ball_X']):
                continue
                
            players, ball = self.extract_nodes_from_frame(frame)
            if not players:
                continue
                
            carrier = self.identify_carrier(players, ball)
            
            # Skip loose balls / passes in mid-air
            if np.linalg.norm(carrier['pos'] - ball) > 2.0:
                continue
                
            # Dynamically determine attacking direction
            direction = self.get_play_direction(frame, carrier['team'])
                
            # Build normalized graph
            nodes, edges = self.build_ego_graph(players, carrier, play_direction=direction)
            
            # ==========================================
            # LABEL GENERATION: THE 3-SECOND SURVIVAL TEST
            # Look 75 frames into the future. Did the carrier's team keep the ball?
            # ==========================================
            y_label = 0 # Default to Turnover (0)
            future_idx = frame_idx + 75
            
            if future_idx < len(momentum_df):
                fut_frame = momentum_df.iloc[future_idx]
                if not np.isnan(fut_frame['Home_Ball_X']):
                    fut_players, fut_ball = self.extract_nodes_from_frame(fut_frame)
                    if fut_players:
                        fut_carrier = self.identify_carrier(fut_players, fut_ball)
                        
                        # If the team who has the ball in the future is the same team 
                        # who has the ball right now, it's a successful possession retention!
                        if fut_carrier['team'] == carrier['team']:
                            y_label = 1
                            
            # Add the graph to the dataset with the new 'y' label
            dataset.append({
                'frame': frame_idx,
                'carrier_id': carrier['id'],
                'carrier_team': carrier['team'],
                'play_direction': direction,
                'nodes': nodes,
                'edges': edges,
                'label': y_label  # <--- Our Answer Key!
            })
            
            
        print(f"\nSaving {len(dataset)} graphs to {output_filename}...")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'wb') as f:
            pickle.dump(dataset, f)

# ==========================================
# MULTI-MATCH DATASET GENERATION PIPELINE
# ==========================================
if __name__ == "__main__":
    builder = CCSNBuilder()
    parser = MetricaParser()
    kinematics = KinematicsEngine()
    
    # Define all the games in your dataset
    games = [
        {
            "home": "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
            "away": "../../data/raw/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"
        },
        {
            "home": "../../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
            "away": "../../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"
        }
    ]
    
    master_dataset = []
    
    # THE FIX: Define the explicit output directory up front
    output_dir = "../../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, game in enumerate(games):
        print(f"\n=============================================")
        print(f" PROCESSING GAME {idx + 1}/{len(games)}")
        print(f"=============================================")
        
        print("1. Parsing CSVs...")
        clean_df = parser.merge_and_clean(game["home"], game["away"])
        
        print("2. Calculating Kinematics...")
        momentum_df = kinematics.add_kinematics(clean_df)
        
        print(f"3. Building Graphs for Game {idx + 1}...")
        
        # THE FIX: Route the temp file directly into the processed data folder
        temp_file = f"{output_dir}/temp_game_{idx}.pkl"
        
        builder.process_match(momentum_df, temp_file)
        
        # Merge into master dataset
        with open(temp_file, 'rb') as f:
            game_graphs = pickle.load(f)
            master_dataset.extend(game_graphs)
            
        # Clean up temp file to save disk space
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    # Save the final, unified dataset
    final_output = f"{output_dir}/ccsn_master_dataset.pkl"
    print(f"\nSaving {len(master_dataset)} TOTAL graphs to {final_output}...")
    
    with open(final_output, 'wb') as f:
        pickle.dump(master_dataset, f)
        
    print("Multi-Match Pipeline Complete! The dataset is ready for PyTorch.")