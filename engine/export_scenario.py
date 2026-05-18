import torch
import numpy as np
import pandas as pd
import json
import sys
import os

# Import your pipeline components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parsers.metrica_parser import MetricaParser
from physics.kinematics import KinematicsEngine
from models.gnn_architecture import CCSNet
from models.ccsn_builder import CCSNBuilder

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy/torch types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NumpyEncoder, self).default(obj)

class ScenarioExporter:
    def __init__(self, model_path, device_type='cpu'):
        self.device = torch.device(device_type)
        print(f"Data Exporter starting on: {self.device}")
        
        self.model = CCSNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.builder = CCSNBuilder()

    def get_survival_probability(self, players, carrier, play_direction):
        """Runs the GATv2 model (Same as visualizer)."""
        flip = -1 if play_direction == "Left" else 1
        nodes, edge_sources, edge_targets, edge_attr = [], [], [], []
        
        carrier_pos = carrier['pos'] * flip
        carrier_vel = carrier['vel'] * flip
        nodes.append([carrier_pos[0], carrier_pos[1], carrier_vel[0], carrier_vel[1], 1.0])
        
        visual_edges = []
        for i, p in enumerate(players):
            if p['id'] == carrier['id']: continue
            is_teammate = 1.0 if p['team'] == carrier['team'] else 0.0
            norm_p_pos = p['pos'] * flip
            norm_p_vel = p['vel'] * flip
            
            vec = carrier_pos - norm_p_pos
            dist = np.linalg.norm(vec)
            if dist == 0: dist = 1e-9
            
            closing_speed = np.dot(norm_p_vel, vec / dist)
            tti = dist / closing_speed if closing_speed > 0 else 10.0
            
            if tti <= 3.5 or dist <= 15.0:
                nodes.append([norm_p_pos[0], norm_p_pos[1], norm_p_vel[0], norm_p_vel[1], is_teammate])
                edge_sources.append(len(nodes) - 1)
                edge_targets.append(0)
                edge_attr.append([dist, min(tti, 10.0), closing_speed, is_teammate])
                if not is_teammate:
                    visual_edges.append({'pos': p['pos'], 'tti': tti})

        if len(edge_sources) == 0: return 0.99, []

        x = torch.tensor(nodes, dtype=torch.float).to(self.device)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long).to(self.device)
        attr = torch.tensor(edge_attr, dtype=torch.float).to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x, edge_index, attr, batch)
            prob = torch.sigmoid(logits).item()
        return prob, visual_edges

    def export_clip(self, df, start_frame, num_frames, output_path, scenario_name):
        print(f"Exporting '{scenario_name}' ({num_frames} frames) to JSON...")
        
        scenario_data = {
            "metadata": {
                "name": scenario_name,
                "start_frame": start_frame,
                "num_frames": num_frames,
                "fps": 25.0
            },
            "frames": []
        }

        for i in range(num_frames):
            frame_idx = start_frame + i
            if frame_idx >= len(df): break
            frame = df.iloc[frame_idx]
            current_time_sec = i / 25.0
            
            frame_payload = {
                "frame_idx": int(frame_idx),
                "time_sec": current_time_sec,
                "status": "dead_ball"
            }
            
            if not np.isnan(frame['Home_Ball_X']):
                players, ball = self.builder.extract_nodes_from_frame(frame)
                if players:
                    carrier = self.builder.identify_carrier(players, ball)
                    direction = self.builder.get_play_direction(frame, carrier['team'])
                    
                    frame_payload["ball"] = ball
                    frame_payload["players"] = players
                    
                    if np.linalg.norm(carrier['pos'] - ball) > 2.0:
                        frame_payload["status"] = "loose_ball"
                    else:
                        prob, edges = self.get_survival_probability(players, carrier, direction)
                        frame_payload["status"] = "possession"
                        frame_payload["carrier"] = carrier
                        frame_payload["prob"] = prob
                        frame_payload["edges"] = edges
            
            scenario_data["frames"].append(frame_payload)

        # Save to the React public directory
        with open(output_path, 'w') as f:
            json.dump(scenario_data, f, cls=NumpyEncoder)
            
        print(f"Success! Saved payload to {output_path}")

if __name__ == "__main__":
    exporter = ScenarioExporter("../models/ccsnet_best_model.pth")
    
    parser = MetricaParser()
    clean_df = parser.merge_and_clean(
        "../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv", 
        "../data/raw/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"
    )
    
    momentum_df = KinematicsEngine().add_kinematics(clean_df)
    
    # 1. Baseline Phase
    exporter.export_clip(momentum_df, start_frame=65000, num_frames=220, 
                         output_path="../dashboard/public/data/scenario_midfield_possession.json", 
                         scenario_name="Sustained Midfield Possession")

    # 2. The Goal 
    exporter.export_clip(momentum_df, start_frame=12102, num_frames=220, 
                         output_path="../dashboard/public/data/scenario_attacking_goal.json", 
                         scenario_name="Attacking Phase & Finish")

    # 3. Defensive Constriction
    exporter.export_clip(momentum_df, start_frame=42200, num_frames=180, 
                         output_path="../dashboard/public/data/scenario_high_press.json", 
                         scenario_name="The High-Press Trap")

    # 4. Set Piece Dynamics (Corner Kick)
    exporter.export_clip(momentum_df, start_frame=10438, num_frames=200, 
                         output_path="../dashboard/public/data/scenario_corner_kick.json", 
                         scenario_name="Set Piece Dynamics")

    # 5. NEW: The Masterclass Build-Up (18 Passes)
    exporter.export_clip(momentum_df, start_frame=3190, num_frames=1250, 
                         output_path="../dashboard/public/data/scenario_masterclass.json", 
                         scenario_name="The Masterclass Build-Up")

    # 6. Tight Spaces
    exporter.export_clip(momentum_df, start_frame=93200, num_frames=200, 
                         output_path="../dashboard/public/data/scenario_tight_spaces.json", 
                         scenario_name="Possession in Tight Spaces")

    # 7. Dead Ball Edge Case
    exporter.export_clip(momentum_df, start_frame=115000, num_frames=220, 
                         output_path="../dashboard/public/data/scenario_penalty_kick.json", 
                         scenario_name="The Penalty Kick")